import numpy as np
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Turns a grid of physical fields into a sequence of tokens."""

    def __init__(self, in_channels, patch=32, dim=256):
        super().__init__()
        # stride == kernel size, so each window covers one tile and no tile twice
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        # x arrives as (batch, channels, height, width)
        x = self.proj(x)         # -> (batch, dim, 16, 16)
        x = x.flatten(2)         # -> (batch, dim, 256)
        x = x.transpose(1, 2)    # -> (batch, 256, dim)
        return x


class BiasedAttention(nn.Module):
    """Self-attention with an additive bias on the scores."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.qkv = nn.Linear(dim, dim * 3)   # one layer makes query, key and value
        self.out = nn.Linear(dim, dim)

    def forward(self, x, bias=None):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        scores = q @ k.transpose(-2, -1) / self.dim ** 0.5   # (batch, tokens, tokens)
        if bias is not None:
            scores = scores + bias                           # physics goes in here
        weights = scores.softmax(dim=-1)
        return self.out(weights @ v)


class MultiHeadBiasedAttention(nn.Module):
    """Self-attention with several heads and one shared additive bias."""

    def __init__(self, dim, heads=4):
        super().__init__()
        assert dim % heads == 0, "dim must divide evenly by heads"
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def split(self, t):
        b, n, _ = t.shape
        # (batch, tokens, dim) -> (batch, heads, tokens, head_dim)
        return t.view(b, n, self.heads, self.head_dim).transpose(1, 2)

    def forward(self, x, bias=None):
        b, n, _ = x.shape
        q, k, v = (self.split(t) for t in self.qkv(x).chunk(3, dim=-1))
        scores = q @ k.transpose(-2, -1) / self.head_dim ** 0.5
        if bias is not None:
            scores = scores + bias   # already (b, heads, n, n)
        weights = scores.softmax(dim=-1)
        merged = (weights @ v).transpose(1, 2).reshape(b, n, -1)
        return self.out(merged)


class Block(nn.Module):
    """Attention then a small per-token MLP, each with a residual and pre-norm."""

    def __init__(self, dim, heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadBiasedAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x, bias=None):
        x = x + self.attn(self.norm1(x), bias)   # bias travels down to attention
        x = x + self.mlp(self.norm2(x))
        return x


class RelPosMLP(nn.Module):
    """Relative position bias from a small net on (dr, dc). Any grid, any offset."""

    def __init__(self, heads=4, hidden=32, scale=16.0):
        super().__init__()
        self.scale = scale
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, heads),
        )

    def forward(self, grid_h, grid_w, device=None):
        rows = torch.arange(grid_h, device=device).repeat_interleave(grid_w)
        cols = torch.arange(grid_w, device=device).repeat(grid_h)
        dr = (rows[:, None] - rows[None, :]).float() / self.scale
        dc = (cols[:, None] - cols[None, :]).float() / self.scale
        off = torch.stack([dr, dc], dim=-1)   # (n, n, 2)
        return self.net(off).permute(2, 0, 1)  # (heads, n, n)


class HydraulicTransformer(nn.Module):
    """Patch embed, N biased blocks, then predict depth change per patch."""

    def __init__(self, in_channels, depth=6, dim=192, heads=4, patch=32):
        super().__init__()
        self.patch = patch
        self.embed = PatchEmbed(in_channels, patch=patch, dim=dim)
        self.relpos = RelPosMLP(heads=heads)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, patch * patch)

    def forward(self, x, tau_bias=None):
        b, _, h, w = x.shape
        gh, gw = h // self.patch, w // self.patch      # grid read from the input
        bias = self.relpos(gh, gw, x.device).unsqueeze(0)
        if tau_bias is not None:
            bias = bias + tau_bias.unsqueeze(1)
        t = self.embed(x)
        for blk in self.blocks:
            t = blk(t, bias)
        t = self.head(self.norm(t))
        t = t.view(b, gh, gw, self.patch, self.patch)
        t = t.permute(0, 1, 3, 2, 4).reshape(b, 1, gh * self.patch, gw * self.patch)
        return t


class PhysicsBias(nn.Module):
    """Phi = -alpha * tau_norm + beta * S.

    Convention: scores[i, j] is receiver i reading source j, so the gate keys on
    the COLUMN. S_ij opens when the water surface at source j, eta_j = h_j + b_j,
    overtops the barrier B_ij. B is symmetric; all directionality comes from eta.
    S = 1 open, 0 blocked, so beta > 0 rewards open pairs.
    """

    def __init__(self, dt_seconds, tau_clamp=10.0, eps=0.25):
        super().__init__()
        self.dt, self.tau_clamp = dt_seconds, tau_clamp
        self.a = nn.Parameter(torch.zeros(1))          # alpha = softplus(a) >= 0
        self.beta = nn.Parameter(torch.zeros(1))
        # store the inverse of softplus so eps() starts exactly at the given value
        self.e = nn.Parameter(torch.tensor([float(np.log(np.expm1(eps)))]))

    def alpha(self):
        return nn.functional.softplus(self.a)

    def eps(self):
        return nn.functional.softplus(self.e)

    def forward(self, tau, eta, barrier, hard=False):
        tau_n = (tau / self.dt).clamp(max=self.tau_clamp)
        head = eta[:, None, :] - barrier               # source = column
        gate = (head > 0).float() if hard else torch.sigmoid(head / self.eps())
        return -self.alpha() * tau_n + self.beta * gate
