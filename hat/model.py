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
            scores = scores + bias.unsqueeze(1)   # (b,1,n,n) copies to every head
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
