"""Does a blocked pair actually receive less attention than an open one?

Everything so far proves the plumbing runs. This proves the mechanism bites:
two patches at equal distance from a receiver, one behind a barrier, one not.
"""
import torch
from hat.model import PhysicsBias, MultiHeadBiasedAttention

N, DIM, HEADS = 16, 192, 4
RECEIVER, OPEN, BLOCKED = 0, 1, 2


def build():
    """Identical distance, identical depth. Only the barrier differs."""
    tau = torch.full((1, N, N), 3600.0)
    eta = torch.full((1, N), 1.0)
    barrier = torch.zeros(N, N)
    barrier[RECEIVER, BLOCKED] = 5.0    # source column 2 is walled off from 0
    return tau, eta, barrier


def weights(phi):
    torch.manual_seed(0)
    attn = MultiHeadBiasedAttention(DIM, HEADS)
    x = torch.randn(1, N, DIM)
    q, k, _ = (attn.split(t) for t in attn.qkv(x).chunk(3, dim=-1))
    scores = q @ k.transpose(-2, -1) / attn.head_dim ** 0.5
    if phi is not None:
        scores = scores + phi.unsqueeze(1)
    return scores.softmax(dim=-1).mean(dim=1)[0]   # average over heads


if __name__ == "__main__":
    tau, eta, barrier = build()
    p = PhysicsBias(dt_seconds=7200.0)
    p.beta.data.fill_(2.0)

    base = weights(None)
    gated = weights(p(tau, eta, barrier))

    print(f"no bias   open {base[RECEIVER, OPEN]:.4f}  blocked {base[RECEIVER, BLOCKED]:.4f}")
    print(f"with phi  open {gated[RECEIVER, OPEN]:.4f}  blocked {gated[RECEIVER, BLOCKED]:.4f}")
    print("blocked loses attention:",
          gated[RECEIVER, BLOCKED] < gated[RECEIVER, OPEN])
    print("blocked drops vs no bias:",
          gated[RECEIVER, BLOCKED] < base[RECEIVER, BLOCKED])
    print("rows still sum to 1:", torch.allclose(gated.sum(-1), torch.ones(N), atol=1e-5))
