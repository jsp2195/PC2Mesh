"""Stage 3 — occupancy network: point cloud -> latent set -> per-query logit.

    encoder:  x (B,N,3|6) -> FPS M centres -> Fourier(xyz)[++ normal] + Linear ->
              cross-attn(q=centres, kv=all points) -> n_layers PreLN self-attn
              -> LN  =>  z (B,M,d)
    decoder:  p (B,Q,3) -> Fourier+Linear -> cross-attn(kv=z) -> +MLP -> 1 logit

`model.use_normals` switches the encoder input between 3 and 6 channels. With 6,
xyz is Fourier-encoded exactly as before and the raw 3-vector normal is
concatenated to those features before the single linear projection to d_model —
the normal is deliberately NOT Fourier-encoded: it is a direction, not a
position, and has no high-frequency spatial structure to unfold.

The flag is read with a default of False, so a checkpoint trained before normals
existed still loads and reproduces its own numbers (`model_cfg` travels inside
the .pt).

Pure torch throughout; FPS is a plain tensor loop, no custom CUDA op.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- FPS


def farthest_point_sample(xyz: torch.Tensor, m: int) -> torch.Tensor:
    """Indices of `m` farthest-point samples. xyz: (B,N,3) -> (B,m) long.

    Deterministic: always seeded from index 0. Batched, pure torch.
    """
    B, N, _ = xyz.shape
    m = min(m, N)
    idx = torch.zeros(B, m, dtype=torch.long, device=xyz.device)
    dist = torch.full((B, N), float("inf"), device=xyz.device, dtype=xyz.dtype)
    far = torch.zeros(B, dtype=torch.long, device=xyz.device)
    ar = torch.arange(B, device=xyz.device)
    for i in range(m):
        idx[:, i] = far
        d = ((xyz - xyz[ar, far].unsqueeze(1)) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        far = dist.argmax(-1)
    return idx


def gather_points(xyz: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """xyz (B,N,C), idx (B,m) -> (B,m,C)."""
    return torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, xyz.size(-1)))


# --------------------------------------------------------------------------- blocks


class FourierEmbed(nn.Module):
    """[sin, cos] over log-spaced frequencies, per axis, then a linear map to d.

    `extra_dim` raw channels beyond the first `in_dim` are appended to the
    Fourier features unchanged and projected by the same linear layer. That is
    how the encoder takes normals: Fourier(xyz) ++ n, one Linear to d_model.
    With extra_dim=0 this class is bit-identical to its previous form, which is
    what lets the pre-normals checkpoint keep loading.
    """

    def __init__(self, n_freq: int, d_model: int, max_freq: float = 64.0,
                 in_dim: int = 3, extra_dim: int = 0):
        super().__init__()
        freqs = 2.0 ** torch.linspace(0.0, math.log2(max_freq), n_freq)
        self.register_buffer("freqs", freqs, persistent=False)
        self.in_dim = in_dim
        self.extra_dim = extra_dim
        self.out_dim = in_dim * n_freq * 2 + extra_dim
        self.proj = nn.Linear(self.out_dim, d_model)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim + extra_dim) -> (..., in_dim*n_freq*2 + extra_dim)
        pos = x[..., : self.in_dim]
        a = pos.unsqueeze(-1) * self.freqs.to(x.dtype) * (2.0 * math.pi)
        f = torch.cat([a.sin(), a.cos()], dim=-1).flatten(-2)
        if self.extra_dim:
            f = torch.cat([f, x[..., self.in_dim: self.in_dim + self.extra_dim]], dim=-1)
        return f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.features(x))


class CrossAttn(nn.Module):
    """PreLN multi-head cross-attention with a residual connection."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.nq = nn.LayerNorm(d_model)
        self.nk = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        k = self.nk(kv)
        out, _ = self.attn(self.nq(q), k, k, need_weights=False)
        return q + self.drop(out)


class Mlp(nn.Module):
    def __init__(self, d_model: int, ratio: int, dropout: float):
        super().__init__()
        h = d_model * ratio
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, h)
        self.fc2 = nn.Linear(h, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(self.drop(F.gelu(self.fc1(self.norm(x)))))
        return x + self.drop(h)


class SelfAttnBlock(nn.Module):
    """PreLN self-attention + MLP."""

    def __init__(self, d_model: int, n_heads: int, ratio: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.mlp = Mlp(d_model, ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        return self.mlp(x + self.drop(a))


# --------------------------------------------------------------------------- model


class PointEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = int(cfg["d_model"])
        self.n_centers = int(cfg["n_centers"])
        self.use_normals = bool(cfg.get("use_normals", False))
        self.in_ch = 6 if self.use_normals else 3
        self.embed = FourierEmbed(int(cfg["n_freq"]), d, float(cfg["fourier_max_freq"]),
                                  in_dim=3, extra_dim=3 if self.use_normals else 0)
        self.cross = CrossAttn(d, int(cfg["n_heads"]), float(cfg["dropout"]))
        self.blocks = nn.ModuleList([
            SelfAttnBlock(d, int(cfg["n_heads"]), int(cfg["mlp_ratio"]), float(cfg["dropout"]))
            for _ in range(int(cfg["n_layers"]))
        ])
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,N,3) or (B,N,6) = xyz [++ unit normal].

        A normals model handed a 3-channel cloud raises rather than silently
        reconstructing from geometry alone — that failure would look like a bad
        run, not like a plumbing bug.
        """
        if self.use_normals and x.shape[-1] != 6:
            raise ValueError(f"model.use_normals is set but the cloud has "
                             f"{x.shape[-1]} channels; expected 6 (xyz + normal)")
        x = x[..., : self.in_ch]
        centers = gather_points(x, farthest_point_sample(x[..., :3], self.n_centers))
        h = self.cross(self.embed(centers), self.embed(x))
        for b in self.blocks:
            h = b(h)
        return self.norm(h)


class OccDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = int(cfg["d_model"])
        self.embed = FourierEmbed(int(cfg["n_freq"]), d, float(cfg["fourier_max_freq"]))
        self.cross = CrossAttn(d, int(cfg["n_heads"]), float(cfg["dropout"]))
        self.mlp = Mlp(d, int(cfg["mlp_ratio"]), float(cfg["dropout"]))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        h = self.cross(self.embed(p), z)
        h = self.mlp(h)
        return self.head(self.norm(h)).squeeze(-1)  # (B,Q) logits


class OccNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = PointEncoder(cfg)
        self.decoder = OccDecoder(cfg)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, p)

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x), p)


def build_model(cfg) -> OccNet:
    return OccNet(dict(cfg))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from pc2mesh.common import load_config

    cfg = load_config()
    m = build_model(cfg.model)
    n = count_params(m)
    print(f"total params: {n:,} ({n/1e6:.2f}M)")
    print(f"  encoder: {count_params(m.encoder):,}")
    print(f"  decoder: {count_params(m.decoder):,}")
    ch = m.encoder.in_ch
    x = torch.randn(2, cfg.model.n_points, ch) * 0.3
    if ch == 6:
        x[..., 3:] = torch.nn.functional.normalize(x[..., 3:], dim=-1)
    p = torch.randn(2, 4096, 3) * 0.3
    with torch.no_grad():
        print(f"encoder input channels: {ch}")
        print("logits:", m(x, p).shape)
