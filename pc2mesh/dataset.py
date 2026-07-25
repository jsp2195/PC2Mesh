"""Dataset + augmentation.

The whole query cache is ~0.74 GB, so it is held resident on the GPU as f16 and
batches are assembled by indexing. No DataLoader, no worker processes, no I/O in
the training loop, and the sampling is exactly reproducible from a torch.Generator.

The augmentation contract, and the single most important invariant in the project:

    rotation and anisotropic scale are applied IDENTICALLY to the point cloud and
    to the query points, so the occupancy labels remain correct.

    jitter and point dropout are input corruptions and touch the point cloud ONLY;
    moving a query would move it relative to its own label.

Normals do NOT follow that same rule, and getting it wrong is silent. Positions
transform as p -> R S p; normals are covectors and transform as

    n -> normalize(R S^-1 n)

which equals R S n only when S is isotropic. Under the +/-10% per-axis scale used
here, applying S instead of S^-1 tilts every off-axis normal the wrong way by
roughly twice the anisotropy while leaving axis-aligned normals untouched, so the
mean cosine error stays small enough to hide. tests/test_augmentation.py checks
the transform against normals recomputed from the transformed mesh by igl, and
carries a negative control that applies S n and must fail.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pc2mesh.common import resolve


# --------------------------------------------------------------------------- transforms


def random_rotations(n: int, generator: torch.Generator, device) -> torch.Tensor:
    """(n,3,3) uniform random rotations, via uniformly sampled unit quaternions."""
    q = torch.randn(n, 4, generator=generator, device=device)
    q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ], dim=1).view(n, 3, 3)


def _axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues. axis (n,3) need not be unit; angle (n,) radians -> (n,3,3)."""
    a = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-12)
    x, y, z = a.unbind(1)
    c, s = angle.cos(), angle.sin()
    C = 1 - c
    return torch.stack([
        c + x * x * C, x * y * C - z * s, x * z * C + y * s,
        y * x * C + z * s, c + y * y * C, y * z * C - x * s,
        z * x * C - y * s, z * y * C + x * s, c + z * z * C,
    ], dim=1).view(-1, 3, 3)


def random_yaw_tilt_rotations(n: int, generator: torch.Generator, device,
                              tilt_deg: float = 15.0, up_axis: int = 2) -> torch.Tensor:
    """Yaw uniformly about `up_axis`, then tilt by up to `tilt_deg` off vertical.

    The corpus is upright: across the 1041 cached shapes the x and y extent
    distributions are the closest pair (KS 0.095 vs 0.13/0.17 against z) and z is
    the shortest axis on average, i.e. objects are wider than tall and the two
    horizontal axes are exchangeable. Rotating freely in SO(3) therefore asks the
    encoder to learn an invariance the held-out data never exercises.
    """
    e = torch.zeros(3, device=device)
    e[up_axis] = 1.0
    up = e.expand(n, 3)

    yaw = torch.rand(n, generator=generator, device=device) * (2 * math.pi)
    r_yaw = _axis_angle_to_matrix(up, yaw)

    # tilt about a uniformly random axis in the horizontal plane
    phi = torch.rand(n, generator=generator, device=device) * (2 * math.pi)
    h = torch.zeros(n, 3, device=device)
    ax = [(up_axis + 1) % 3, (up_axis + 2) % 3]
    h[:, ax[0]] = phi.cos()
    h[:, ax[1]] = phi.sin()
    amp = math.radians(float(tilt_deg))
    tilt = (torch.rand(n, generator=generator, device=device) * 2 - 1) * amp
    r_tilt = _axis_angle_to_matrix(h, tilt)
    return torch.bmm(r_tilt, r_yaw)


def rotate_mode(aug) -> str:
    """`augment.rotate_mode`, falling back to the older boolean `augment.rotate`.

    The fallback exists so a config written before yaw_tilt existed still means
    what it meant then (rotate: true == full SO(3)) rather than silently becoming
    a different augmentation.
    """
    m = aug.get("rotate_mode")
    if m is not None:
        return str(m)
    return "so3" if bool(aug.get("rotate", False)) else "none"


def make_rotations(mode: str, n: int, generator: torch.Generator, device,
                   tilt_deg: float = 15.0, up_axis: int = 2) -> torch.Tensor:
    """`mode` in {none, so3, yaw_tilt}. Unknown modes raise rather than default."""
    if mode == "none":
        return torch.eye(3, device=device).expand(n, 3, 3).contiguous()
    if mode == "so3":
        return random_rotations(n, generator, device)
    if mode == "yaw_tilt":
        return random_yaw_tilt_rotations(n, generator, device, tilt_deg, up_axis)
    raise ValueError(f"unknown augment.rotate_mode {mode!r}; "
                     f"expected one of none / so3 / yaw_tilt")


def apply_affine(pts: torch.Tensor, rot: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Anisotropic scale in the object frame, then rotate.

    pts (B,K,3), rot (B,3,3), scale (B,3) -> (B,K,3). This exact function is used
    for both the point cloud and the queries; that is what keeps labels valid.
    """
    return torch.bmm(pts * scale.unsqueeze(1), rot.transpose(1, 2))


def apply_affine_normals(nrm: torch.Tensor, rot: torch.Tensor,
                         scale: torch.Tensor) -> torch.Tensor:
    """Normals under the p -> R S p of `apply_affine`: n -> normalize(R S^-1 n).

    The inverse scale is the whole point. S is diagonal so S^-1 is an elementwise
    divide; using `* scale` here instead would be correct only for isotropic S and
    is the exact bug tests/test_augmentation.py's negative control applies.
    """
    n = torch.bmm(nrm / scale.unsqueeze(1), rot.transpose(1, 2))
    return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def dropout_and_pad(pc: torch.Tensor, keep_min: float, keep_max: float,
                    generator: torch.Generator) -> torch.Tensor:
    """Keep a random 90-100% of the points, then pad back to N by resampling.

    Channel-agnostic: whatever rides along with xyz (a normal, here) is gathered
    with the same indices, so a point never gets another point's normal.
    """
    B, N, C = pc.shape
    dev = pc.device
    r = torch.rand(B, 1, generator=generator, device=dev)
    k = (N * (keep_min + (keep_max - keep_min) * r)).long().clamp(1, N)  # (B,1)

    perm = torch.argsort(torch.rand(B, N, generator=generator, device=dev), dim=1)
    ar = torch.arange(N, device=dev).unsqueeze(0).expand(B, N)
    fill = (torch.rand(B, N, generator=generator, device=dev) * k).long().clamp(max=N - 1)
    pos = torch.where(ar < k, ar, fill)
    idx = torch.gather(perm, 1, pos)
    return torch.gather(pc, 1, idx.unsqueeze(-1).expand(-1, -1, C))


# --------------------------------------------------------------------------- data


def unit_normal_cloud(pc, nrm, strict: bool = True) -> np.ndarray:
    """(N,6) float32: xyz ++ UNIT normal, from separate position and normal arrays.

    THE definition of "a cloud" for this project. Both the cache reader
    (`cloud_from_npz`) and inference on a bare .npy go through this one function,
    so training and inference cannot disagree about what the encoder is handed.
    See `cloud_from_npz` for why the magnitude is discarded and why a stored
    normal of exactly zero is an error rather than a silence.
    """
    pc = np.asarray(pc, dtype=np.float32)
    nr = np.asarray(nrm, dtype=np.float32)
    if nr.shape != pc.shape:
        raise ValueError(f"positions are {pc.shape} but normals are {nr.shape}")
    norm = np.linalg.norm(nr, axis=1, keepdims=True)
    if strict:
        n_zero = int((norm < 1e-6).sum())
        if n_zero:
            raise ValueError(
                f"{n_zero} of {len(nr)} stored normals have norm < 1e-6 and cannot be "
                f"put on the unit sphere; this cloud would reach the encoder off the "
                f"manifold every other cloud is on. Fix the cloud or exclude the shape "
                f"— do not pass strict=False to make it quiet.")
    return np.concatenate([pc, nr / np.maximum(norm, 1e-12)], axis=1)


def cloud_from_npz(d, strict: bool = True) -> np.ndarray:
    """(N,6) float32: xyz ++ UNIT normal, from one loaded cache .npz.

    Clouds from different sources store normals differently. Interpolated VERTEX
    normals are short on an edge, where the stored vector is the average of two
    faces; the FACE normals `pc2mesh.prepare` writes are already unit. Magnitude
    carries no geometry, and the augmented branch renormalizes after S^-1, so
    normalizing here is what keeps the augmented (train) and un-augmented
    (inference) inputs on the same manifold — and, across sources, what stops
    normal magnitude from being a free domain label the encoder could read.

    Every consumer goes through this one function so train and inference cannot
    disagree about what a cloud is.

    THE ONE INPUT THIS CANNOT NORMALIZE is a stored normal of exactly zero:
    x / max(‖x‖, 1e-12) leaves it at zero, i.e. off the unit sphere and silently
    different from every other point. In the reference corpus 11 of 1250 clouds
    contained such points (10 of them entirely); all 11 were open shells that the
    watertight gate already rejects, so none ever entered a cache. `strict` makes
    that an error instead of a silence if one ever does.
    """
    return unit_normal_cloud(d["pc"], d["pc_normals"], strict=strict)


def load_cloud(cache_dir, stem) -> np.ndarray:
    """(N,6) float32 cloud for `stem` from the query cache."""
    return cloud_from_npz(np.load(Path(cache_dir) / f"{stem}.npz", allow_pickle=True))


class OccData:
    """GPU-resident query cache for one split.

    The cloud is always held as 6 channels (xyz ++ unit normal) whether or not the
    model consumes normals, so there is a single augmentation path to keep
    correct. The encoder slices back to 3 when `model.use_normals` is off.
    """

    def __init__(self, cfg, stems, device="cuda", show_progress=True):
        self.cfg = cfg
        self.device = device
        self.stems = list(stems)
        cache = resolve(cfg.paths.cache)
        n_points = int(cfg.model.n_points)

        pcs, qs, occs = [], [], []
        n_near = None
        it = tqdm(self.stems, desc="load cache", disable=not show_progress)
        for s in it:
            d = np.load(cache / f"{s}.npz", allow_pickle=True)
            pc = cloud_from_npz(d)               # (N,6): xyz ++ unit normal
            assert pc.shape[0] >= n_points, f"{s}: only {pc.shape[0]} points"
            pcs.append(pc.astype(np.float16))
            qs.append(d["queries"])
            occs.append(d["occ"])
            nn_ = int(d["n_near"])
            assert n_near in (None, nn_), "inconsistent n_near across the cache"
            n_near = nn_
        self.n_near = n_near
        self.n_stored_points = pcs[0].shape[0]
        assert all(p.shape == pcs[0].shape for p in pcs), "ragged point clouds"

        self.pc = torch.from_numpy(np.stack(pcs)).to(device)          # (S,N,6) f16
        self.q = torch.from_numpy(np.stack(qs)).to(device)            # (S,Q,3) f16
        self.occ = torch.from_numpy(np.stack(occs)).to(device)        # (S,Q)   u8
        self.n_queries = self.q.shape[1]
        self.n_unif = self.n_queries - self.n_near
        self._qflat = self.q.view(-1, 3)
        self._oflat = self.occ.view(-1)

    def __len__(self):
        return len(self.stems)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.pc, self.q, self.occ))

    # ------------------------------------------------------------------ batching

    def n_near_in_batch(self, n_queries: int) -> int:
        """How many of a batch's queries come from the near-surface pool.

        Read from the SAME `prep.near_frac` that decided the cache composition, so
        the two cannot drift apart: the cache being 75% near-surface would buy
        nothing if the sampler still drew the two pools 50/50.
        """
        return int(round(n_queries * float(self.cfg.prep.near_frac)))

    def sample_batch(self, shape_idx: torch.Tensor, n_queries: int,
                     generator: torch.Generator, augment: bool):
        """-> pc (B,N,6) f32, queries (B,n_queries,3) f32, occ (B,n_queries) f32.

        Queries are split near-surface / uniform in `prep.near_frac`, matching the
        cache's own composition.
        """
        cfg = self.cfg
        dev = self.device
        B = shape_idx.numel()
        n_near_b = min(max(self.n_near_in_batch(n_queries), 0), n_queries)

        near = torch.randint(0, self.n_near, (B, n_near_b), generator=generator, device=dev)
        unif = self.n_near + torch.randint(0, self.n_unif, (B, n_queries - n_near_b),
                                           generator=generator, device=dev)
        qi = torch.cat([near, unif], dim=1) + shape_idx.unsqueeze(1) * self.n_queries
        q = self._qflat[qi.reshape(-1)].view(B, n_queries, 3).float()
        occ = self._oflat[qi.reshape(-1)].view(B, n_queries).float()

        pc = self.pc[shape_idx].float()          # (B,N,6)
        C = pc.shape[-1]
        n_points = int(cfg.model.n_points)
        if self.n_stored_points > n_points:
            # free augmentation when the cloud is denser than the encoder input
            sel = torch.argsort(torch.rand(B, self.n_stored_points, generator=generator,
                                           device=dev), dim=1)[:, :n_points]
            pc = torch.gather(pc, 1, sel.unsqueeze(-1).expand(-1, -1, C))

        if not augment:
            return pc, q, occ

        a = cfg.train.augment
        rot = make_rotations(rotate_mode(a), B, generator, dev,
                             tilt_deg=float(a.get("tilt_deg", 15.0)),
                             up_axis=int(a.get("up_axis", 2)))
        s = float(a.scale_aniso)
        scale = 1.0 + (torch.rand(B, 3, generator=generator, device=dev) * 2 - 1) * s

        xyz = apply_affine(pc[..., :3], rot, scale)
        q = apply_affine(q, rot, scale)          # IDENTICAL transform -> labels hold
        # positions get S, normals get S^-1 -- see apply_affine_normals
        nrm = apply_affine_normals(pc[..., 3:6], rot, scale)

        # input-only corruptions
        if float(a.jitter_sigma) > 0:
            xyz = xyz + torch.randn(xyz.shape, generator=generator,
                                    device=dev) * float(a.jitter_sigma)
        pc = torch.cat([xyz, nrm], dim=-1)
        if float(a.dropout_keep_min) < 1.0:
            pc = dropout_and_pad(pc, float(a.dropout_keep_min), float(a.dropout_keep_max),
                                 generator)
        return pc, q, occ

    def fixed_val_batches(self, n_batches: int, batch_shapes: int, n_queries: int, seed: int):
        """Deterministic, un-augmented val batches (canonical pose = inference pose)."""
        g = torch.Generator(device=self.device).manual_seed(seed)
        order = torch.randperm(len(self), generator=g, device=self.device)
        out = []
        for b in range(n_batches):
            sl = order[b * batch_shapes:(b + 1) * batch_shapes]
            if sl.numel() == 0:
                break
            out.append(self.sample_batch(sl, n_queries, g, augment=False))
        return out


def load_split(cfg):
    with open(resolve(cfg.paths.data) / "split.json") as f:
        return json.load(f)
