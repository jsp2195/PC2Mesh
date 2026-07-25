"""Geometry metrics used by the audit and the evaluation gates.

Nothing here touches torch — these are CPU/numpy routines so they can run inside
a multiprocessing.Pool.

Self-intersection counting is implemented from scratch (vectorized Moller
triangle-triangle overlap over a uniform-grid broad phase) because no collision
backend (python-fcl) is available offline.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- occupancy


def winding_occupancy(mesh, pts: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Inside/outside via igl's fast generalized winding number.

    Robust to the small holes and inconsistent windings that survive repair,
    which `mesh.contains` (ray parity) is not.
    """
    import igl

    V = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    F = np.ascontiguousarray(mesh.faces, dtype=np.int64)
    Q = np.ascontiguousarray(pts, dtype=np.float64)
    w = igl.fast_winding_number(V, F, Q)
    return (w > threshold).astype(np.uint8)


def grid_points(bounds: np.ndarray, res: int) -> np.ndarray:
    """Voxel-centre grid over `bounds`, C-ordered as (res, res, res) flattened."""
    lo, hi = np.asarray(bounds, dtype=np.float64)
    axes = [np.linspace(lo[d], hi[d], res) for d in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return g.reshape(-1, 3)


def grid_iou(mesh_a, mesh_b, bounds: np.ndarray, res: int, threshold: float = 0.5) -> float:
    """IoU of the two solids sampled on a shared res^3 grid."""
    q = grid_points(bounds, res)
    a = winding_occupancy(mesh_a, q, threshold).astype(bool)
    b = winding_occupancy(mesh_b, q, threshold).astype(bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(a, b).sum() / union)


# --------------------------------------------------------------------------- sampling


def sample_surface(mesh, n: int, seed: int):
    """Area-weighted surface samples + the face normal at each sample."""
    import trimesh

    rng = np.random.default_rng(seed)
    pts, fid = trimesh.sample.sample_surface(mesh, n, seed=int(rng.integers(1 << 31)))
    nrm = np.asarray(mesh.face_normals)[fid]
    return np.asarray(pts, dtype=np.float64), np.asarray(nrm, dtype=np.float64)


def chamfer_l2(pa: np.ndarray, pb: np.ndarray) -> float:
    """Symmetric mean SQUARED nearest-neighbour distance (pre-registered form).

    chamfer = 0.5 * ( mean_x min_y |x-y|^2  +  mean_y min_x |x-y|^2 )
    Reported by the gates scaled by 1e3.
    """
    from scipy.spatial import cKDTree

    ta, tb = cKDTree(pa), cKDTree(pb)
    dab, _ = tb.query(pa, k=1, workers=-1)
    dba, _ = ta.query(pb, k=1, workers=-1)
    return float(0.5 * ((dab ** 2).mean() + (dba ** 2).mean()))


def normal_consistency(pa, na, pb, nb) -> float:
    """Symmetric mean |cos| between a point's normal and its nearest neighbour's.

    Absolute value, so a globally flipped prediction is not punished twice (the
    winding-consistency gate is what scores orientation).
    """
    from scipy.spatial import cKDTree

    ta, tb = cKDTree(pa), cKDTree(pb)
    _, iab = tb.query(pa, k=1, workers=-1)
    _, iba = ta.query(pb, k=1, workers=-1)

    def unit(v):
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(n, 1e-12)

    na, nb = unit(na), unit(nb)
    cab = np.abs(np.einsum("ij,ij->i", na, nb[iab])).mean()
    cba = np.abs(np.einsum("ij,ij->i", nb, na[iba])).mean()
    return float(0.5 * (cab + cba))


# --------------------------------------------------------------------------- self-intersection


def _broad_phase(tri: np.ndarray, max_insert: int = 40_000_000):
    """Uniform-grid broad phase. Returns a deduplicated (P,2) array of face pairs
    whose AABBs share a cell. Never misses a true intersection."""
    lo = tri.min(axis=1)
    hi = tri.max(axis=1)
    span = (hi - lo).max(axis=1)
    n = len(tri)
    domain = float(np.max(tri.max(axis=(0, 1)) - tri.min(axis=(0, 1))))
    cell = max(float(np.percentile(span, 99.0)), domain / 4096.0, 1e-9)

    origin = tri.min(axis=(0, 1))
    for _ in range(8):
        ilo = np.floor((lo - origin) / cell).astype(np.int64)
        ihi = np.floor((hi - origin) / cell).astype(np.int64)
        counts = np.prod(ihi - ilo + 1, axis=1)
        if counts.sum() <= max_insert:
            break
        cell *= 2.0
    else:
        raise RuntimeError("broad phase failed to fit in memory budget")

    # Expand each face into the cells its AABB touches.
    total = int(counts.sum())
    face_id = np.repeat(np.arange(n, dtype=np.int64), counts)
    # per-insertion offset within that face's cell block
    off = np.arange(total, dtype=np.int64) - np.repeat(
        np.concatenate([[0], np.cumsum(counts)[:-1]]), counts
    )
    dims = (ihi - ilo + 1)[face_id]
    dz = off % dims[:, 2]
    t = off // dims[:, 2]
    dy = t % dims[:, 1]
    dx = t // dims[:, 1]
    cx = ilo[face_id, 0] + dx
    cy = ilo[face_id, 1] + dy
    cz = ilo[face_id, 2] + dz

    # Hash cell coordinates to a single int64 key.
    key = (cx * np.int64(73856093)) ^ (cy * np.int64(19349663)) ^ (cz * np.int64(83492791))

    order = np.argsort(key, kind="stable")
    key, face_id = key[order], face_id[order]
    bounds = np.flatnonzero(np.concatenate([[True], key[1:] != key[:-1], [True]]))
    sizes = np.diff(bounds)

    out = []
    for s in np.unique(sizes):
        if s < 2:
            continue
        starts = bounds[:-1][sizes == s]
        members = face_id[starts[:, None] + np.arange(s)[None, :]]  # (G, s)
        ii, jj = np.triu_indices(s, k=1)
        out.append(np.stack([members[:, ii].ravel(), members[:, jj].ravel()], axis=1))
    if not out:
        return np.zeros((0, 2), dtype=np.int64)
    pairs = np.concatenate(out, axis=0)
    pairs = np.sort(pairs, axis=1)
    # A pair can appear in several shared cells; keep one copy.
    packed = pairs[:, 0] * np.int64(n) + pairs[:, 1]
    packed = np.unique(packed)
    return np.stack([packed // n, packed % n], axis=1)


def _coplanar_overlap(A: np.ndarray, B: np.ndarray, n1: np.ndarray) -> np.ndarray:
    """2D separating-axis test for coplanar triangle pairs."""
    drop = np.argmax(np.abs(n1), axis=1)
    keep = np.array([[1, 2], [0, 2], [0, 1]])[drop]  # (P,2) axes to keep
    idx = np.arange(len(A))[:, None, None]
    a2 = A[idx, np.arange(3)[None, :, None], keep[:, None, :]]  # (P,3,2)
    b2 = B[idx, np.arange(3)[None, :, None], keep[:, None, :]]

    sep = np.zeros(len(A), dtype=bool)
    for poly, other in ((a2, b2), (b2, a2)):
        for e in range(3):
            edge = poly[:, (e + 1) % 3] - poly[:, e]
            axis = np.stack([-edge[:, 1], edge[:, 0]], axis=1)  # (P,2)
            pa = np.einsum("pvd,pd->pv", poly, axis)
            pb = np.einsum("pvd,pd->pv", other, axis)
            sep |= (pa.min(1) > pb.max(1)) | (pb.min(1) > pa.max(1))
    return ~sep


def _tri_tri_overlap(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Vectorized Moller triangle-triangle overlap. A, B are (P,3,3)."""
    P = len(A)
    if P == 0:
        return np.zeros(0, dtype=bool)

    def plane(T):
        nrm = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
        return nrm, -np.einsum("pd,pd->p", nrm, T[:, 0])

    n2, d2 = plane(B)
    dv = np.einsum("pd,pvd->pv", n2, A) + d2[:, None]
    n1, d1 = plane(A)
    du = np.einsum("pd,pvd->pv", n1, B) + d1[:, None]

    # Scale-relative snapping so exactly-touching geometry is not called a crossing.
    sa = np.maximum(np.abs(dv).max(1), eps)[:, None]
    sb = np.maximum(np.abs(du).max(1), eps)[:, None]
    dv = np.where(np.abs(dv) < 1e-10 * sa, 0.0, dv)
    du = np.where(np.abs(du) < 1e-10 * sb, 0.0, du)

    alive = ~(((dv > 0).all(1) | (dv < 0).all(1)) | ((du > 0).all(1) | (du < 0).all(1)))
    coplanar = alive & (dv == 0).all(1)
    alive = alive & ~coplanar

    res = np.zeros(P, dtype=bool)
    if coplanar.any():
        res[coplanar] = _coplanar_overlap(A[coplanar], B[coplanar], n1[coplanar])
    if not alive.any():
        return res

    idx = np.flatnonzero(alive)
    D = np.cross(n1[idx], n2[idx])
    ax = np.argmax(np.abs(D), axis=1)
    pA = np.take_along_axis(A[idx], ax[:, None, None].repeat(3, 1), axis=2)[:, :, 0]
    pB = np.take_along_axis(B[idx], ax[:, None, None].repeat(3, 1), axis=2)[:, :, 0]

    def interval(p, d):
        """Moller's odd-vertex-out interval on the intersection line."""
        d0, d1_, d2_ = d[:, 0], d[:, 1], d[:, 2]
        odd = np.full(len(d), 0, dtype=np.int64)
        odd = np.where(d0 * d1_ > 0, 2, odd)
        m = (d0 * d1_ <= 0) & (d0 * d2_ > 0)
        odd = np.where(m, 1, odd)
        # remaining cases (d1*d2>0, or zeros) all reduce to vertex 0 unless it is
        # the one lying in the plane
        m2 = (d0 * d1_ <= 0) & (d0 * d2_ <= 0) & (d1_ * d2_ <= 0) & (d0 == 0)
        odd = np.where(m2 & (d1_ != 0), 1, odd)
        odd = np.where(m2 & (d1_ == 0) & (d2_ != 0), 2, odd)
        o = odd
        i = (o + 1) % 3
        j = (o + 2) % 3
        r = np.arange(len(d))
        po, pi, pj = p[r, o], p[r, i], p[r, j]
        do, di, dj = d[r, o], d[r, i], d[r, j]
        t1 = pi + (po - pi) * di / np.where(di - do == 0, 1e-300, di - do)
        t2 = pj + (po - pj) * dj / np.where(dj - do == 0, 1e-300, dj - do)
        return np.minimum(t1, t2), np.maximum(t1, t2)

    a0, a1 = interval(pA, dv[idx])
    b0, b1 = interval(pB, du[idx])
    res[idx] = (a1 >= b0) & (b1 >= a0)
    return res


def count_self_intersections(mesh, chunk: int = 400_000) -> int:
    """Number of non-adjacent face pairs that actually intersect.

    Faces sharing a vertex are excluded (they are meant to touch). Degenerate
    zero-area faces are excluded; they carry no surface.
    """
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    if len(F) == 0:
        return 0
    tri = V[F]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    good = np.flatnonzero(area2 > 1e-18)
    if len(good) < 2:
        return 0
    F, tri = F[good], tri[good]

    pairs = _broad_phase(tri)
    if len(pairs) == 0:
        return 0

    fi, fj = F[pairs[:, 0]], F[pairs[:, 1]]
    shares = np.zeros(len(pairs), dtype=bool)
    for a in range(3):
        for b in range(3):
            shares |= fi[:, a] == fj[:, b]
    pairs = pairs[~shares]

    total = 0
    for s in range(0, len(pairs), chunk):
        p = pairs[s : s + chunk]
        total += int(_tri_tri_overlap(tri[p[:, 0]], tri[p[:, 1]]).sum())
    return total
