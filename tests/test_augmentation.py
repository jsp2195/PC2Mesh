"""The augmentation invariants, tested against igl rather than against themselves.

Two things here are silent when wrong.

1. QUERIES. If rotation/scale are applied to the point cloud but not identically
   to the queries, the labels stop describing the shape the encoder sees and the
   run quietly learns mush. Tested by transforming the GT mesh by the same affine
   and re-running the winding number.

2. NORMALS. Positions transform as p -> R S p; normals are covectors and
   transform as n -> normalize(R S^-1 n). Using S instead of S^-1 is wrong but
   *nearly* right: it is exactly right for isotropic S, and exactly right for any
   axis-aligned normal at any S, because S and S^-1 both scale a basis vector and
   the renormalization erases the magnitude. The error only appears on off-axis
   normals, and even there it is about twice the anisotropy -- roughly 4 degrees
   at the +/-10% configured here. Tested by recomputing normals from the
   transformed mesh with igl.

Both have negative controls, and a control that cannot fail proves nothing. That
constrains how the normal control is measured: at +/-10% anisotropy the wrong
rule still scores cos = 0.998 against the truth, so a ">0.99 mean cosine" bar is
passed by the bug as comfortably as by the fix. The discriminating measurement is
therefore made where the two rules actually differ -- points whose stored normal
IS the local face normal AND which are genuinely off-axis -- and only there is
the exactness bar applied. The specified >0.99 statistic is reported too, so both
are on the record.

EVERYTHING THIS FILE NEEDS SHIPS WITH THE REPOSITORY. The mesh/cloud pairs come
from examples/, the occupancy labels are computed here with the same recipe
`prep_queries` uses, and the batching test builds a small cache in a temporary
directory. Nothing has to be prepared or trained first, so these tests run on a
fresh clone -- which is the only condition under which they are worth running.

Run:  python tests/test_augmentation.py     (or: python -m pytest tests/ -q)
"""
from __future__ import annotations

import copy
import functools
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from pc2mesh.common import (  # noqa: E402
    REPO_ROOT, Cfg, load_config, load_mesh, padded_bounds, repair_for_labeling, resolve,
    stable_seed,
)
from pc2mesh.dataset import (  # noqa: E402
    OccData, apply_affine, apply_affine_normals, dropout_and_pad, make_rotations,
    random_rotations, random_yaw_tilt_rotations, rotate_mode, unit_normal_cloud,
)

CFG = load_config()
DEV = "cpu"
EXAMPLES = REPO_ROOT / "examples"
N_QUERIES = 6000          # queries per shape in the label tests
CACHE_QUERIES = 8000      # queries per shape in the temporary cache

# --- normal-transform test parameters, unchanged from the reference run
NRM_WELL_DEFINED = 0.99  # cos(stored normal, local face normal) above this = usable
NRM_OFFAXIS = 0.2        # 2nd-largest |component| above this = genuinely off-axis
NRM_EXACT = 0.9999       # the bar the correct rule clears and the wrong one must not
NRM_MIN_POOL = 50        # below this the control could pass vacuously -> hard error


# --------------------------------------------------------------------- fixtures


def _affine_np(pts, rot, scale):
    """numpy mirror of apply_affine, for transforming mesh vertices."""
    return (pts * scale) @ rot.T


@functools.lru_cache(maxsize=1)
def _stems() -> tuple[str, ...]:
    stems = tuple(sorted(p.stem for p in (EXAMPLES / "clouds").glob("*.npy")))
    assert stems, f"no example clouds under {EXAMPLES / 'clouds'}"
    for s in stems:
        assert (EXAMPLES / "stl" / f"{s}.stl").exists(), f"{s}: no matching example mesh"
    return stems


@functools.lru_cache(maxsize=None)
def _mesh_for(stem: str):
    """The example mesh, put through the SAME labeling repair prep_queries uses."""
    rep, _, ok = repair_for_labeling(load_mesh(EXAMPLES / "stl" / f"{stem}.stl",
                                               process=True))
    assert ok, f"{stem}: example mesh is not labelable after fill_holes+fix_normals"
    return rep


@functools.lru_cache(maxsize=None)
def _cloud_for(stem: str) -> np.ndarray:
    """(N,6) float32 xyz ++ UNIT normal, through the one function that defines it."""
    arr = np.load(EXAMPLES / "clouds" / f"{stem}.npy")
    return unit_normal_cloud(arr[:, :3], arr[:, 3:6], strict=True)


@functools.lru_cache(maxsize=1)
def _bounds() -> np.ndarray:
    """The padded global bbox, from the shipped checkpoint's own training bbox.

    Taking it from the checkpoint rather than from data/global_bounds.json is what
    lets these tests run before `prepare` has ever been called.
    """
    ck = torch.load(resolve("checkpoints/pc2mesh_v3.pt"), map_location="cpu",
                    weights_only=False)
    gb = ck["global_bounds"]
    return padded_bounds(np.array([gb["min"], gb["max"]], dtype=np.float64),
                         float(CFG.prep.bbox_pad))


@functools.lru_cache(maxsize=None)
def _queries_for(stem: str, n_queries: int):
    """(queries, occ) with prep_queries' recipe: near_frac near-surface, rest uniform.

    Labels come from igl's fast generalized winding number on the repaired mesh,
    at the f16-rounded coordinates that would actually be stored -- the model must
    never be supervised at a coordinate it is not shown.
    """
    import igl
    import trimesh

    gb = _bounds()
    mesh = _mesh_for(stem)
    rng = np.random.default_rng(stable_seed(stem))
    n_near = int(round(n_queries * float(CFG.prep.near_frac)))
    n_unif = n_queries - n_near

    surf, _ = trimesh.sample.sample_surface(mesh, n_near, seed=int(rng.integers(1 << 31)))
    sig = np.asarray(CFG.prep.near_sigmas, dtype=np.float64)
    which = rng.integers(0, len(sig), size=n_near)
    q_near = np.asarray(surf, dtype=np.float64) + rng.normal(size=(n_near, 3)) * sig[which][:, None]
    q_unif = rng.uniform(gb[0], gb[1], size=(n_unif, 3))

    q = np.clip(np.concatenate([q_near, q_unif], 0).astype(np.float16).astype(np.float64),
                gb[0], gb[1])
    w = igl.fast_winding_number(
        np.ascontiguousarray(mesh.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh.faces, dtype=np.int64),
        np.ascontiguousarray(q))
    occ = (w > float(CFG.prep.wn_threshold)).astype(np.uint8)
    return q.astype(np.float16), occ, n_near


@functools.lru_cache(maxsize=1)
def _tiny_cache():
    """A real query cache for the example shapes, in a temp dir, plus a config
    pointing at it. Same .npz layout `prep_queries` writes, so OccData is exercised
    as it is in training and not through a stub."""
    tmp = tempfile.mkdtemp(prefix="pc2mesh_test_cache_")
    for stem in _stems():
        q, occ, n_near = _queries_for(stem, CACHE_QUERIES)
        pc = _cloud_for(stem)
        np.savez(Path(tmp) / f"{stem}.npz",
                 pc=pc[:, :3].astype(np.float16),
                 pc_normals=pc[:, 3:6].astype(np.float16),
                 queries=q, occ=occ, n_near=np.int32(n_near),
                 bbox=_bounds(), stem=np.array(stem), folder=np.array("examples"))
    cfg = Cfg(copy.deepcopy(dict(CFG)))
    cfg["paths"]["cache"] = tmp
    return cfg


# --------------------------------------------------------------------- rotations


def test_rotation_is_a_rotation():
    g = torch.Generator(device=DEV).manual_seed(0)
    R = random_rotations(256, g, DEV)
    eye = torch.eye(3).expand(256, 3, 3)
    assert torch.allclose(torch.bmm(R, R.transpose(1, 2)), eye, atol=1e-5), "not orthogonal"
    assert torch.allclose(torch.linalg.det(R), torch.ones(256), atol=1e-5), "det != +1"
    # and they actually cover SO(3) rather than clustering near identity
    assert R[:, 0, 0].min() < -0.5 and R[:, 0, 0].max() > 0.5, "rotations not diverse"
    print("  ok  random_rotations: orthogonal, det=+1, diverse")


def test_yaw_tilt_rotation():
    """The restricted rotation must actually be restricted, and still be a rotation."""
    g = torch.Generator(device=DEV).manual_seed(11)
    tilt_deg = float(CFG.train.augment.get("tilt_deg", 15.0))
    up = int(CFG.train.augment.get("up_axis", 2))
    R = random_yaw_tilt_rotations(512, g, DEV, tilt_deg=tilt_deg, up_axis=up)
    eye = torch.eye(3).expand(512, 3, 3)
    assert torch.allclose(torch.bmm(R, R.transpose(1, 2)), eye, atol=1e-5), "not orthogonal"
    assert torch.allclose(torch.linalg.det(R), torch.ones(512), atol=1e-5), "det != +1"

    e = torch.zeros(3)
    e[up] = 1.0
    moved = torch.rad2deg(torch.arccos((R @ e).matmul(e).clamp(-1, 1)))
    assert moved.max() <= tilt_deg + 1e-3, (
        f"up axis tilted {moved.max():.2f} deg, above the {tilt_deg} deg budget")
    # and the yaw really does sweep the full circle, or this is not an augmentation
    horiz = [(up + 1) % 3, (up + 2) % 3]
    v = R @ torch.eye(3)[horiz[0]]
    ang = torch.atan2(v[:, horiz[1]], v[:, horiz[0]])
    assert ang.min() < -2.5 and ang.max() > 2.5, "yaw does not cover the full circle"
    print(f"  ok  yaw_tilt: orthogonal, det=+1, up axis within {moved.max():.2f} deg "
          f"of vertical (budget {tilt_deg}), yaw covers the circle")


def test_rotate_mode_is_read_and_unknown_modes_raise():
    """A typo in `rotate_mode` must stop the run, not silently mean 'none'."""
    g = torch.Generator(device=DEV).manual_seed(2)
    assert rotate_mode(CFG.train.augment) == str(CFG.train.augment.rotate_mode)
    assert rotate_mode({"rotate": True}) == "so3", "the pre-rotate_mode fallback changed"
    assert rotate_mode({"rotate": False}) == "none"
    ident = make_rotations("none", 4, g, DEV)
    assert torch.allclose(ident, torch.eye(3).expand(4, 3, 3)), "'none' is not identity"
    try:
        make_rotations("yaw-tilt", 4, g, DEV)   # a plausible typo
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown rotate_mode was accepted instead of raising")
    print("  ok  rotate_mode: config value read, legacy bool honoured, typos raise")


def test_dropout_and_pad():
    g = torch.Generator(device=DEV).manual_seed(0)
    pc = torch.randn(8, 1000, 3, generator=g)
    out = dropout_and_pad(pc, 0.9, 1.0, g)
    assert out.shape == pc.shape, "point count must be padded back to N"
    for b in range(8):
        src = {tuple(np.round(v, 6)) for v in pc[b].numpy()}
        got = {tuple(np.round(v, 6)) for v in out[b].numpy()}
        assert got <= src, "padding invented points that were not in the input"
        assert 0.85 * 1000 <= len(got) <= 1000, f"kept {len(got)} unique points"
    print("  ok  dropout_and_pad: subset of input, count preserved, 90-100% kept")


# --------------------------------------------------------------------- labels


def test_labels_survive_augmentation():
    """THE test: augmented queries against the augmented mesh give the same labels."""
    import igl

    g = torch.Generator(device=DEV).manual_seed(1234)
    worst = 1.0
    stems = _stems()
    for stem in stems:
        q16, occ, _ = _queries_for(stem, N_QUERIES)
        q = torch.from_numpy(q16.astype(np.float32))
        mesh = _mesh_for(stem)
        V = np.asarray(mesh.vertices, dtype=np.float64)
        F = np.asarray(mesh.faces, dtype=np.int64)

        for trial in range(3):
            rot = random_rotations(1, g, DEV)
            scale = 1.0 + (torch.rand(1, 3, generator=g) * 2 - 1) * float(
                CFG.train.augment.scale_aniso)
            qa = apply_affine(q.unsqueeze(0), rot, scale)[0].double().numpy()
            Va = _affine_np(V, rot[0].double().numpy(), scale[0].double().numpy())

            w = igl.fast_winding_number(np.ascontiguousarray(Va), F,
                                        np.ascontiguousarray(qa))
            occ_a = (w > float(CFG.prep.wn_threshold)).astype(np.uint8)
            agree = float((occ_a == occ).mean())
            worst = min(worst, agree)
            assert agree > 0.999, (
                f"{stem} trial {trial}: labels changed under augmentation "
                f"(agreement {agree:.4f}) -- rotation/scale are NOT identical "
                f"between the point cloud and the queries")
    print(f"  ok  labels survive rotation+scale on {len(stems)} shapes x3 trials "
          f"(worst agreement {worst:.5f})")


def test_negative_control():
    """Mismatched transforms MUST break the labels, or the test above is vacuous."""
    import igl

    stem = _stems()[0]
    q16, occ, _ = _queries_for(stem, N_QUERIES)
    q = torch.from_numpy(q16.astype(np.float32))
    mesh = _mesh_for(stem)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)

    g = torch.Generator(device=DEV).manual_seed(7)
    rot_q = random_rotations(1, g, DEV)
    rot_mesh = random_rotations(1, g, DEV)  # the bug: two different rotations
    one = torch.ones(1, 3)
    qa = apply_affine(q.unsqueeze(0), rot_q, one)[0].double().numpy()
    Va = _affine_np(V, rot_mesh[0].double().numpy(), one[0].double().numpy())
    w = igl.fast_winding_number(np.ascontiguousarray(Va), F, np.ascontiguousarray(qa))
    agree = float(((w > 0.5).astype(np.uint8) == occ).mean())
    assert agree < 0.999, (
        "mismatched rotations did NOT change the labels -- the invariant test is "
        "not sensitive to the bug it exists to catch")
    print(f"  ok  negative control: mismatched rotation breaks labels "
          f"(agreement {agree:.4f})")


# --------------------------------------------------------------------- normals


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _closest_face_normals(V, F, P):
    """Unit normal of the mesh face nearest each point -- 'the mesh's own normal'."""
    import igl

    _, I, _ = igl.point_mesh_squared_distance(
        np.ascontiguousarray(P, dtype=np.float64),
        np.ascontiguousarray(V, dtype=np.float64),
        np.ascontiguousarray(F, dtype=np.int64))
    tri = V[F[I]]
    return _unit(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))


_POOL = {}


def _normal_pool():
    """Both rules evaluated in ONE pass, so they cannot differ by anything else.

    The scale is the deterministic corner (1+a, 1, 1-a) of the configured
    +/-`scale_aniso` box: a legitimate draw, reproducible, and the most sensitive
    one -- a random draw can come out near-isotropic, where no measurement could
    tell the two rules apart.
    """
    if _POOL:
        return _POOL
    stems = _stems()
    a = float(CFG.train.augment.scale_aniso)
    scale = torch.tensor([[1.0 + a, 1.0, 1.0 - a]], dtype=torch.float32)
    g = torch.Generator(device=DEV).manual_seed(4321)

    wide, sharp_ok, sharp_bad = [], [], []
    for stem in stems:
        pc = _cloud_for(stem)
        P = pc[:, :3].astype(np.float64)
        N = pc[:, 3:6].astype(np.float64)
        mesh = _mesh_for(stem)
        V = np.asarray(mesh.vertices, dtype=np.float64)
        F = np.asarray(mesh.faces, dtype=np.int64)

        # which stored normals ARE the mesh's normal, and which are off-axis
        a0 = (_closest_face_normals(V, F, P) * N).sum(1)
        well = a0 > NRM_WELL_DEFINED
        offaxis = np.sort(np.abs(N), axis=1)[:, 1] > NRM_OFFAXIS
        strict = (a0 > 1 - 1e-4) & offaxis

        rot = random_rotations(1, g, DEV)
        R = rot[0].double().numpy()
        S = scale[0].double().numpy()
        Pa = apply_affine(torch.from_numpy(P).float().unsqueeze(0), rot, scale)[0].double().numpy()
        truth = _closest_face_normals((V * S) @ R.T, F, Pa)   # igl, from the moved mesh

        got = apply_affine_normals(torch.from_numpy(N).float().unsqueeze(0),
                                   rot, scale)[0].double().numpy()
        bug = _unit((N * S) @ R.T)                            # S instead of S^-1

        wide.append((truth * got).sum(1)[well])
        sharp_ok.append((truth * got).sum(1)[strict])
        sharp_bad.append((truth * bug).sum(1)[strict])

    _POOL.update(n_shapes=len(stems),
                 wide=np.concatenate(wide),
                 ok=np.concatenate(sharp_ok),
                 bad=np.concatenate(sharp_bad))
    return _POOL


def test_normals_survive_augmentation():
    """POSITIVE: transformed cloud normals match normals recomputed from the mesh."""
    p = _normal_pool()
    wide, ok = p["wide"], p["ok"]
    assert len(wide) > 1000, f"only {len(wide)} usable points -- test is not meaningful"
    assert wide.mean() > 0.99, (
        f"transformed cloud normals disagree with igl's recomputation from the "
        f"transformed mesh (mean cos {wide.mean():.5f} <= 0.99)")
    assert ok.mean() >= NRM_EXACT and ok.min() > 0.999, (
        f"the normal transform is not exact on off-axis normals: mean cos "
        f"{ok.mean():.6f}, min {ok.min():.6f} -- expected >= {NRM_EXACT}")
    print(f"  ok  normals survive rotation+aniso scale on {p['n_shapes']} shapes: "
          f"mean cos {wide.mean():.5f} over {len(wide)} well-defined points "
          f"(>0.99); exact to {ok.mean():.6f} on the {len(ok)} off-axis points")


def test_normal_negative_control():
    """NEGATIVE: S in place of S^-1 MUST fail the bar the correct rule clears.

    Restricted to off-axis normals on purpose. For an axis-aligned normal the two
    rules are provably identical after renormalization, so including those points
    would dilute the control toward passing -- i.e. would make it unable to fail.
    """
    p = _normal_pool()
    bad, ok = p["bad"], p["ok"]
    assert len(bad) >= NRM_MIN_POOL, (
        f"only {len(bad)} off-axis points pooled (need {NRM_MIN_POOL}); the "
        f"control could pass for lack of evidence rather than for being right")
    assert bad.mean() < NRM_EXACT, (
        f"applying S instead of S^-1 to the normals did NOT break them "
        f"(mean cos {bad.mean():.6f}) -- the positive test above is vacuous")
    assert bad.mean() < ok.mean(), "the wrong rule scored no worse than the right one"
    frac = float(np.mean(bad > NRM_EXACT))
    print(f"  ok  negative control: S instead of S^-1 gives mean cos "
          f"{bad.mean():.6f} vs {ok.mean():.6f} correct; only {frac:.1%} of the "
          f"{len(bad)} off-axis points still clear {NRM_EXACT}")


def test_normal_test_can_fail():
    """META: inject the S-for-S^-1 bug and require the POSITIVE test to fail.

    The negative control shows the bug is detectable in principle. This shows the
    test that guards the shipped code actually catches it, which is the claim that
    matters and the one that rots silently if a threshold is ever loosened.
    """
    def buggy(nrm, rot, scale):
        n = torch.bmm(nrm * scale.unsqueeze(1), rot.transpose(1, 2))   # S, not S^-1
        return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    g = globals()
    real, saved = g["apply_affine_normals"], dict(_POOL)
    try:
        g["apply_affine_normals"] = buggy
        _POOL.clear()
        try:
            test_normals_survive_augmentation()
        except AssertionError:
            pass
        else:
            raise AssertionError(
                "the positive normal test PASSES with the S-for-S^-1 bug injected "
                "-- it cannot fail, so it proves nothing")
    finally:
        g["apply_affine_normals"] = real
        _POOL.clear()
        _POOL.update(saved)
    print("  ok  meta: injecting S-for-S^-1 makes the positive normal test fail")


def test_normals_reach_the_output():
    """The 6th channel must change the prediction, or it is decoration.

    Cheap guard against the whole plumbing failure class -- normals sliced off,
    projected through a zero block, or shadowed by a stale config -- which would
    look exactly like "normals did not help" in the gates.
    """
    from pc2mesh.model import build_model

    g = torch.Generator(device=DEV).manual_seed(3)
    x6 = torch.randn(2, 256, 6, generator=g) * 0.3
    x6[..., 3:] = x6[..., 3:] / x6[..., 3:].norm(dim=-1, keepdim=True)
    p = torch.randn(2, 512, 3, generator=g) * 0.3
    flipped = x6.clone()
    flipped[..., 3:] = -flipped[..., 3:]

    def spread(use_normals):
        cfg = dict(CFG.model)
        cfg["use_normals"] = use_normals
        torch.manual_seed(0)
        m = build_model(cfg).eval()
        with torch.no_grad():
            a, b = m(x6, p), m(flipped, p)
        return ((a - b).abs().mean() / a.abs().mean().clamp_min(1e-9)).item()

    on, off = spread(True), spread(False)
    assert on > 1e-3, (f"flipping every input normal moved the output by only {on:.2e} "
                       f"relative -- the normal channel is not reaching the decoder")
    assert off < 1e-9, (f"a use_normals=False model reacted to the normal channel "
                        f"({off:.2e}) -- it should never see columns 3:6")
    print(f"  ok  normals reach the output: flipping them moves it {on:.3f} relative "
          f"with use_normals on, {off:.1e} with it off")


# --------------------------------------------------------------------- batching


def test_batch_shapes_and_determinism():
    cfg = _tiny_cache()
    data = OccData(cfg, _stems(), device=DEV, show_progress=False)
    g1 = torch.Generator(device=DEV).manual_seed(5)
    g2 = torch.Generator(device=DEV).manual_seed(5)
    idx = torch.arange(4)
    a = data.sample_batch(idx, 2048, g1, augment=True)
    b = data.sample_batch(idx, 2048, g2, augment=True)
    for x, y, name in zip(a, b, ("pc", "queries", "occ")):
        assert torch.equal(x, y), f"{name} is not reproducible from the same seed"
    pc, q, occ = a
    assert pc.shape == (4, int(CFG.model.n_points), 6), pc.shape
    assert q.shape == (4, 2048, 3) and occ.shape == (4, 2048)
    # the normals ride along as UNIT vectors after augmentation
    nn_ = pc[..., 3:6].norm(dim=-1)
    assert torch.allclose(nn_, torch.ones_like(nn_), atol=1e-3), \
        f"normals are not unit after augmentation (min {nn_.min():.4f}, max {nn_.max():.4f})"
    # the near/uniform split follows prep.near_frac, and near-surface queries are
    # far more often inside than uniform ones
    k = data.n_near_in_batch(2048)
    assert k == int(round(2048 * float(CFG.prep.near_frac))), k
    near_rate = occ[:, :k].mean().item()
    unif_rate = occ[:, k:].mean().item()
    assert near_rate > unif_rate, "near/uniform blocks look swapped"
    print(f"  ok  batching deterministic; {k}/{2048} near ({float(CFG.prep.near_frac):.0%}); "
          f"occ near {near_rate:.3f} / uniform {unif_rate:.3f}; normals unit")


if __name__ == "__main__":
    fails = 0
    for fn in (test_rotation_is_a_rotation, test_yaw_tilt_rotation,
               test_rotate_mode_is_read_and_unknown_modes_raise, test_dropout_and_pad,
               test_labels_survive_augmentation, test_negative_control,
               test_normals_survive_augmentation, test_normal_negative_control,
               test_normal_test_can_fail, test_normals_reach_the_output,
               test_batch_shapes_and_determinism):
        try:
            fn()
        except AssertionError as e:
            fails += 1
            print(f"  FAIL  {fn.__name__}: {e}")
    print("\nAUGMENTATION TESTS:", "ALL PASS" if not fails else f"{fails} FAILED")
    raise SystemExit(1 if fails else 0)
