"""Self-check: is this clone actually able to do what the README says?

Runs on a FRESH clone, before `prepare` and before `train`. It needs no corpus,
no query cache and no network — only what ships in the repository.

Seven checks, each of which has a way to fail:

    1  environment    every import the pipeline makes, with versions
    2  config         the seven pre-registered gate thresholds are the published
                      ones and the two resolutions are still 128
    3  checkpoint     loads, builds, parameter count matches, and its `model_cfg`
                      agrees with config.yaml — a checkpoint whose architecture
                      has drifted from the config loads fine and decodes garbage
    4  grid           the checkpoint carries the training bbox it was fitted to
    5  examples       5 STL/cloud pairs, matched stems, (n_points, 6) clouds with
                      no zero-norm normal, all inside the training bbox
    6  frame          each example STL is already at centroid ~0, max extent ~1,
                      i.e. the frame `prepare` would put it in
    7  decode         ONE example is actually decoded at R=128 and the surface
                      must come out watertight and winding-consistent. This is the
                      only check that proves the weights, the grid and the
                      extractor agree; the other six prove files are present.

Exit code is 0 only if every check passes.

    python main.py verify
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pc2mesh.common import REPO_ROOT, load_config, load_mesh, padded_bounds, resolve  # noqa: E402

DEFAULT_CKPT = "checkpoints/pc2mesh_v3.pt"
EXPECTED_GATES = {
    "watertight_rate": (0.98, 0.90, "higher"),
    "winding_consistent_rate": (0.98, 0.90, "higher"),
    "self_intersection_free_rate": (1.00, 0.90, "higher"),
    "iou128_mean": (0.90, 0.80, "higher"),
    "chamfer_l2_x1e3_mean": (1.00, 2.00, "lower"),
    "normal_consistency_mean": (0.90, 0.80, "higher"),
    "euler_match_rate": (0.85, 0.70, "higher"),
}
N_EXAMPLES = 5
FRAME_TOL = 1e-5     # float32 STL round-trip, not a measurement tolerance


class Checks:
    def __init__(self):
        self.failed = []

    def section(self, name):
        print(f"\n--- {name} ---")

    def ok(self, label, detail=""):
        print(f"  ok    {label}" + (f"  {detail}" if detail else ""))

    def fail(self, label, detail=""):
        self.failed.append(label)
        print(f"  FAIL  {label}" + (f"  {detail}" if detail else ""))

    def check(self, cond, label, detail=""):
        (self.ok if cond else self.fail)(label, detail)
        return bool(cond)


def check_environment(c: Checks) -> None:
    c.section("1  environment")
    mods = [("numpy", None), ("torch", None), ("trimesh", None), ("skimage", None),
            ("scipy", None), ("yaml", "__version__"), ("igl", None), ("tqdm", None)]
    for name, attr in mods:
        try:
            m = __import__(name)
            v = getattr(m, attr or "__version__", "(no __version__)")
            c.ok(f"import {name}", str(v))
        except Exception as e:
            c.fail(f"import {name}", f"{type(e).__name__}: {e}")
    import torch

    c.ok("torch device", "cuda: " + (torch.cuda.get_device_name(0)
                                     if torch.cuda.is_available()
                                     else "not available — CPU is fine, just slower"))
    # Optional, and reported as such: tets are a --tets flag, not a requirement.
    for name in ("gmsh", "meshio", "tetgen", "wildmeshing"):
        try:
            m = __import__(name)
            v = getattr(m, "__version__", None) or getattr(m, "GMSH_API_VERSION", "?")
            c.ok(f"optional {name}", str(v))
        except Exception:
            print(f"  --    optional {name}  not installed (only needed for --tets)")


def check_config(c: Checks, cfg) -> None:
    c.section("2  config — the pre-registered thresholds")
    gates = dict(cfg.eval.gates)
    if not c.check(set(gates) == set(EXPECTED_GATES), "the seven gates are registered",
                   f"{set(gates) ^ set(EXPECTED_GATES) or 'exactly the published set'}"):
        return
    for k, (p, pa, d) in EXPECTED_GATES.items():
        g = gates[k]
        good = (float(g["pass"]) == p and float(g["partial"]) == pa
                and str(g["direction"]) == d)
        op = ">=" if d == "higher" else "<="
        c.check(good, f"{k:30s} {op} {p}",
                "" if good else f"EDITED: {dict(g)} != pass {p} / partial {pa} / {d}")
    c.check(int(cfg.eval.iou_resolution) == 128, "eval.iou_resolution == 128",
            str(cfg.eval.iou_resolution))
    c.check(int(cfg.meshify.resolution) == 128, "meshify.resolution == 128",
            str(cfg.meshify.resolution))


def check_checkpoint(c: Checks, cfg, ckpt_path: Path, device: str):
    c.section("3  checkpoint")
    if not c.check(ckpt_path.exists(), f"{ckpt_path.relative_to(REPO_ROOT)} exists"):
        return None, None
    import torch

    from pc2mesh.model import build_model, count_params

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    size_mb = ckpt_path.stat().st_size / 1e6
    c.ok("loads", f"{size_mb:.1f} MB, step {ck.get('step', '?')}, "
                  f"val BCE {ck.get('val_bce', float('nan')):.4f}")

    model_cfg = ck.get("model_cfg")
    if not c.check(model_cfg is not None, "carries its own model_cfg"):
        return None, ck
    drift = {k: (v, dict(cfg.model).get(k)) for k, v in model_cfg.items()
             if dict(cfg.model).get(k) != v}
    c.check(not drift, "model_cfg agrees with config.yaml",
            "" if not drift else f"DRIFT (checkpoint, config): {drift}")

    model = build_model(model_cfg).to(device)
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    c.check(not missing and not unexpected, "state dict fits the architecture",
            "" if not (missing or unexpected)
            else f"missing {list(missing)[:3]}, unexpected {list(unexpected)[:3]}")
    n = count_params(model)
    c.check(n == ck.get("n_params", n), "parameter count matches the checkpoint",
            f"{n:,} ({n/1e6:.2f}M)")
    model.eval()
    return model, ck


def check_grid(c: Checks, cfg, ck) -> np.ndarray | None:
    c.section("4  decode grid")
    if ck is None:
        c.fail("training bbox available")
        return None
    gb = ck.get("global_bounds")
    if not c.check(gb is not None, "checkpoint carries the training bbox",
                   "" if gb is not None else
                   "a checkpoint without it falls back to data/global_bounds.json, "
                   "which `prepare` rewrites for whatever corpus is on disk"):
        return None
    b = np.array([gb["min"], gb["max"]], dtype=np.float64)
    pad = float(cfg.meshify.bbox_pad)
    grid = padded_bounds(b, pad)
    c.ok("training bbox", f"{np.round(b[0], 4).tolist()} .. {np.round(b[1], 4).tolist()}")
    c.ok(f"decode grid ({int(cfg.meshify.resolution)}^3, padded {pad:.0%})",
         f"{np.round(grid[0], 4).tolist()} .. {np.round(grid[1], 4).tolist()}")
    return b


def check_examples(c: Checks, cfg, train_bounds):
    c.section("5  examples")
    from pc2mesh.dataset import unit_normal_cloud
    from pc2mesh.infer import bbox_check

    ex = REPO_ROOT / "examples"
    stls = sorted((ex / "stl").glob("*.stl"))
    npys = sorted((ex / "clouds").glob("*.npy"))
    c.check(len(stls) == N_EXAMPLES, f"examples/stl holds {N_EXAMPLES} meshes",
            f"{len(stls)} found")
    c.check(len(npys) == N_EXAMPLES, f"examples/clouds holds {N_EXAMPLES} clouds",
            f"{len(npys)} found")
    c.check({p.stem for p in stls} == {p.stem for p in npys},
            "every mesh has a matching cloud",
            "" if {p.stem for p in stls} == {p.stem for p in npys}
            else f"unmatched: {({p.stem for p in stls} ^ {p.stem for p in npys})}")

    n_points = int(cfg.model.n_points)
    for p in npys:
        arr = np.load(p)
        if not c.check(arr.ndim == 2 and arr.shape == (n_points, 6),
                       f"{p.stem}: cloud is ({n_points}, 6)", str(arr.shape)):
            continue
        try:
            pc6 = unit_normal_cloud(arr[:, :3], arr[:, 3:6], strict=True)
        except ValueError as e:
            c.fail(f"{p.stem}: normals are unit-normalizable", str(e)[:90])
            continue
        bbox = np.stack([pc6[:, :3].min(0), pc6[:, :3].max(0)])
        if train_bounds is None:
            c.ok(f"{p.stem}: normals unit-normalizable", "(bbox not checked: no grid)")
            continue
        inside, overflow, _ = bbox_check(bbox, train_bounds)
        c.check(inside, f"{p.stem}: inside the training bbox",
                f"extent {np.round(bbox[1] - bbox[0], 3).tolist()}" if inside
                else f"outside by {overflow:.4f}")
    return stls


def check_frame(c: Checks, stls) -> None:
    c.section("6  frame of the shipped meshes")
    for p in stls:
        m = load_mesh(p, process=True)
        cen = float(np.abs(np.asarray(m.centroid)).max())
        ext = float(np.max(m.extents))
        good = cen < 1e-3 and abs(ext - 1.0) < 1e-3
        c.check(good, f"{p.stem}: centroid at origin, max extent 1",
                f"|centroid| {cen:.2e}, max extent {ext:.6f}")


def check_decode(c: Checks, cfg, model, train_bounds, device: str, skip: bool) -> None:
    c.section("7  end-to-end decode")
    if skip:
        print("  --    skipped (--no-decode)")
        return
    if model is None or train_bounds is None:
        c.fail("decode one example", "no usable checkpoint or grid")
        return
    from pc2mesh.dataset import unit_normal_cloud
    from pc2mesh.meshify import meshify_one
    from pc2mesh.remesh import flags, remesh_one

    npys = sorted((REPO_ROOT / "examples" / "clouds").glob("*.npy"))
    if not npys:
        c.fail("decode one example", "no example clouds")
        return
    # The smallest file, so the check is as cheap as it can be while still being
    # a real decode of real data at the shipped resolution.
    p = min(npys, key=lambda q: q.stat().st_size)
    arr = np.load(p)
    pc6 = unit_normal_cloud(arr[:, :3], arr[:, 3:6], strict=True)
    grid = padded_bounds(train_bounds, float(cfg.meshify.bbox_pad))
    res = int(cfg.meshify.resolution)

    t0 = time.time()
    mesh = meshify_one(model, pc6, cfg, grid, device, drop_floaters=True, resolution=res)
    dt = time.time() - t0
    if not c.check(len(mesh.faces) > 0, f"{p.stem}: marching cubes found a level set",
                   f"{len(mesh.faces)} faces at R={res} in {dt:.1f}s on {device}"):
        return
    wt, wc = flags(mesh)
    c.check(wt, f"{p.stem}: marching-cubes surface is watertight")
    c.check(wc, f"{p.stem}: marching-cubes surface is winding-consistent")

    target = int((cfg.get("infer") or {}).get("target_faces") or cfg.remesh.target_faces)
    final, failure = remesh_one(mesh, target)
    c.check(failure is None, f"{p.stem}: decimation to {target} faces held the invariant",
            f"{len(mesh.faces)} -> {len(final.faces)} faces" if failure is None
            else f"reverted at '{failure['step']}': {failure['reason']}")
    wt2, wc2 = flags(final)
    c.check(wt2 and wc2, f"{p.stem}: emitted surface is watertight and winding-consistent",
            f"{len(final.faces)} faces")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default=None, help="cuda | cpu")
    ap.add_argument("--no-decode", action="store_true",
                    help="skip check 7 (the only slow one)")
    args = ap.parse_args()

    import torch

    cfg = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.ckpt) if args.ckpt else resolve(DEFAULT_CKPT)

    print("================ PC2MESH VERIFY ================")
    print(f"repo   : {REPO_ROOT}")
    print(f"device : {device}")

    c = Checks()
    check_environment(c)
    check_config(c, cfg)
    model, ck = check_checkpoint(c, cfg, ckpt_path, device)
    train_bounds = check_grid(c, cfg, ck)
    stls = check_examples(c, cfg, train_bounds)
    check_frame(c, stls)
    check_decode(c, cfg, model, train_bounds, device, args.no_decode)

    print("\n================================================")
    if c.failed:
        print(f"VERIFY: {len(c.failed)} CHECK(S) FAILED")
        for f in c.failed:
            print(f"  - {f}")
        print("================================================")
        return 1
    print("VERIFY: ALL CHECKS PASS")
    print("================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
