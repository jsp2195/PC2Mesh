"""Inference — a directory of clouds or meshes in, watertight surfaces out.

There is NO ground truth at inference time, so nothing here prints IoU, Chamfer,
normal consistency or Euler match. Those are all GT-dependent and the only way to
show one would be to invent a reference. What IS reportable without a reference
is the set of topological invariants the extractor is supposed to guarantee, plus
whether the input looks like the data the checkpoint was trained on:

    watertight / winding consistent / self-intersections / face count / tet success
    WARNING when the normalized bbox falls outside the training range
    WARNING when the cloud does not carry exactly `model.n_points` points

Every step is flagged in the per-shape report so a bad output can be attributed:

    1  load        .stl -> mesh, or .npy -> (N,6) cloud
    2  normalize   STL only: trimesh area-weighted `.centroid` to the origin,
                   max extent scaled to 1.0 -- the convention that verifies on the
                   corpus. The bbox centre stated in the original brief holds for
                   1.8% of the 1250 legacy meshes and is off by up to 0.2589; it is
                   never used here, because normalizing an input differently from
                   the training data misframes it inside the grid.
    3  sample      STL only: `sample_surface_even` for exactly n_points, topped up
                   from `sample_surface` if the Poisson-disk cull came up short,
                   carrying the sampled FACE normal. `pc2mesh.prepare` calls the
                   same two functions, so a shape cannot be framed and sampled one
                   way in training and another way here.
    4  check       zero-norm normals REJECT the cloud (they cannot be put on the
                   unit sphere and would reach the encoder off the manifold every
                   other cloud is on). Nothing is repaired; the reason is recorded.
                   bbox vs the training range, and the point count, are warned on.
    5  meshify     R=128 over the padded training bbox, --drop-floaters
    6  decimate    quadric to 4000 faces, REVERT-ON-BREAK: a step that costs
                   watertightness or winding consistency reverts the whole shape
                   to the marching-cubes mesh. A broken mesh is never written.
    7  tets        optional, gmsh, per closed shell

The grid comes from the CHECKPOINT when it carries its own `global_bounds`, so a
released checkpoint keeps meshifying on the grid it was trained for even after
`prepare` has rewritten data/global_bounds.json for somebody else's corpus.

    python -m pc2mesh.infer --in examples/clouds --out out/ --tets
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pc2mesh.common import (  # noqa: E402
    load_config, load_mesh, padded_bounds, resolve, set_seed,
)

warnings.filterwarnings("ignore")

REPORT_FIELDS = [
    "stem", "input", "input_kind", "status", "reject_reason",
    "n_points", "src_ext_max", "scale", "tx", "ty", "tz",
    "bbox_min_x", "bbox_min_y", "bbox_min_z",
    "bbox_max_x", "bbox_max_y", "bbox_max_z",
    "bbox_in_training_range", "bbox_overflow_max", "point_count_expected",
    "mc_faces", "faces", "n_shells",
    "watertight", "winding_consistent", "self_intersections", "self_intersection_free",
    "decimate_reverted", "revert_step", "revert_reason",
    "tet_backend", "tet_ok", "n_tets", "n_tet_nodes", "tet_inverted",
    "min_dihedral_deg", "tet_volume_rel_error", "tet_error",
    "warnings", "seconds", "error",
]


# ----------------------------------------------------------------- normalization

def normalize_to_training_frame(mesh):
    """Centre on trimesh's area-weighted `.centroid`, scale max extent to 1.0.

    Returns (mesh, translation, scale). The centre is pinned rather than measured:
    the measurement that chose it is recorded in the module docstring above, and
    `pc2mesh.prepare` calls this same function so training and inference cannot
    drift apart. Scaling happens about the origin AFTER the translation, so the
    centroid stays at the origin.
    """
    m = mesh.copy()
    t = -np.asarray(m.centroid, dtype=np.float64)
    m.apply_translation(t)
    ext = float(np.max(m.extents))
    s = 1.0 / ext if ext > 0 else 1.0
    m.apply_scale(s)
    return m, t, s


def sample_cloud(mesh, n_points: int, seed: int):
    """(N,6) float64 xyz ++ unit face normal, exactly `n_points` rows.

    `sample_surface_even` culls to a Poisson-disk radius and so returns AT MOST
    the count asked for; the shortfall is topped up from the plain area-weighted
    sampler and reported. It is never padded by duplicating a point.
    """
    import trimesh

    pts, fidx = trimesh.sample.sample_surface_even(mesh, n_points, seed=seed)
    pts = np.asarray(pts, dtype=np.float64)
    fidx = np.asarray(fidx, dtype=np.int64)
    n_even = len(pts)
    n_topup = 0
    if n_even < n_points:
        n_topup = n_points - n_even
        p2, f2 = trimesh.sample.sample_surface(mesh, n_topup, seed=seed + 1)
        pts = np.concatenate([pts, np.asarray(p2, dtype=np.float64)], axis=0)
        fidx = np.concatenate([fidx, np.asarray(f2, dtype=np.int64)], axis=0)
    elif n_even > n_points:                       # defensive; the sampler caps
        pts, fidx = pts[:n_points], fidx[:n_points]
        n_even = n_points
    nrm = np.asarray(mesh.face_normals, dtype=np.float64)[fidx]
    return np.concatenate([pts, nrm], axis=1), n_even, n_topup


# ----------------------------------------------------------------- bounds / OOD

def training_bounds(cfg, ckpt: dict | None, override: str | None) -> tuple[np.ndarray, str]:
    """The UNION bbox of the corpus the checkpoint was trained on, and where it came from.

    Preference order, most specific first:
      1. --bounds <json>
      2. the checkpoint's own `global_bounds` (released checkpoints carry it)
      3. data/global_bounds.json

    3 is the source-repo path and is the one that moves: `prepare` rewrites it for
    whatever corpus a user builds. A checkpoint that carries its own bounds keeps
    decoding on the grid it was fitted to regardless.
    """
    if override:
        with open(override) as f:
            d = json.load(f)
        return np.array([d["min"], d["max"]], dtype=np.float64), str(override)
    if ckpt is not None and ckpt.get("global_bounds") is not None:
        d = ckpt["global_bounds"]
        return np.array([d["min"], d["max"]], dtype=np.float64), "checkpoint"
    p = resolve(cfg.paths.data) / "global_bounds.json"
    with open(p) as f:
        d = json.load(f)
    return np.array([d["min"], d["max"]], dtype=np.float64), str(p)


def target_faces(cfg, args) -> int:
    """The decimation budget: --target-faces, else `infer.target_faces` (the
    operating point), else `remesh.target_faces` (the probe's current budget)."""
    if args.target_faces:
        return int(args.target_faces)
    inf = cfg.get("infer") or {}
    return int(inf.get("target_faces") or cfg.remesh.target_faces)


def bbox_check(bbox: np.ndarray, train: np.ndarray) -> tuple[bool, float, np.ndarray]:
    """(inside, worst overflow, per-axis signed overflow) of `bbox` against `train`.

    Overflow is measured in the normalized units the frame is defined in, so it is
    directly comparable to the max extent of 1.0 every input is scaled to.
    """
    under = train[0] - bbox[0]     # >0 => input reaches below the training minimum
    over = bbox[1] - train[1]      # >0 => input reaches above the training maximum
    worst = float(max(under.max(), over.max()))
    return worst <= 0.0, worst, np.stack([under, over])


# ----------------------------------------------------------------- tets

def tet_one(mesh, backend: str, size_factor: float, optimize: bool):
    """Fill one watertight surface. Returns a dict of measurements, never raises."""
    from pc2mesh.tetrahedralize import (
        TETRAHEDRALIZE, min_dihedrals, shell_report, signed_volumes,
    )

    out = {"tet_ok": 0, "n_tets": "", "n_tet_nodes": "", "tet_inverted": "",
           "min_dihedral_deg": "", "tet_volume_rel_error": "", "tet_error": ""}
    try:
        V = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        F = np.ascontiguousarray(mesh.faces, dtype=np.int64)
        _, _, _, vs = shell_report(mesh)
        nodes, tets = TETRAHEDRALIZE[backend](V, F, size_factor, optimize)
        if len(tets) == 0:
            out["tet_error"] = "backend produced 0 tetrahedra"
            return out
        sv = signed_volumes(nodes, tets)
        scale = float(np.abs(sv).max())
        deg = int((np.abs(sv) <= 1e-12 * max(scale, 1e-30)).sum())
        neg = int((sv < 0).sum()) - int(((sv < 0) & (np.abs(sv) <= 1e-12 * scale)).sum())
        pos = len(sv) - neg - deg
        n_inv = (neg if pos >= neg else pos) + deg
        dih = min_dihedrals(nodes, tets)
        vt = float(np.abs(sv).sum())
        out.update(tet_ok=1, n_tets=len(tets), n_tet_nodes=len(nodes),
                   tet_inverted=n_inv, min_dihedral_deg=float(dih.min()),
                   tet_volume_rel_error=abs(vt - vs) / max(vs, 1e-30))
        out["_nodes"], out["_tets"] = nodes, tets
        return out
    except Exception as e:
        out["tet_error"] = f"{type(e).__name__}: {e}"
        return out


# ----------------------------------------------------------------- one shape

def infer_one(path: Path, model, cfg, grid_bounds, train_bounds, out_dirs, args,
              backend: str | None) -> dict:
    """Everything for a single input file. Returns one report row."""
    import trimesh

    from pc2mesh.dataset import unit_normal_cloud
    from pc2mesh.geom import count_self_intersections
    from pc2mesh.meshify import meshify_one
    from pc2mesh.remesh import flags, remesh_one

    row = {k: "" for k in REPORT_FIELDS}
    stem = path.stem
    kind = "stl" if path.suffix.lower() in (".stl",) else "npy"
    row.update(stem=stem, input=str(path), input_kind=kind, status="ok")
    warn: list[str] = []
    t0 = time.time()
    n_points = int(cfg.model.n_points)

    def finish(**kw):
        row.update(kw)
        row["warnings"] = "; ".join(warn)
        row["seconds"] = round(time.time() - t0, 3)
        if row["status"] != "ok":
            # Say it here, where the step that stopped is still visible, as well as
            # in the summary table. Nothing is repaired and nothing is written.
            print(f"  {row['status'].upper():<11s} {row['reject_reason'] or row['error']}")
            print(f"              no surface written for this input.")
        return row

    try:
        # -------------------------------------------------- 1 load / 2 normalize / 3 sample
        if kind == "stl":
            raw = load_mesh(path, process=True)
            if len(raw.faces) == 0:
                return finish(status="reject", reject_reason="empty mesh (0 faces)")
            row["src_ext_max"] = float(np.max(raw.extents))
            norm, t, s = normalize_to_training_frame(raw)
            row.update(scale=s, tx=t[0], ty=t[1], tz=t[2])
            cloud, n_even, n_topup = sample_cloud(
                norm, n_points, seed=int(cfg.seed) * 100003 + (abs(hash(stem)) % 100003))
            if n_topup:
                warn.append(f"{n_topup} of {n_points} points topped up from "
                            f"sample_surface (Poisson-disk cull came up short)")
            np.save(out_dirs["clouds"] / f"{stem}.npy", cloud)
            print(f"  normalize   centroid -> origin, max extent {np.max(norm.extents):.6f} "
                  f"(source max extent {row['src_ext_max']:.6g}, scale {s:.6g})")
            print(f"  sample      {len(cloud)} points  (even {n_even} + topup {n_topup}), "
                  f"face normals")
        else:
            arr = np.load(path)
            if arr.ndim != 2 or arr.shape[1] < 6:
                return finish(status="reject",
                              reject_reason=f"expected an (N,6) array of xyz ++ normal, "
                                            f"got {arr.shape}")
            cloud = np.asarray(arr[:, :6], dtype=np.float64)
            print(f"  load        {len(cloud)} points from .npy (already in the "
                  f"training frame; not re-normalized)")

        # -------------------------------------------------- 4 checks
        row["n_points"] = len(cloud)
        try:
            pc6 = unit_normal_cloud(cloud[:, :3], cloud[:, 3:6], strict=True)
        except ValueError as e:
            # Not repaired: a zero normal is a defect in the input, and silently
            # substituting one would put this cloud off the manifold every other
            # cloud is on without saying so.
            return finish(status="reject", reject_reason=f"zero-norm normals: {e}")

        bbox = np.stack([pc6[:, :3].min(0), pc6[:, :3].max(0)]).astype(np.float64)
        inside, overflow, per_axis = bbox_check(bbox, train_bounds)
        row.update(bbox_min_x=bbox[0, 0], bbox_min_y=bbox[0, 1], bbox_min_z=bbox[0, 2],
                   bbox_max_x=bbox[1, 0], bbox_max_y=bbox[1, 1], bbox_max_z=bbox[1, 2],
                   bbox_in_training_range=int(inside), bbox_overflow_max=overflow,
                   point_count_expected=int(len(cloud) == n_points))
        print(f"  checks      bbox {np.round(bbox[0], 4).tolist()} .. "
              f"{np.round(bbox[1], 4).tolist()}")
        if not inside:
            warn.append(f"normalized bbox falls {overflow:.4f} outside the training "
                        f"range on at least one axis")
            print(f"  WARNING     bbox is OUTSIDE the training range "
                  f"{np.round(train_bounds[0], 4).tolist()} .. "
                  f"{np.round(train_bounds[1], 4).tolist()} by up to {overflow:.4f}.")
            print(f"              The decode grid is that box padded "
                  f"{float(cfg.meshify.bbox_pad):.0%} and its outer voxel shell is "
                  f"forced outside, so geometry beyond the grid IS CLIPPED, and the "
                  f"shape is out of distribution besides.")
        if len(cloud) != n_points:
            warn.append(f"point count {len(cloud)} != {n_points}")
            print(f"  WARNING     point count is {len(cloud)}, not the {n_points} the "
                  f"encoder was trained on; it subsamples to {n_points} if there are "
                  f"more and cannot make up the difference if there are fewer.")

        # -------------------------------------------------- 5 meshify
        res = int(args.resolution or cfg.meshify.resolution)
        mesh = meshify_one(model, pc6, cfg, grid_bounds, args.device,
                           drop_floaters=not args.no_drop_floaters, resolution=res)
        row["mc_faces"] = len(mesh.faces)
        if len(mesh.faces) == 0:
            return finish(status="fail",
                          reject_reason="marching cubes found no level set (the field "
                                        "is entirely inside or entirely outside)")
        n_shells = len(mesh.split(only_watertight=False))
        row["n_shells"] = n_shells
        print(f"  meshify     R={res}{'' if args.no_drop_floaters else ', drop-floaters'}"
              f" -> {len(mesh.faces)} faces, {n_shells} shell(s)")

        # -------------------------------------------------- 6 decimate (revert on break)
        target = target_faces(cfg, args)
        if args.no_decimate:
            final = mesh
            print("  decimate    skipped (--no-decimate)")
        else:
            final, failure = remesh_one(mesh, target)
            if failure is None:
                print(f"  decimate    {len(mesh.faces)} -> {len(final.faces)} faces "
                      f"(target {target}, kept)")
            else:
                row.update(decimate_reverted=1, revert_step=failure["step"],
                           revert_reason=failure["reason"])
                warn.append(f"decimation reverted at step '{failure['step']}': "
                            f"{failure['reason']}")
                print(f"  decimate    REVERTED at '{failure['step']}' "
                      f"({failure['reason']}); the {len(mesh.faces)}-face marching-cubes "
                      f"mesh is written instead. A broken mesh is never emitted.")

        dest = out_dirs["surface"] / f"{stem}.stl"
        final.export(dest)
        # The artifact is the FILE. STL is a float32 triangle soup and export is the
        # last step that can break a mesh, so the invariants are re-read from disk.
        final = load_mesh(dest, process=True)
        wt, wc = flags(final)
        si = count_self_intersections(final)
        row.update(faces=len(final.faces), watertight=int(wt), winding_consistent=int(wc),
                   self_intersections=si, self_intersection_free=int(si == 0))
        print(f"  topology    watertight {'yes' if wt else 'NO'} | winding consistent "
              f"{'yes' if wc else 'NO'} | self-intersecting face pairs {si} | "
              f"{len(final.faces)} faces")
        if not (wt and wc):
            warn.append("emitted surface is not watertight/winding consistent")
            print(f"  WARNING     the emitted surface does NOT hold the invariant this "
                  f"pipeline exists to guarantee.")
        if si:
            print(f"              self-intersections predict tetrahedralization failure: "
                  f"gmsh cannot fill a surface whose facets overlap.")

        # -------------------------------------------------- 7 tets
        if args.tets:
            row["tet_backend"] = backend or ""
            if backend is None:
                row["tet_error"] = "no tetrahedralization backend importable"
                print(f"  tets        no backend available; nothing written")
            elif not wt:
                row["tet_error"] = "input surface is not watertight"
                print(f"  tets        skipped: the surface is not watertight")
            else:
                t = tet_one(final, backend, args.size_factor, not args.no_optimize)
                nodes, tets = t.pop("_nodes", None), t.pop("_tets", None)
                row.update(t)
                if t["tet_ok"]:
                    import meshio

                    meshio.write_points_cells(out_dirs["tets"] / f"{stem}.vtu",
                                              nodes, [("tetra", tets)])
                    print(f"  tets        {backend}: {t['n_tets']} tets, "
                          f"{t['tet_inverted']} inverted, min dihedral "
                          f"{t['min_dihedral_deg']:.3f} deg, volume error "
                          f"{t['tet_volume_rel_error']:.2e}")
                else:
                    warn.append(f"tetrahedralization failed: {t['tet_error']}")
                    print(f"  tets        {backend}: FAILED — {t['tet_error']}")
        return finish()
    except Exception as e:
        return finish(status="error", error=f"{type(e).__name__}: {e}")


# ----------------------------------------------------------------- driver

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Point cloud or mesh -> watertight surface (and optional tets).")
    ap.add_argument("--in", dest="in_dir", required=True,
                    help="directory of .npy clouds and/or .stl meshes")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint (default: checkpoints/pc2mesh_v3.pt if present)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--bounds", default=None,
                    help="override the training bbox JSON (min/max)")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--resolution", type=int, default=0, help="override meshify.resolution")
    ap.add_argument("--target-faces", type=int, default=0, help="override remesh.target_faces")
    ap.add_argument("--no-decimate", action="store_true", help="emit the marching-cubes mesh")
    ap.add_argument("--no-drop-floaters", action="store_true",
                    help="keep every connected component, however small")
    ap.add_argument("--tets", action="store_true", help="also fill each surface with tetrahedra")
    ap.add_argument("--size-factor", type=float, default=1.0, help="gmsh Mesh.MeshSizeFactor")
    ap.add_argument("--no-optimize", action="store_true",
                    help="skip the backend's tet-quality optimization pass")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch

    from pc2mesh.meshify import load_checkpoint

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = Path(args.ckpt) if args.ckpt else resolve("checkpoints/pc2mesh_v3.pt")
    if not Path(ckpt_path).exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}  (pass --ckpt)")

    in_dir = Path(args.in_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"--in must be a directory: {in_dir}")
    paths = sorted([p for p in in_dir.rglob("*")
                    if p.suffix.lower() in (".npy", ".stl") and p.is_file()])
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no .npy or .stl under {in_dir}")

    out = Path(args.out)
    out_dirs = {"surface": out / "surface", "clouds": out / "clouds", "tets": out / "tets"}
    for k, d in out_dirs.items():
        if k != "tets" or args.tets:
            d.mkdir(parents=True, exist_ok=True)

    model, ck = load_checkpoint(ckpt_path, cfg, args.device)
    train_bounds, bounds_src = training_bounds(cfg, ck, args.bounds)
    grid_bounds = padded_bounds(train_bounds, float(cfg.meshify.bbox_pad))

    backend = None
    if args.tets:
        from pc2mesh.tetrahedralize import available_backend

        name, version = available_backend()
        backend = name
        backend_label = f"{name} {version}" if name else "NONE AVAILABLE"
    n_stl = sum(1 for p in paths if p.suffix.lower() == ".stl")
    res = int(args.resolution or cfg.meshify.resolution)

    print("================ PC2MESH INFERENCE ================")
    print(f"checkpoint     : {ckpt_path}")
    print(f"device         : {args.device}"
          + ("" if args.device == "cpu" else f" ({torch.cuda.get_device_name(0)})"))
    print(f"input          : {in_dir}  ({len(paths)} files: {n_stl} .stl, "
          f"{len(paths)-n_stl} .npy)")
    print(f"output         : {out}")
    print(f"training bbox  : {np.round(train_bounds[0], 4).tolist()} .. "
          f"{np.round(train_bounds[1], 4).tolist()}   (from {bounds_src})")
    print(f"decode grid    : {res}^3 over that box padded "
          f"{float(cfg.meshify.bbox_pad):.0%}, outer shell forced outside")
    print(f"face budget    : {'none (--no-decimate)' if args.no_decimate else target_faces(cfg, args)}"
          f"   floaters: {'kept' if args.no_drop_floaters else 'dropped'}")
    if args.tets:
        print(f"tet backend    : {backend_label}")
    print("no ground truth exists at inference; topology and OOD flags only, no IoU")
    print("=" * 50)

    rows = []
    t0 = time.time()
    for i, p in enumerate(paths, 1):
        print(f"\n[{i}/{len(paths)}] {p.stem}  ({p.suffix.lstrip('.').lower()})")
        rows.append(infer_one(p, model, cfg, grid_bounds, train_bounds, out_dirs,
                              args, backend))
    wall = time.time() - t0

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "infer_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REPORT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    def rate(col):
        v = [r[col] for r in ok if r[col] != ""]
        return float(np.mean(v)) if v else float("nan")

    summary = {
        "checkpoint": str(ckpt_path), "device": args.device,
        "input_dir": str(in_dir), "output_dir": str(out),
        "n_inputs": len(rows), "n_ok": len(ok),
        "n_rejected": sum(1 for r in rows if r["status"] == "reject"),
        "n_failed": sum(1 for r in rows if r["status"] in ("fail", "error")),
        "training_bbox": {"min": train_bounds[0].tolist(), "max": train_bounds[1].tolist(),
                          "source": bounds_src},
        "meshify_resolution": res, "bbox_pad": float(cfg.meshify.bbox_pad),
        "target_faces": None if args.no_decimate else target_faces(cfg, args),
        "drop_floaters": not args.no_drop_floaters,
        "watertight_rate": rate("watertight"),
        "winding_consistent_rate": rate("winding_consistent"),
        "self_intersection_free_rate": rate("self_intersection_free"),
        "faces_mean": rate("faces"),
        "n_reverted": sum(1 for r in ok if r["decimate_reverted"] == 1),
        "n_bbox_out_of_range": sum(1 for r in ok if r["bbox_in_training_range"] == 0),
        "n_wrong_point_count": sum(1 for r in ok if r["point_count_expected"] == 0),
        "tets_requested": bool(args.tets),
        "tet_backend": backend,
        "tet_success_rate": (rate("tet_ok") if args.tets else None),
        "wall_seconds": wall,
        "note": "no ground truth at inference: topology invariants and OOD flags only",
    }
    with open(out / "infer_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n================ SUMMARY ================")
    print(f"{'shape':44s} {'faces':>7s} {'wt':>3s} {'wc':>3s} {'selfX':>6s} "
          f"{'tets':>8s}  flags")
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['stem'][:44]:44s} {'-':>7s} {'-':>3s} {'-':>3s} {'-':>6s} "
                  f"{'-':>8s}  {r['status'].upper()}: "
                  f"{(r['reject_reason'] or r['error'])[:60]}")
            continue
        tet = ("-" if not args.tets else
               (str(r["n_tets"]) if r["tet_ok"] == 1 else "FAILED"))
        fl = []
        if r["bbox_in_training_range"] == 0:
            fl.append("WARNING bbox out of training range")
        if r["point_count_expected"] == 0:
            fl.append(f"WARNING point count {r['n_points']}")
        if r["decimate_reverted"] == 1:
            fl.append("decimation reverted")
        print(f"{r['stem'][:44]:44s} {r['faces']:7d} "
              f"{'y' if r['watertight'] else 'N':>3s} "
              f"{'y' if r['winding_consistent'] else 'N':>3s} "
              f"{r['self_intersections']:6d} {tet:>8s}  {'; '.join(fl)}")
    print(f"\ninputs {len(rows)}  ok {len(ok)}  rejected {summary['n_rejected']}  "
          f"failed {summary['n_failed']}")
    if ok:
        print(f"watertight {summary['watertight_rate']:.4f}   winding consistent "
              f"{summary['winding_consistent_rate']:.4f}   self-intersection free "
              f"{summary['self_intersection_free_rate']:.4f}")
        if args.tets:
            print(f"tet success {summary['tet_success_rate']:.4f} "
                  f"(backend {backend or 'none'})")
        if summary["n_bbox_out_of_range"]:
            print(f"!! {summary['n_bbox_out_of_range']} input(s) OUTSIDE the training "
                  f"bbox — those outputs may be clipped by the grid shell")
        if summary["n_wrong_point_count"]:
            print(f"!! {summary['n_wrong_point_count']} input(s) did not carry "
                  f"{int(cfg.model.n_points)} points")
    print(f"wall {wall:.1f}s ({wall/max(1,len(paths)):.2f}s per shape) -> "
          f"{out / 'infer_report.csv'}")
    print("=" * 40)
    # A rejected or failed input is a reportable outcome and exits 0; the pipeline
    # is only broken if it EMITTED something that violates its own invariant.
    broken = [r for r in ok if not (r["watertight"] and r["winding_consistent"])]
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
