"""Stage 1 — pair the two source trees and audit them. Run this FIRST.

This is the main risk in the project: if the point clouds and the meshes are not
in the same frame, or are not actually the same objects, every downstream number
is meaningless and nothing else in the pipeline would notice.

Outputs
    data/manifest.csv            one row per matched pair, with the facts
    data/pair_audit.csv          checks (a) FRAME, (b) NORMALIZATION, (c) COUNT
    data/unmatched.csv           every file that failed to pair, on BOTH sides
    data/global_bounds.json      the single frame shared by all later stages
    data/pair_scan_summary.json  aggregate stats + STOP-gate verdicts

Exits non-zero if a STOP gate trips.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pc2mesh.common import (  # noqa: E402
    index_tree,
    load_config,
    load_mesh,
    norm_stem,
    rel_folder,
    resolve,
    set_seed,
    sha1_file,
)

warnings.filterwarnings("ignore")

MANIFEST_FIELDS = [
    "stem", "folder", "mesh_path", "pc_path", "mesh_sha1", "pc_sha1",
    "n_vertices", "n_faces",
    "mesh_min_x", "mesh_min_y", "mesh_min_z",
    "mesh_max_x", "mesh_max_y", "mesh_max_z",
    "mesh_ext_x", "mesh_ext_y", "mesh_ext_z", "mesh_ext_max",
    "is_watertight", "is_winding_consistent", "euler_number",
    "npy_rows", "npy_cols", "npy_dtype",
    "pc_min_x", "pc_min_y", "pc_min_z",
    "pc_max_x", "pc_max_y", "pc_max_z",
    "error",
]

AUDIT_FIELDS = [
    "stem", "folder", "n_points",
    "pc2mesh_mean", "pc2mesh_p99", "pc2mesh_max", "frame_flag",
    "mesh_ext_max", "abs_coord_max", "centroid_off",
    "is_watertight", "is_winding_consistent", "error",
]


def _scan_one(job):
    stem, folder, mesh_path, pc_path, p99_max = job
    row = {k: "" for k in MANIFEST_FIELDS}
    aud = {k: "" for k in AUDIT_FIELDS}
    row.update(stem=stem, folder=folder, mesh_path=mesh_path, pc_path=pc_path)
    aud.update(stem=stem, folder=folder)
    try:
        row["mesh_sha1"] = sha1_file(mesh_path)
        row["pc_sha1"] = sha1_file(pc_path)

        mesh = load_mesh(mesh_path, process=True)
        b = np.asarray(mesh.bounds, dtype=np.float64)
        ext = b[1] - b[0]
        row.update(
            n_vertices=len(mesh.vertices), n_faces=len(mesh.faces),
            mesh_min_x=b[0, 0], mesh_min_y=b[0, 1], mesh_min_z=b[0, 2],
            mesh_max_x=b[1, 0], mesh_max_y=b[1, 1], mesh_max_z=b[1, 2],
            mesh_ext_x=ext[0], mesh_ext_y=ext[1], mesh_ext_z=ext[2],
            mesh_ext_max=ext.max(),
            is_watertight=int(mesh.is_watertight),
            is_winding_consistent=int(mesh.is_winding_consistent),
            euler_number=int(mesh.euler_number),
        )

        arr = np.load(pc_path)
        row.update(npy_rows=arr.shape[0], npy_cols=arr.shape[1] if arr.ndim > 1 else 1,
                   npy_dtype=str(arr.dtype))
        pts = np.ascontiguousarray(arr[:, :3], dtype=np.float64)
        pb = np.stack([pts.min(0), pts.max(0)])
        row.update(pc_min_x=pb[0, 0], pc_min_y=pb[0, 1], pc_min_z=pb[0, 2],
                   pc_max_x=pb[1, 0], pc_max_y=pb[1, 1], pc_max_z=pb[1, 2])

        # ---- (a) FRAME: how far are the cloud points from the mesh surface?
        import trimesh

        _, dist, _ = trimesh.proximity.closest_point(mesh, pts)
        dist = np.asarray(dist, dtype=np.float64)
        aud.update(
            n_points=len(pts),
            pc2mesh_mean=float(dist.mean()),
            pc2mesh_p99=float(np.percentile(dist, 99)),
            pc2mesh_max=float(dist.max()),
            frame_flag=int(float(np.percentile(dist, 99)) > p99_max),
            # ---- (b) NORMALIZATION evidence, per mesh
            mesh_ext_max=float(ext.max()),
            abs_coord_max=float(np.abs(b).max()),
            centroid_off=float(np.abs(b.mean(0)).max()),
            is_watertight=int(mesh.is_watertight),
            is_winding_consistent=int(mesh.is_winding_consistent),
        )
        return row, aud, np.stack([np.minimum(b[0], pb[0]), np.maximum(b[1], pb[1])])
    except Exception as e:  # a broken file must be reported, never dropped
        row["error"] = f"{type(e).__name__}: {e}"
        aud["error"] = row["error"]
        return row, aud, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug: only scan N pairs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed, torch_too=False)
    data_dir = resolve(cfg.paths.data)
    data_dir.mkdir(parents=True, exist_ok=True)

    mesh_root = resolve(cfg.paths.meshes)
    pc_root = resolve(cfg.paths.pointclouds)
    print(f"meshes      : {mesh_root}")
    print(f"pointclouds : {pc_root}")

    mesh_idx, mesh_col = index_tree(mesh_root, cfg.paths.mesh_glob)
    pc_idx, pc_col = index_tree(pc_root, cfg.paths.pc_glob)
    print(f"indexed {len(mesh_idx)} meshes, {len(pc_idx)} point clouds")
    for label, col in (("mesh", mesh_col), ("pc", pc_col)):
        if col:
            print(f"!! {len(col)} normalized-stem COLLISIONS on the {label} side:")
            for k, v in list(col.items())[:10]:
                print(f"   {k}: {[str(x) for x in v]}")

    paired = sorted(set(mesh_idx) & set(pc_idx))
    only_mesh = sorted(set(mesh_idx) - set(pc_idx))
    only_pc = sorted(set(pc_idx) - set(mesh_idx))

    with open(data_dir / "unmatched.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["side", "stem", "path"])
        for s in only_mesh:
            w.writerow(["mesh_without_pointcloud", s, str(mesh_idx[s])])
        for s in only_pc:
            w.writerow(["pointcloud_without_mesh", s, str(pc_idx[s])])
        for side, col in (("mesh_stem_collision", mesh_col), ("pc_stem_collision", pc_col)):
            for k, v in col.items():
                for p in v:
                    w.writerow([side, k, str(p)])
    print(f"paired {len(paired)} | mesh-only {len(only_mesh)} | pc-only {len(only_pc)}"
          f"  -> data/unmatched.csv")

    if args.limit:
        paired = paired[: args.limit]

    jobs = [
        (s, rel_folder(mesh_idx[s], mesh_root), str(mesh_idx[s]), str(pc_idx[s]),
         float(cfg.scan.frame_p99_max))
        for s in paired
    ]

    rows, auds, boxes = [], [], []
    with Pool(int(cfg.scan.workers)) as pool:
        for row, aud, box in tqdm(pool.imap_unordered(_scan_one, jobs, chunksize=4),
                                  total=len(jobs), desc="scan"):
            rows.append(row)
            auds.append(aud)
            if box is not None:
                boxes.append(box)
    rows.sort(key=lambda r: r["stem"])
    auds.sort(key=lambda r: r["stem"])

    with open(data_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(data_dir / "pair_audit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        w.writeheader()
        w.writerows(auds)

    # ------------------------------------------------------------------ verdicts
    ok = [a for a in auds if not a["error"]]
    errs = [a for a in auds if a["error"]]
    n = len(ok)
    if n == 0:
        print("FATAL: no pair could be read")
        return 2

    p99 = np.array([a["pc2mesh_p99"] for a in ok])
    mean_d = np.array([a["pc2mesh_mean"] for a in ok])
    wt = np.array([a["is_watertight"] for a in ok])
    wc = np.array([a["is_winding_consistent"] for a in ok])
    ext = np.array([a["mesh_ext_max"] for a in ok])
    absc = np.array([a["abs_coord_max"] for a in ok])
    npts = np.array([a["n_points"] for a in ok])

    frame_fail = int((p99 > cfg.scan.frame_p99_max).sum())
    frame_fail_rate = frame_fail / n
    wt_rate = float(wt.mean())
    scale_spread = float((ext.max() - ext.min()) / ext.mean())

    gbox = np.stack(boxes)
    gmin = gbox[:, 0].min(0)
    gmax = gbox[:, 1].max(0)
    with open(data_dir / "global_bounds.json", "w") as f:
        json.dump({"min": gmin.tolist(), "max": gmax.tolist(),
                   "note": "union of mesh and point-cloud bounds over all matched pairs; "
                           "consumers apply their own padding"}, f, indent=2)

    stops = []
    if wt_rate < cfg.scan.min_watertight_rate:
        stops.append(f"watertight rate {wt_rate:.3f} < {cfg.scan.min_watertight_rate}")
    if frame_fail_rate > cfg.scan.max_frame_fail_rate:
        stops.append(f"FRAME check fail rate {frame_fail_rate:.3f} > {cfg.scan.max_frame_fail_rate}")
    if absc.max() > cfg.scan.frame_abs_coord_max:
        stops.append(f"vertices outside [-1,1]: max |coord| = {absc.max():.4f}")
    if scale_spread > cfg.scan.frame_scale_spread_max:
        stops.append(f"scale spread {scale_spread:.4f} > {cfg.scan.frame_scale_spread_max}"
                     " -> meshes are NOT in a common frame")

    summary = {
        "n_mesh_files": len(mesh_idx), "n_pc_files": len(pc_idx),
        "n_paired": len(paired), "n_scanned": len(auds),
        "n_mesh_only": len(only_mesh), "n_pc_only": len(only_pc),
        "n_errors": len(errs),
        "check_a_frame": {
            "pc2mesh_mean_of_means": float(mean_d.mean()),
            "pc2mesh_p99_median": float(np.median(p99)),
            "pc2mesh_p99_max": float(p99.max()),
            "threshold": float(cfg.scan.frame_p99_max),
            "n_flagged": frame_fail, "fail_rate": frame_fail_rate,
        },
        "check_b_normalization": {
            "global_bbox_min": gmin.tolist(), "global_bbox_max": gmax.tolist(),
            "global_bbox_extent": (gmax - gmin).tolist(),
            "max_abs_coord": float(absc.max()),
            "mesh_max_extent_min": float(ext.min()),
            "mesh_max_extent_max": float(ext.max()),
            "mesh_max_extent_mean": float(ext.mean()),
            "scale_spread": scale_spread,
            "common_frame": bool(scale_spread <= cfg.scan.frame_scale_spread_max
                                 and absc.max() <= cfg.scan.frame_abs_coord_max),
        },
        "check_c_point_count": {
            "N_min": int(npts.min()), "N_max": int(npts.max()),
            "N_median": float(np.median(npts)),
            "N_hist": {str(int(k)): int(v) for k, v in
                       zip(*np.unique(npts, return_counts=True))},
        },
        "topology": {
            "watertight_rate": wt_rate,
            "winding_consistent_rate": float(wc.mean()),
        },
        "stop_gates": {"tripped": stops, "passed": not stops},
    }
    with open(data_dir / "pair_scan_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ PAIR SCAN ================")
    print(f"pairs scanned            : {n} (errors: {len(errs)})")
    print(f"(a) FRAME  mean dist     : {mean_d.mean():.6f}")
    print(f"           p99 median/max: {np.median(p99):.6f} / {p99.max():.6f}")
    print(f"           flagged >{cfg.scan.frame_p99_max}: {frame_fail} ({frame_fail_rate:.2%})")
    print(f"(b) NORM   global bbox   : {np.round(gmin,4).tolist()} .. {np.round(gmax,4).tolist()}")
    print(f"           max |coord|   : {absc.max():.4f}")
    print(f"           max-extent    : {ext.min():.6f} .. {ext.max():.6f} (spread {scale_spread:.2e})")
    print(f"(c) COUNT  N min/max     : {npts.min()} / {npts.max()}  distinct={len(set(npts.tolist()))}")
    print(f"    topology watertight  : {wt_rate:.3f}   winding-consistent: {wc.mean():.3f}")
    for s in stops:
        print(f"!! STOP: {s}")
    print("verdict                  :", "PASS" if not stops else "STOP")
    print("===========================================")
    return 0 if not stops else 1


if __name__ == "__main__":
    raise SystemExit(main())
