"""Stage 1 — normalize the STLs in data/stl/ and sample a point cloud from each.

Input is whatever you drop in `data/stl/`, in any units and any frame, optionally
in subdirectories. One subdirectory = one "folder" label, and that label is what
the per-folder gate breakdown and the 90/10 split are grouped by, so putting two
different sources in two subdirectories is how you keep them separable.

Output is the shelf every later stage reads:

    data/corpus/<subdir>/<stem>.stl    normalized mesh
    data/clouds/<subdir>/<stem>.npy    (n_points, 6) float64, xyz ++ unit face normal

THE NORMALIZATION is `pc2mesh.infer.normalize_to_training_frame` — the same
function inference uses, deliberately, so a shape cannot be framed one way in
training and another way at inference. It centres on trimesh's area-weighted
`.centroid` and scales the max extent to 1.0. It does NOT centre on the bbox
centre: measured over the 1250 meshes of the reference corpus, the bbox centre
holds for 1.8% of them and is off by up to 0.2589, while the centroid holds for
100% of them to 3.19e-08.

REJECTED HERE, with the reason recorded, rather than later:

    empty meshes (0 faces)
    meshes still not watertight / winding-consistent after the labeling repair
    (fill_holes -> fix_normals), because the winding number cannot label them and
    they would cost a sample and a cache entry before being thrown out
    stem collisions across subdirectories, because the query cache is flat

SAMPLING is `trimesh.sample.sample_surface_even` for exactly `model.n_points`.
That sampler culls to a Poisson-disk radius and so returns AT MOST the count
asked for; the shortfall is topped up from `sample_surface` and the count is
written per shape to data/corpus_build.csv. It is never padded by duplicating an
existing point, and never left short.

    python main.py prepare
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
    load_config, load_mesh, norm_stem, rel_folder, repair_for_labeling, resolve,
)
from pc2mesh.infer import normalize_to_training_frame, sample_cloud  # noqa: E402

warnings.filterwarnings("ignore")

BUILD_FIELDS = [
    "folder", "stem", "src_path", "mesh_out", "cloud_out", "status",
    "n_faces", "src_ext_max", "scale", "tx", "ty", "tz",
    "n_even", "n_topup", "shortfall_frac",
    "centroid_off", "bbox_center_off", "ext_max_after",
    "normal_norm_min", "normal_norm_max", "error",
]

_G = {}


def _init(n_points, corpus_dir, clouds_dir, seed):
    _G.update(n_points=int(n_points), corpus=Path(corpus_dir),
              clouds=Path(clouds_dir), seed=int(seed))


def _build_one(job):
    folder, stem, src, idx = job
    row = {k: "" for k in BUILD_FIELDS}
    row.update(folder=folder, stem=stem, src_path=src)
    n_points = _G["n_points"]
    try:
        raw = load_mesh(src, process=True)
        if len(raw.faces) == 0:
            row.update(status="reject", error="empty mesh (0 faces)")
            return row
        _, _, ok = repair_for_labeling(raw)
        if not ok:
            row.update(status="reject", n_faces=len(raw.faces),
                       error="not watertight/winding-consistent after fill_holes+fix_normals")
            return row

        row["src_ext_max"] = float(np.max(raw.extents))
        norm, t, s = normalize_to_training_frame(raw)
        row.update(scale=s, tx=t[0], ty=t[1], tz=t[2])

        mesh_out = _G["corpus"] / folder / f"{stem}.stl"
        mesh_out.parent.mkdir(parents=True, exist_ok=True)
        norm.export(mesh_out)

        # Sample the mesh AS IT IS ON DISK: every later stage re-reads the STL, and
        # STL is float32, so sampling the in-memory float64 copy would put the cloud
        # a rounding step off the surface the frame check measures.
        mesh = load_mesh(mesh_out, process=True)
        row.update(n_faces=len(mesh.faces),
                   centroid_off=float(np.abs(mesh.centroid).max()),
                   bbox_center_off=float(np.abs(np.asarray(mesh.bounds).mean(0)).max()),
                   ext_max_after=float(np.max(mesh.extents)))

        cloud, n_even, n_topup = sample_cloud(
            mesh, n_points, seed=_G["seed"] * 100003 + idx)
        assert cloud.shape == (n_points, 6), f"{stem}: cloud is {cloud.shape}"
        nn = np.linalg.norm(cloud[:, 3:6], axis=1)

        cloud_out = _G["clouds"] / folder / f"{stem}.npy"
        cloud_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(cloud_out, cloud)

        row.update(status="ok", mesh_out=str(mesh_out), cloud_out=str(cloud_out),
                   n_even=n_even, n_topup=n_topup, shortfall_frac=n_topup / n_points,
                   normal_norm_min=float(nn.min()), normal_norm_max=float(nn.max()))
        return row
    except Exception as e:
        row.update(status="error", error=f"{type(e).__name__}: {e}")
        return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="rebuild shapes whose normalized mesh already exists")
    args = ap.parse_args()

    cfg = load_config(args.config)
    workers = args.workers or int(cfg.scan.workers)
    data_dir = resolve(cfg.paths.data)
    data_dir.mkdir(parents=True, exist_ok=True)
    stl_in = resolve(cfg.paths.stl_in)
    corpus_dir = resolve(cfg.paths.meshes)
    clouds_dir = resolve(cfg.paths.pointclouds)

    srcs = sorted(p for p in stl_in.rglob("*") if p.is_file()
                  and p.suffix.lower() == ".stl")
    if args.limit:
        srcs = srcs[: args.limit]
    print(f"input   : {stl_in}")
    print(f"corpus  : {corpus_dir}")
    print(f"clouds  : {clouds_dir}")
    print(f"found   : {len(srcs)} .stl file(s)")
    if not srcs:
        print(f"\n!! nothing to prepare. Drop .stl files into {stl_in} "
              f"(subdirectories become folder labels) and run this again.")
        return 1

    # The query cache is FLAT, so two subdirectories cannot claim the same stem.
    jobs, seen = [], {}
    for i, p in enumerate(srcs):
        stem = norm_stem(p.stem)
        folder = rel_folder(p, stl_in)
        if stem in seen:
            print(f"!! STOP: stem collision — {p} and {seen[stem]} both normalize to "
                  f"'{stem}', and the query cache is flat. Rename one.")
            return 1
        seen[stem] = p
        if not args.force and (corpus_dir / folder / f"{stem}.stl").exists() \
                and (clouds_dir / folder / f"{stem}.npy").exists():
            continue
        jobs.append((folder, stem, str(p), i))
    print(f"to build: {len(jobs)} (use --force to rebuild the rest)")

    rows = []
    if jobs:
        with Pool(workers, initializer=_init,
                  initargs=(cfg.model.n_points, str(corpus_dir), str(clouds_dir),
                            cfg.seed)) as pool:
            rows = list(tqdm(pool.imap_unordered(_build_one, jobs, chunksize=4),
                             total=len(jobs), desc="normalize+sample"))
    rows.sort(key=lambda r: (r["folder"], r["stem"]))
    with open(data_dir / "corpus_build.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=BUILD_FIELDS)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    rej = [r for r in rows if r["status"] == "reject"]
    err = [r for r in rows if r["status"] == "error"]
    short = [r for r in ok if r["n_topup"]]
    n_on_shelf = sum(1 for _ in corpus_dir.rglob("*.stl"))
    summary = {
        "stl_in": str(stl_in), "n_source_files": len(srcs), "n_jobs": len(jobs),
        "n_ok": len(ok), "n_reject": len(rej), "n_error": len(err),
        "n_shapes_on_shelf": n_on_shelf,
        "n_points": int(cfg.model.n_points),
        "center_convention": "trimesh_centroid (area-weighted mean of triangle "
                             "centroids) at the origin, max extent scaled to 1.0",
        "shortfall": {
            "n_shapes_topped_up": len(short),
            "frac_shapes_topped_up": len(short) / max(1, len(ok)),
            "topup_points_total": int(sum(r["n_topup"] for r in short)),
            "worst_shortfall_frac": max([r["shortfall_frac"] for r in short], default=0.0),
        },
        "normalized_frame": {
            "max_centroid_offset": max([r["centroid_off"] for r in ok], default=0.0),
            "max_ext_max_error": max([abs(r["ext_max_after"] - 1.0) for r in ok],
                                     default=0.0),
        },
        "folders": sorted({r["folder"] for r in rows}),
    }
    with open(data_dir / "prepare_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ PREPARE ================")
    print(f"centre convention : {summary['center_convention']}")
    print(f"built {len(ok)}  rejected {len(rej)}  errored {len(err)}")
    for r in (rej + err)[:10]:
        print(f"  - {r['stem']}: {r['error']}")
    if len(rej) + len(err) > 10:
        print(f"  ... {len(rej) + len(err) - 10} more, all in data/corpus_build.csv")
    print(f"shapes on the shelf        : {n_on_shelf}")
    if ok:
        print(f"shortfall (topped up)      : {len(short)} shapes "
              f"({summary['shortfall']['frac_shapes_topped_up']:.4f}), "
              f"{summary['shortfall']['topup_points_total']} points, worst "
              f"{summary['shortfall']['worst_shortfall_frac']:.4f} of a cloud")
        print(f"frame: max |centroid|      : {summary['normalized_frame']['max_centroid_offset']:.3e}")
        print(f"       max |max-extent-1|  : {summary['normalized_frame']['max_ext_max_error']:.3e}")
    print(f"folders                    : {', '.join(summary['folders']) or '(none)'}")
    print("========================================")
    return 0 if n_on_shelf else 1


if __name__ == "__main__":
    raise SystemExit(main())
