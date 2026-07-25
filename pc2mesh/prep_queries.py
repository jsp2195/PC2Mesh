"""Stage 2 — generate occupancy supervision, one .npz per pair.

The point clouds are given; the only thing produced here is the inside/outside
label field the decoder is trained against.

Repair is LABELING-ONLY: fill_holes -> fix_normals on a copy, used purely to make
the winding number well-posed. The mesh that later stages score against is always
re-loaded from the original STL.

Outputs
    data/cache/<stem>.npz     pc, pc_normals, queries, occ, bbox, stem, n_near
    data/rejects.csv          pairs still broken after repair (excluded)
    data/split.json           90/10 train/val, taken WITHIN each folder
    data/prep_summary.json
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
    load_config, load_mesh, padded_bounds, repair_for_labeling, resolve, set_seed,
)

warnings.filterwarnings("ignore")

REJECT_FIELDS = ["stem", "folder", "mesh_path", "reason",
                 "watertight_before", "winding_before",
                 "watertight_after", "winding_after"]

_G = {}


def _init(cfg_d, gbounds, cache_dir):
    _G["cfg"] = cfg_d
    _G["gbounds"] = np.asarray(gbounds)
    _G["cache"] = Path(cache_dir)


def _prep_one(job):
    """Returns (stem, folder, status, info). status in {ok, reject, error}."""
    stem, folder, mesh_path, pc_path, seed = job
    cfg = _G["cfg"]
    gb = _G["gbounds"]
    try:
        mesh = load_mesh(mesh_path, process=True)
        wt0, wc0 = bool(mesh.is_watertight), bool(mesh.is_winding_consistent)

        rep, _, ok = repair_for_labeling(mesh)
        if not ok:
            return stem, folder, "reject", {
                "mesh_path": mesh_path,
                "reason": "not watertight/winding-consistent after fill_holes+fix_normals",
                "watertight_before": int(wt0), "winding_before": int(wc0),
                "watertight_after": int(rep.is_watertight),
                "winding_after": int(rep.is_winding_consistent),
            }

        # ---- assert the pair really is in the shared frame before spending work
        arr = np.load(pc_path)
        pc = np.ascontiguousarray(arr[:, :3], dtype=np.float64)
        nrm = (np.ascontiguousarray(arr[:, 3:6], dtype=np.float64)
               if arr.shape[1] >= 6 else np.zeros_like(pc))
        # A stored normal of exactly zero cannot be put on the unit sphere, so it
        # would reach the encoder off the manifold every other point is on. Reject
        # the pair here, where the reason is recorded, rather than let
        # dataset.cloud_from_npz raise in the middle of a training run.
        n_zero = int((np.linalg.norm(nrm, axis=1) < 1e-6).sum())
        if n_zero:
            return stem, folder, "reject", {
                "mesh_path": mesh_path,
                "reason": f"{n_zero}/{len(nrm)} stored normals have norm < 1e-6; "
                          f"they cannot be unit-normalized",
                "watertight_before": int(wt0), "winding_before": int(wc0),
                "watertight_after": int(rep.is_watertight),
                "winding_after": int(rep.is_winding_consistent),
            }
        mb = np.asarray(mesh.bounds, dtype=np.float64)
        assert (mb[0] >= gb[0] - 1e-6).all() and (mb[1] <= gb[1] + 1e-6).all(), \
            f"{stem}: mesh outside the shared frame {mb.tolist()} vs {gb.tolist()}"
        assert (pc.min(0) >= gb[0] - 1e-6).all() and (pc.max(0) <= gb[1] + 1e-6).all(), \
            f"{stem}: point cloud outside the shared frame"

        rng = np.random.default_rng(seed)
        nq = int(cfg["n_queries"])
        n_near = int(round(nq * float(cfg["near_frac"])))
        n_unif = nq - n_near

        # ---- near-surface half: surface point + N(0, sigma), sigmas mixed 50/50
        import trimesh

        surf, _ = trimesh.sample.sample_surface(
            rep, n_near, seed=int(rng.integers(1 << 31)))
        surf = np.asarray(surf, dtype=np.float64)
        sig = np.asarray(cfg["near_sigmas"], dtype=np.float64)
        which = rng.integers(0, len(sig), size=n_near)
        q_near = surf + rng.normal(size=(n_near, 3)) * sig[which][:, None]

        # ---- uniform half over the padded global bbox
        q_unif = rng.uniform(gb[0], gb[1], size=(n_unif, 3))

        q = np.concatenate([q_near, q_unif], axis=0)
        # Store f16, and label the STORED positions: the model must never be
        # supervised at a coordinate it is not shown.
        q16 = q.astype(np.float16)
        q = q16.astype(np.float64)
        q = np.clip(q, gb[0], gb[1])
        q16 = q.astype(np.float16)

        import igl

        V = np.ascontiguousarray(rep.vertices, dtype=np.float64)
        F = np.ascontiguousarray(rep.faces, dtype=np.int64)
        w = igl.fast_winding_number(V, F, np.ascontiguousarray(q))
        occ = (w > float(cfg["wn_threshold"])).astype(np.uint8)

        pc16 = pc.astype(np.float16)
        out = _G["cache"] / f"{stem}.npz"
        np.savez(
            out,
            pc=pc16,
            pc_normals=nrm.astype(np.float16),
            queries=q16,
            occ=occ,
            n_near=np.int32(n_near),
            bbox=gb.astype(np.float64),
            mesh_bbox=mb.astype(np.float64),
            stem=np.array(stem),
            folder=np.array(folder),
        )
        return stem, folder, "ok", {
            "occ_rate_near": float(occ[:n_near].mean()),
            "occ_rate_unif": float(occ[n_near:].mean()),
            "occ_rate": float(occ.mean()),
            "n_faces": int(len(rep.faces)),
            "repaired": int(not (wt0 and wc0)),
        }
    except Exception as e:
        return stem, folder, "error", {"mesh_path": mesh_path,
                                       "reason": f"{type(e).__name__}: {e}",
                                       "watertight_before": "", "winding_before": "",
                                       "watertight_after": "", "winding_after": ""}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="rebuild existing .npz")
    ap.add_argument("--pin-val", default=None,
                    help="JSON with a 'val' list of stems that MUST be held out")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed, torch_too=False)
    data_dir = resolve(cfg.paths.data)
    cache_dir = resolve(cfg.paths.cache)
    cache_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "manifest.csv") as f:
        rows = [r for r in csv.DictReader(f) if not r["error"]]
    if args.limit:
        rows = rows[: args.limit]

    raw_bounds = np.array(json.load(open(data_dir / "global_bounds.json"))["min"]), \
        np.array(json.load(open(data_dir / "global_bounds.json"))["max"])
    gb = padded_bounds(np.stack(raw_bounds), float(cfg.prep.bbox_pad))
    print(f"padded global bbox: {np.round(gb[0],4).tolist()} .. {np.round(gb[1],4).tolist()}")

    jobs = []
    for i, r in enumerate(rows):
        if not args.force and (cache_dir / f"{r['stem']}.npz").exists():
            continue
        jobs.append((r["stem"], r["folder"], r["mesh_path"], r["pc_path"], cfg.seed * 100003 + i))
    print(f"{len(rows)} pairs, {len(jobs)} to build")

    results = []
    if jobs:
        with Pool(int(cfg.prep.workers), initializer=_init,
                  initargs=(dict(cfg.prep), gb.tolist(), str(cache_dir))) as pool:
            for res in tqdm(pool.imap_unordered(_prep_one, jobs, chunksize=2),
                            total=len(jobs), desc="prep"):
                results.append(res)

    ok = [r for r in results if r[2] == "ok"]
    bad = [r for r in results if r[2] != "ok"]
    # Re-running with the cache already built (to re-derive the split, say) builds
    # nothing, and rewriting the reject log from an empty result list would erase
    # the record of WHY the cache is smaller than the manifest. Only the run that
    # produced the rejects gets to write them.
    rejects_preserved = not jobs and (data_dir / "rejects.csv").exists()
    if rejects_preserved:
        with open(data_dir / "rejects.csv") as f:
            n_rejected = sum(1 for _ in csv.DictReader(f))
        print(f"nothing to build; keeping the existing rejects.csv ({n_rejected} rows)")
    else:
        n_rejected = len(bad)
        with open(data_dir / "rejects.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=REJECT_FIELDS)
            w.writeheader()
            for stem, folder, status, info in bad:
                w.writerow({"stem": stem, "folder": folder,
                            "mesh_path": info.get("mesh_path", ""),
                            "reason": f"[{status}] {info.get('reason','')}",
                            **{k: info.get(k, "") for k in REJECT_FIELDS[4:]}})

    # ---------------------------------------------------------------- split 90/10
    kept = {}
    for r in rows:
        if (cache_dir / f"{r['stem']}.npz").exists():
            kept.setdefault(r["folder"], []).append(r["stem"])

    # A stem named in --pin-val is held out, always. Use it when a previous run's
    # validation set has to stay held out: without it a fresh 90/10 scatters those
    # stems across train and val, and any comparison against that earlier run is
    # then scored partly on shapes the new model trained on. Pinning is applied
    # WITHIN the folder, so the 90/10 rule still holds per folder wherever the
    # pinned count allows.
    pinned = set()
    if args.pin_val:
        with open(resolve(args.pin_val)) as f:
            d = json.load(f)
        pinned = {s for s in (d["val"] if isinstance(d, dict) else d)}
        print(f"pinned val stems: {len(pinned)} from {args.pin_val}")

    rng = np.random.default_rng(cfg.seed)
    split = {"train": [], "val": [], "by_folder": {}, "pinned_val_source": args.pin_val or ""}
    pin_report = {}
    for folder in sorted(kept):
        stems = sorted(kept[folder])
        n_val = max(1, int(round(len(stems) * float(cfg.prep.val_frac))))
        pin = sorted(s for s in stems if s in pinned)
        rest = [s for s in stems if s not in pinned]
        idx = rng.permutation(len(rest))
        need = max(0, n_val - len(pin))
        val = sorted(pin + [rest[i] for i in idx[:need]])
        tr = sorted(rest[i] for i in idx[need:])
        if len(pin) > n_val:
            print(f"!! {folder}: {len(pin)} pinned val stems exceeds the 90/10 target "
                  f"of {n_val}; all pinned stems are held out and this folder's val "
                  f"share becomes {len(val)/len(stems):.3f}")
        pin_report[folder] = {"n_cached": len(stems), "n_val_target": n_val,
                              "n_pinned": len(pin), "n_val": len(val),
                              "val_frac_actual": len(val) / max(1, len(stems))}
        split["train"] += tr
        split["val"] += val
        split["by_folder"][folder] = {"train": tr, "val": val}
    split["train"].sort()
    split["val"].sort()
    missing = sorted(pinned - set(split["val"]))
    split["pinned_not_in_cache"] = missing
    if missing:
        print(f"!! {len(missing)} pinned stem(s) are not in the cache and cannot be "
              f"held out: {missing[:5]}")
    with open(data_dir / "split.json", "w") as f:
        json.dump(split, f, indent=2)

    occ_all = [i["occ_rate"] for _, _, _, i in ok] or [float("nan")]
    occ_near = [i["occ_rate_near"] for _, _, _, i in ok] or [float("nan")]
    occ_unif = [i["occ_rate_unif"] for _, _, _, i in ok] or [float("nan")]
    n_repaired = sum(i.get("repaired", 0) for _, _, _, i in ok)
    summary = {
        "n_pairs": len(rows),
        "n_built_this_run": len(ok),
        "n_rejected": n_rejected,
        "rejects_csv_preserved_from_earlier_run": rejects_preserved,
        "n_cached_total": sum(len(v) for v in kept.values()),
        "n_repaired_for_labeling": n_repaired,
        "reject_rate": n_rejected / max(1, len(jobs)) if jobs else "",
        "padded_global_bbox": {"min": gb[0].tolist(), "max": gb[1].tolist()},
        "occ_rate_mean": float(np.mean(occ_all)),
        "occ_rate_near_mean": float(np.mean(occ_near)),
        "occ_rate_uniform_mean": float(np.mean(occ_unif)),
        "n_train": len(split["train"]), "n_val": len(split["val"]),
        "pin_val_source": args.pin_val or "",
        "pinned_val": pin_report,
        "pinned_not_in_cache": missing,
        "folders": {k: {"train": len(v["train"]), "val": len(v["val"])}
                    for k, v in split["by_folder"].items()},
    }
    with open(data_dir / "prep_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ PREP QUERIES ================")
    print(f"built {len(ok)}  rejected {n_rejected}  cached total {summary['n_cached_total']}")
    print(f"repaired for labeling      : {n_repaired}")
    print(f"occupancy rate  all/near/uniform: {summary['occ_rate_mean']:.3f} / "
          f"{summary['occ_rate_near_mean']:.3f} / {summary['occ_rate_uniform_mean']:.3f}")
    print(f"split: train {len(split['train'])}  val {len(split['val'])} "
          f"across {len(split['by_folder'])} folder(s)")
    print("=============================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
