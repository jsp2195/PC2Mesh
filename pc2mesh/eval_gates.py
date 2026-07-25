"""Stage 6 — pre-registered PASS / PARTIAL / FAIL gates on the held-out split.

The thresholds live in config.yaml and were fixed before any result was seen.
Nothing in this file adapts a threshold to an outcome; a gate that is missed is
reported as PARTIAL or FAIL.

    watertight              >= 0.98
    winding consistent      >= 0.98
    self-intersections      == 0        (reported as the fraction of meshes with none)
    IoU@128                 >= 0.90     (mean over held-out shapes)
    Chamfer-L2 x1e3         <= 1.0      (mean)
    normal consistency      >= 0.90     (mean)
    Euler match             >= 0.85     (fraction with euler_pred == euler_gt)

Ground truth is the SAME repaired mesh that defined the training labels, so the
score and the supervision cannot disagree about what the shape is.

    python pc2mesh/eval_gates.py --ckpt runs/<ts>/ckpt/best.pt --drop-floaters
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pc2mesh.common import (  # noqa: E402
    load_config, load_mesh, repair_for_labeling, resolve, set_seed,
)

warnings.filterwarnings("ignore")

PER_SHAPE_FIELDS = [
    "stem", "folder", "pred_faces", "gt_faces",
    "watertight", "winding_consistent", "self_intersections", "self_intersection_free",
    "iou128", "chamfer_l2_x1e3", "normal_consistency",
    "euler_pred", "euler_gt", "euler_match",
    "n_components", "n_components_gt", "components_match", "error",
]
# Extra IoU grids are reported alongside the gate, never in place of it: the gate
# is pre-registered at cfg.eval.iou_resolution and that is the number it scores.
EXTRA_IOU_FIELD = "iou{res}"

# metric name -> (per-shape column, aggregation)
GATE_SPEC = {
    "watertight_rate": ("watertight", "mean"),
    "winding_consistent_rate": ("winding_consistent", "mean"),
    "self_intersection_free_rate": ("self_intersection_free", "mean"),
    "iou128_mean": ("iou128", "mean"),
    "chamfer_l2_x1e3_mean": ("chamfer_l2_x1e3", "mean"),
    "normal_consistency_mean": ("normal_consistency", "mean"),
    "euler_match_rate": ("euler_match", "mean"),
}

# Reported BESIDE the gates and never as one. No threshold was pre-registered for
# these, so giving them a PASS/FAIL here would be inventing a gate after seeing
# the data. They exist because exact Euler equality can be unreachable for reasons
# that have nothing to do with the model: scanned-mesh GT genus is frequently
# noise -- speed_boat's GT Euler is -180, i.e. genus 91, which is a scanning
# artefact and not a claim about the boat. Component count is the part of the
# topology that IS meaningful on such ground truth.
DIAG_SPEC = {
    "components_match_rate": "components_match",
    "n_components_mean": "n_components",
    "n_components_gt_mean": "n_components_gt",
    "euler_pred_mean": "euler_pred",
    "euler_gt_mean": "euler_gt",
    "pred_faces_mean": "pred_faces",
}


def diagnostics(rows) -> dict:
    out = {}
    for name, col in DIAG_SPEC.items():
        vals = [r[col] for r in rows if r[col] != "" and not (
            isinstance(r[col], float) and np.isnan(r[col]))]
        out[name] = float(np.mean(vals)) if vals else float("nan")
    return out

_G = {}


def _init(cfg_d, bounds, extra_iou=()):
    _G["cfg"] = cfg_d
    _G["bounds"] = np.asarray(bounds)
    _G["extra_iou"] = list(extra_iou)


def _eval_one(job):
    from pc2mesh.geom import chamfer_l2, count_self_intersections, grid_iou, normal_consistency, sample_surface

    stem, folder, pred_path, gt_path, seed = job
    row = {k: "" for k in PER_SHAPE_FIELDS}
    row.update(stem=stem, folder=folder)
    cfg = _G["cfg"]
    try:
        import trimesh

        pred = trimesh.load(pred_path, process=True, force="mesh")
        gt, _, ok = repair_for_labeling(load_mesh(gt_path, process=True))
        row["gt_faces"] = len(gt.faces)
        row["pred_faces"] = len(pred.faces)
        if len(pred.faces) == 0:
            row["error"] = "empty prediction"
            row.update(watertight=0, winding_consistent=0, self_intersection_free=0,
                       iou128=0.0, chamfer_l2_x1e3=float("nan"),
                       normal_consistency=0.0, euler_match=0, components_match=0)
            for r in _G.get("extra_iou", ()):
                row[EXTRA_IOU_FIELD.format(res=r)] = 0.0
            return row

        row["watertight"] = int(pred.is_watertight)
        row["winding_consistent"] = int(pred.is_winding_consistent)
        row["n_components"] = len(pred.split(only_watertight=False))
        row["n_components_gt"] = len(gt.split(only_watertight=False))
        row["components_match"] = int(row["n_components"] == row["n_components_gt"])
        si = count_self_intersections(pred)
        row["self_intersections"] = si
        row["self_intersection_free"] = int(si == 0)
        row["euler_pred"] = int(pred.euler_number)
        row["euler_gt"] = int(gt.euler_number)
        row["euler_match"] = int(pred.euler_number == gt.euler_number)

        # The GATE uses cfg.eval.iou_resolution and only that. Any extra grid is
        # reported beside it as a diagnostic and never substituted for it.
        row["iou128"] = grid_iou(pred, gt, _G["bounds"], int(cfg["iou_resolution"]))
        for r in _G.get("extra_iou", ()):
            row[EXTRA_IOU_FIELD.format(res=r)] = grid_iou(pred, gt, _G["bounds"], int(r))

        pa, na = sample_surface(pred, int(cfg["chamfer_samples"]), seed)
        pb, nb = sample_surface(gt, int(cfg["chamfer_samples"]), seed + 1)
        row["chamfer_l2_x1e3"] = chamfer_l2(pa, pb) * 1e3
        row["normal_consistency"] = normal_consistency(pa, na, pb, nb)
        return row
    except Exception as e:
        # A shape that could not be scored is a FAILURE, not a shape that never
        # existed. Leaving these blank would drop them from the gate denominator
        # and quietly make every rate gate easier.
        row["error"] = f"{type(e).__name__}: {e}"
        row.update(watertight=0, winding_consistent=0, self_intersection_free=0,
                   iou128=0.0, normal_consistency=0.0, euler_match=0, components_match=0)
        row["chamfer_l2_x1e3"] = float("nan")   # no meaningful distance to report
        for r in _G.get("extra_iou", ()):
            row.setdefault(EXTRA_IOU_FIELD.format(res=r), 0.0)
        return row


def verdict(name, value, gate) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "FAIL"
    if str(gate["direction"]) == "higher":
        if value >= float(gate["pass"]):
            return "PASS"
        return "PARTIAL" if value >= float(gate["partial"]) else "FAIL"
    if value <= float(gate["pass"]):
        return "PASS"
    return "PARTIAL" if value <= float(gate["partial"]) else "FAIL"


def aggregate(rows, gates):
    out = {}
    for name, (col, _agg) in GATE_SPEC.items():
        vals = [r[col] for r in rows if r[col] != "" and not (
            isinstance(r[col], float) and np.isnan(r[col]))]
        v = float(np.mean(vals)) if vals else float("nan")
        g = gates[name]
        out[name] = {
            "value": v, "n": len(vals), "n_missing": len(rows) - len(vals),
            "pass_at": float(g["pass"]), "partial_at": float(g["partial"]),
            "direction": str(g["direction"]), "verdict": verdict(name, v, g),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None,
                    help="required unless --mesh-dir supplies meshes to score")
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--drop-floaters", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mesh-dir", default=None, help="reuse meshes instead of decoding")
    ap.add_argument("--logs-dir", default=None,
                    help="where gates_<tag>.json / eval_per_shape_<tag>.csv go; "
                         "defaults to the checkpoint's run directory")
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--per-shape-tag", default="", help="basename for eval_per_shape_<tag>.csv")
    ap.add_argument("--resolution", type=int, default=0,
                    help="override cfg.meshify.resolution (marching-cubes grid) only")
    ap.add_argument("--extra-iou", type=int, nargs="*", default=[],
                    help="additional IoU grids to REPORT beside the gate (never instead of it)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--stems-file", default=None,
                    help="JSON with a 'val' list (or a bare list) of stems to score "
                         "instead of the whole split -- e.g. to score one earlier "
                         "run's frozen validation set on its own")
    ap.add_argument("--domain-gap-max", type=float, default=0.10,
                    help="STOP if any source folder's iou128 mean falls more than "
                         "this below the overall mean (domain interference)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    from pc2mesh.dataset import load_split
    from pc2mesh.meshify import load_checkpoint, meshify_bounds, meshify_one

    if args.ckpt is None and args.mesh_dir is None:
        raise SystemExit("--ckpt is required unless --mesh-dir supplies meshes to score")
    if args.ckpt is None and args.logs_dir is None:
        raise SystemExit("--logs-dir is required when scoring --mesh-dir without a --ckpt")
    run_dir = (Path(args.logs_dir).resolve().parent if args.ckpt is None
               else Path(args.ckpt).resolve().parent.parent)
    if args.stems_file:
        d = json.load(open(resolve(args.stems_file)))
        want = list(d["val"] if isinstance(d, dict) else d)
        held = set(load_split(cfg)[args.split])
        stems = [s for s in want if s in held]
        skipped = [s for s in want if s not in held]
        print(f"--stems-file {args.stems_file}: {len(want)} requested, "
              f"{len(stems)} are in the '{args.split}' split")
        if skipped:
            # Scoring a shape the model trained on as if it were held out would be
            # the single most damaging thing this file could do quietly.
            print(f"!! {len(skipped)} requested stem(s) are NOT in the '{args.split}' "
                  f"split and are NOT scored: {skipped[:8]}")
    else:
        stems = load_split(cfg)[args.split]
    if int(cfg.eval.n_eval_shapes):
        stems = stems[: int(cfg.eval.n_eval_shapes)]
    if args.limit:
        stems = stems[: args.limit]

    manifest = {r["stem"]: r for r in csv.DictReader(
        open(resolve(cfg.paths.data) / "manifest.csv"))}
    bounds = meshify_bounds(cfg)
    tag = args.out_tag or ("floaters_dropped" if args.drop_floaters else "raw")
    ps_tag = args.per_shape_tag or tag
    res = args.resolution or int(cfg.meshify.resolution)
    workers = args.workers or int(cfg.scan.workers)
    gate_res = int(cfg.eval.iou_resolution)
    gate_col = GATE_SPEC["iou128_mean"][0]
    if gate_col != f"iou{gate_res}":
        raise SystemExit(
            f"config eval.iou_resolution={gate_res} but the gate is registered as "
            f"'{gate_col}' / 'iou128_mean'. The gate name would misdescribe the "
            f"number it scores; update GATE_SPEC and config.eval.gates together.")
    fields = list(PER_SHAPE_FIELDS)
    for r in args.extra_iou:
        f = EXTRA_IOU_FIELD.format(res=r)
        if f == gate_col:
            raise SystemExit(
                f"--extra-iou {r} collides with the gated column '{gate_col}'. An "
                f"extra grid is reported beside the gate and must never be able to "
                f"write the number the gate scores.")
        if f not in fields:
            fields.insert(fields.index("chamfer_l2_x1e3"), f)
    mesh_dir = Path(args.mesh_dir) if args.mesh_dir else run_dir / "meshes" / tag
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- decode (GPU)
    meshify_s = 0.0
    if args.mesh_dir is None:
        from pc2mesh.dataset import load_cloud

        model, _ = load_checkpoint(args.ckpt, cfg, args.device)
        cache = resolve(cfg.paths.cache)
        t0 = time.time()
        for stem in tqdm(stems, desc=f"meshify[{tag}] R={res}"):
            pc = load_cloud(cache, stem)
            m = meshify_one(model, pc, cfg, bounds, args.device, args.drop_floaters,
                            resolution=res)
            m.export(mesh_dir / f"{stem}.stl")
        meshify_s = time.time() - t0
        del model
        print(f"meshify[{tag}] R={res}: {meshify_s:.1f}s total, "
              f"{meshify_s/max(1,len(stems)):.2f}s per shape")

    # ---------------------------------------------------------------- metrics (CPU)
    jobs = [(s, manifest[s]["folder"], str(mesh_dir / f"{s}.stl"),
             manifest[s]["mesh_path"], cfg.seed + 17 * i)
            for i, s in enumerate(stems)]
    rows = []
    t0 = time.time()
    with Pool(workers, initializer=_init,
              initargs=(dict(cfg.eval), bounds.tolist(), args.extra_iou)) as pool:
        for r in tqdm(pool.imap_unordered(_eval_one, jobs, chunksize=1),
                      total=len(jobs), desc="metrics"):
            rows.append(r)
    metrics_s = time.time() - t0
    rows.sort(key=lambda r: r["stem"])
    print(f"metrics: {metrics_s:.1f}s over {workers} workers")

    logs = Path(args.logs_dir) if args.logs_dir else run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with open(logs / f"eval_per_shape_{ps_tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    gates = dict(cfg.eval.gates)
    overall = aggregate(rows, gates)
    per_folder = {}
    for folder in sorted({r["folder"] for r in rows}):
        per_folder[folder] = aggregate([r for r in rows if r["folder"] == folder], gates)

    diag_overall = diagnostics(rows)
    diag_folder = {f: diagnostics([r for r in rows if r["folder"] == f])
                   for f in per_folder}

    # ---- domain interference: aggregates hide a domain that the corpus broke.
    overall_iou = overall["iou128_mean"]["value"]
    domain = {"overall_iou128_mean": overall_iou, "max_gap_allowed": args.domain_gap_max,
              "per_folder": {}, "tripped": []}
    for f, agg in per_folder.items():
        v = agg["iou128_mean"]["value"]
        gap = overall_iou - v
        domain["per_folder"][f] = {"iou128_mean": v, "gap_below_overall": gap,
                                   "n": agg["iou128_mean"]["n"]}
        if gap > args.domain_gap_max:
            domain["tripped"].append({"folder": f, "iou128_mean": v, "gap": gap})
    domain["passed"] = not domain["tripped"]

    extra = {}
    for r in args.extra_iou:
        f = EXTRA_IOU_FIELD.format(res=r)
        vals = [x[f] for x in rows if x.get(f, "") != ""]
        extra[f"iou{r}_mean"] = float(np.mean(vals)) if vals else float("nan")

    result = {
        "ckpt": str(args.ckpt), "split": args.split, "n_shapes": len(rows),
        "n_stems_requested": len(stems),
        # meshes_reused: --mesh-dir scored pre-existing meshes, so meshify_resolution
        # describes the request, not something this invocation produced.
        "meshes_reused": bool(args.mesh_dir is not None),
        "meshify_resolution": (None if args.mesh_dir is not None else res),
        "meshify_resolution_requested": res,
        "gate_iou_resolution": int(cfg.eval.iou_resolution),
        "meshify_seconds": meshify_s, "metrics_seconds": metrics_s,
        "metrics_workers": workers,
        "pred_faces_mean": float(np.mean([r["pred_faces"] for r in rows if r["pred_faces"] != ""])),
        "pred_faces_total": int(np.sum([r["pred_faces"] for r in rows if r["pred_faces"] != ""])),
        "extra_iou_reported": extra,
        "drop_floaters": bool(args.drop_floaters), "mesh_dir": str(mesh_dir),
        "gt_definition": "original STL, process=True, then labeling repair "
                         "(fill_holes -> fix_normals) — identical to training labels",
        "chamfer_definition": "0.5*(mean_x min_y |x-y|^2 + mean_y min_x |x-y|^2) x 1e3, "
                              f"{int(cfg.eval.chamfer_samples)} surface samples each",
        "normal_consistency_definition": "symmetric mean |cos| between a sample's normal "
                                         "and its nearest neighbour's normal",
        "iou_definition": f"winding-number occupancy on a shared "
                          f"{int(cfg.eval.iou_resolution)}^3 grid over the padded global bbox",
        "overall": overall, "per_folder": per_folder,
        "diagnostics_overall": diag_overall, "diagnostics_per_folder": diag_folder,
        "domain_interference": domain,
        "stems_file": args.stems_file or "",
        "n_errors": sum(1 for r in rows if r["error"]),
    }
    with open(logs / f"gates_{tag}.json", "w") as f:
        json.dump(result, f, indent=2)

    # ---------------------------------------------------------------- report
    def table(title, agg):
        print(f"\n--- {title} ---")
        print(f"{'gate':30s} {'value':>10s} {'target':>18s}  verdict")
        for k, v in agg.items():
            op = ">=" if v["direction"] == "higher" else "<="
            print(f"{k:30s} {v['value']:10.4f} {op+' '+format(v['pass_at'],'.2f'):>18s}"
                  f"  {v['verdict']}")

    n_ok = sum(1 for r in rows if not r["error"])
    print(f"\n================ EVAL GATES ({tag}, R={res}) ================")
    print(f"checkpoint : {args.ckpt if args.ckpt else '(none — scoring ' + str(mesh_dir) + ')'}")
    print(f"held-out   : {len(rows)} shapes ({n_ok} scored, {len(rows)-n_ok} errored)")
    if len(rows) - n_ok:
        print(f"!! {len(rows)-n_ok} shape(s) could not be scored and are counted as "
              f"FAILURES in every rate gate; Chamfer is averaged over the "
              f"{overall['chamfer_l2_x1e3_mean']['n']} shapes that produced one")
    table("OVERALL", overall)
    if len(per_folder) > 1:
        for folder, agg in per_folder.items():
            table(f"folder: {folder}", agg)
    print("\n--- diagnostics, reported beside the gates (NOT gates) ---")
    print(f"{'metric':30s} {'overall':>10s}"
          + "".join(f" {f[:14]:>15s}" for f in per_folder))
    for k in DIAG_SPEC:
        print(f"{k:30s} {diag_overall[k]:10.4f}"
              + "".join(f" {diag_folder[f][k]:15.4f}" for f in per_folder))
    if extra:
        print("\n--- extra IoU grids, reported beside the gate (NOT a gate) ---")
        for k, v in extra.items():
            print(f"{k:30s} {v:10.4f}")
    if len(per_folder) > 1:
        print(f"\n--- DOMAIN INTERFERENCE (iou128 mean {overall_iou:.4f}, "
              f"stop if a folder is > {args.domain_gap_max:.2f} below) ---")
        for f, d in sorted(domain["per_folder"].items(), key=lambda kv: kv[1]["iou128_mean"]):
            flag = "  <-- STOP" if d["gap_below_overall"] > args.domain_gap_max else ""
            print(f"{f:44s} {d['iou128_mean']:8.4f}  gap {d['gap_below_overall']:+.4f}{flag}")
        print("verdict:", "PASS" if domain["passed"] else
              "STOP -- domain interference: " + ", ".join(
                  f"{t['folder']} ({t['gap']:+.4f})" for t in domain["tripped"]))
    print(f"\ncost: meshify {meshify_s:.1f}s  metrics {metrics_s:.1f}s  "
          f"mean faces {result['pred_faces_mean']:.0f}")
    counts = {}
    for v in overall.values():
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
    print(f"\nsummary: " + "  ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    print("=" * 52)
    # A missed quality gate is a RESULT and exits 0 -- PARTIAL and FAIL are outcomes
    # this pipeline is allowed to report. Domain interference is different: it says
    # the corpus broke a domain, which invalidates the aggregate rather than scoring
    # it, so it exits non-zero. Only when the whole split was scored: a --stems-file
    # subset has one folder in it and no aggregate to be hidden behind.
    if len(per_folder) > 1 and not domain["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
