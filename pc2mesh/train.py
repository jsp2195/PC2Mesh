"""Stage 4 — train the occupancy net.

BCEWithLogitsLoss only; there are no auxiliary losses anywhere in this file.

1250 shapes is a small corpus, so augmentation carries the run: every step each
shape gets a fresh rotation (`train.augment.rotate_mode`), anisotropic scale,
jitter and point dropout, with the rotation and scale applied identically to the
queries and the inverse-transpose applied to the normals (see dataset.py and
tests/test_augmentation.py).

    python pc2mesh/train.py --smoke     # GATE: overfit 8 shapes, then meshify one
    python pc2mesh/train.py             # full run, sized to fill the time budget
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pc2mesh.common import CONFIG_PATH, load_config, resolve, set_seed  # noqa: E402
from pc2mesh.dataset import OccData, load_split  # noqa: E402
from pc2mesh.model import build_model, count_params  # noqa: E402

LOG_FIELDS = ["step", "wall_s", "lr", "train_bce", "val_bce", "val_acc", "best_val_bce"]


def lr_at(step: int, base_lr: float, min_lr: float, warmup: int, total: int,
          cos_start: int | None = None) -> float:
    """Linear warmup -> (optional) constant plateau -> cosine decay to min_lr.

    The plateau exists so the run length can be fitted from steady-state
    throughput instead of from the much faster cold-GPU rate; with
    cos_start == warmup this is exactly warmup-then-cosine.
    """
    cos_start = warmup if cos_start is None else max(warmup, cos_start)
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step < cos_start:
        return base_lr
    if total <= cos_start:
        return min_lr
    t = min(1.0, (step - cos_start) / max(1, total - cos_start))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))


def _stratified(split, n: int) -> list[str]:
    """`n` train stems taken round-robin across source folders, deterministically.

    Falls back to the plain head of the train list when the split has no folder
    breakdown, so an older split.json still means what it meant.
    """
    by = split.get("by_folder")
    if not by:
        return split["train"][:n]
    pools = [sorted(v["train"]) for _, v in sorted(by.items()) if v["train"]]
    out, i = [], 0
    while len(out) < n and any(i < len(p) for p in pools):
        for p in pools:
            if i < len(p) and len(out) < n:
                out.append(p[i])
        i += 1
    return out


def make_run_dir(cfg, tag: str = "", run_dir: str | None = None,
                 config_path=None) -> Path:
    """`run_dir` pins the directory so a caller can nohup straight into its logs/.

    The copied config is the one that ACTUALLY ran: copying the default path while
    --config pointed somewhere else would record a config the run never used.
    """
    if run_dir:
        run = Path(run_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{tag}" if tag else "")
        run = resolve(cfg.paths.runs) / ts
    for sub in ("ckpt", "logs", "figs", "meshes"):
        (run / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path or CONFIG_PATH, run / "config.yaml")
    return run


@torch.no_grad()
def evaluate(model, batches, lossf, device, amp_dtype):
    model.eval()
    tot_loss = tot_correct = tot_n = 0.0
    for pc, q, occ in batches:
        with torch.autocast(device, dtype=amp_dtype, enabled=device == "cuda"):
            logits = model(pc, q)
        logits = logits.float()
        tot_loss += lossf(logits, occ).item() * occ.numel()
        tot_correct += ((logits > 0).float() == occ).sum().item()
        tot_n += occ.numel()
    model.train()
    return tot_loss / tot_n, tot_correct / tot_n


def train(cfg, args) -> int:
    device = args.device
    smoke = args.smoke
    set_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp_dtype = torch.bfloat16 if str(cfg.train.amp_dtype).lower() == "bf16" else torch.float16

    split = load_split(cfg)
    if smoke:
        # STRATIFIED across source folders. `split["train"][:8]` is 8 shapes from
        # whichever folder sorts first, which under one corpus was the whole corpus
        # and under nine is one ninth of it -- the probe would then say nothing
        # about the frame, sign convention or label polarity of the other eight.
        # The GATE THRESHOLD is untouched: watertight and IoU@128 >= 0.50 on the
        # first shape, exactly as pre-registered.
        train_stems = _stratified(split, int(cfg.train.smoke.n_shapes))
        val_stems = train_stems  # deliberately: the smoke gate is an overfit check
        batch_shapes = min(int(cfg.train.batch_shapes), len(train_stems))
        total_steps = int(cfg.train.smoke.steps)
        warmup = int(cfg.train.smoke.warmup_steps)
        log_every = int(cfg.train.smoke.log_every)
        time_budget = None
        if not bool(cfg.train.smoke.get("augment", True)):
            # see config.yaml: an overfit probe must actually overfit
            cfg["train"]["augment"] = {"rotate_mode": "none", "rotate": False,
                                       "scale_aniso": 0.0, "jitter_sigma": 0.0,
                                       "dropout_keep_min": 1.0, "dropout_keep_max": 1.0}
            print("[smoke] augmentation disabled for the overfit probe")
    else:
        train_stems, val_stems = split["train"], split["val"]
        batch_shapes = int(cfg.train.batch_shapes)
        total_steps = int(cfg.train.max_steps)   # provisional; fixed at end of warmup
        warmup = int(cfg.train.warmup_steps)
        log_every = int(cfg.train.log_every)
        time_budget = float(cfg.train.time_budget_min) * 60.0
    if args.steps:
        total_steps, time_budget = args.steps, None
    # The plateau exists to fit the run length from STEADY-STATE throughput. A fixed
    # step count only does that if the GPU has actually throttled by then, and how
    # many steps that takes depends on the model: at dropout 0 the fused attention
    # path runs ~2x faster cold, so step 2000 arrives at 96 s -- while the card is
    # still cold -- and the fit overestimates capacity ~2x, which is the exact
    # failure this mechanism exists to prevent. schedule_probe_seconds fires the
    # probe on the wall clock instead, at the same operating point the baseline
    # happened to hit (its step-2000 probe landed at 229 s).
    probe_s = float(cfg.train.get("schedule_probe_seconds", 0) or 0)
    if time_budget is None:
        cos_start = warmup
    elif probe_s > 0:
        cos_start = int(cfg.train.max_steps)   # hold the plateau until the probe fires
    else:
        cos_start = int(cfg.train.schedule_probe_step)
    schedule_fitted = time_budget is None
    fit_info = {}
    warmup_wall = 0.0

    run = make_run_dir(cfg, "smoke" if smoke else "", getattr(args, "run_dir", None),
                       getattr(args, "config", None))
    print(f"run dir: {run}")

    tr = OccData(cfg, train_stems, device)
    va = tr if smoke else OccData(cfg, val_stems, device)
    print(f"train {len(tr)} shapes | val {len(va)} shapes | "
          f"cache resident {(tr.nbytes() + (0 if smoke else va.nbytes()))/1e9:.2f} GB")

    model = build_model(cfg.model).to(device)
    n_params = count_params(model)
    print(f"model: {n_params:,} params ({n_params/1e6:.2f}M)")

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.train.lr),
                            weight_decay=float(cfg.train.weight_decay),
                            betas=tuple(float(b) for b in cfg.train.betas))
    lossf = nn.BCEWithLogitsLoss()

    nq = int(cfg.train.queries_per_shape)
    val_batches = va.fixed_val_batches(int(cfg.train.val_batches), batch_shapes, nq,
                                       seed=cfg.seed + 1)
    print(f"val: {len(val_batches)} fixed un-augmented batches "
          f"({sum(b[2].numel() for b in val_batches):,} queries)")

    gen = torch.Generator(device=device).manual_seed(cfg.seed + 2)
    log_path = run / "logs" / "train_log.csv"
    logf = open(log_path, "w", newline="")
    logw = csv.DictWriter(logf, fieldnames=LOG_FIELDS)
    logw.writeheader()

    best_val = float("inf")
    best_step = -1
    since_improve = 0
    run_loss, run_n = 0.0, 0
    t0 = time.time()
    stop_reason = "max_steps"
    order, order_pos = None, 0
    model.train()

    step = 0
    while step < total_steps:
        # epoch-shuffled shape sampling
        if order is None or order_pos + batch_shapes > len(tr):
            order = torch.randperm(len(tr), generator=gen, device=device)
            order_pos = 0
        idx = order[order_pos:order_pos + batch_shapes]
        order_pos += batch_shapes

        lr = lr_at(step, float(cfg.train.lr), float(cfg.train.min_lr), warmup,
                   total_steps, cos_start)
        for g in opt.param_groups:
            g["lr"] = lr

        pc, q, occ = tr.sample_batch(idx, nq, gen, augment=True)
        with torch.autocast(device, dtype=amp_dtype, enabled=device == "cuda"):
            logits = model(pc, q)
        loss = lossf(logits.float(), occ)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
        opt.step()
        opt.zero_grad(set_to_none=True)

        run_loss += loss.item()
        run_n += 1
        step += 1

        # Fit the run length to the time budget from STEADY-STATE throughput,
        # measured over [warmup, schedule_probe_step] once the GPU has throttled.
        if time_budget is not None and step == warmup:
            warmup_wall = time.time() - t0
        if not schedule_fitted:
            elapsed = time.time() - t0
            ready = ((elapsed >= probe_s and step >= warmup + 200) if probe_s > 0
                     else step == cos_start)
            if ready:
                cos_start = step
                rate = (elapsed - warmup_wall) / max(1, cos_start - warmup)
                budget = time_budget * float(cfg.train.schedule_margin)
                fitted = int(cos_start + max(1, (budget - elapsed) / rate))
                total_steps = min(int(cfg.train.max_steps), fitted)
                schedule_fitted = True
                fit_info = {"probe_seconds": probe_s, "probe_step": cos_start,
                            "probe_wall_s": round(elapsed, 2),
                            "measured_steps_per_s": round(1 / rate, 3),
                            "fitted_total_steps": total_steps}
                print(f"[schedule] probe at {elapsed:.0f}s / step {cos_start}: "
                      f"{rate*1000:.1f} ms/step ({1/rate:.1f} steps/s) -> "
                      f"total_steps = {total_steps:,} to fill "
                      f"{time_budget/60:.0f} min; cosine starts now", flush=True)

        if step % log_every == 0 or step == total_steps:
            vl, vacc = evaluate(model, val_batches, lossf, device, amp_dtype)
            tl = run_loss / max(1, run_n)
            run_loss, run_n = 0.0, 0
            improved = vl < best_val - 1e-6
            if improved:
                best_val, best_step, since_improve = vl, step, 0
                torch.save({"model": model.state_dict(), "model_cfg": dict(cfg.model),
                            "step": step, "val_bce": vl, "val_acc": vacc,
                            "n_params": n_params, "smoke": smoke},
                           run / "ckpt" / "best.pt")
            else:
                since_improve += 1
            logw.writerow({"step": step, "wall_s": round(time.time() - t0, 2),
                           "lr": lr, "train_bce": tl, "val_bce": vl, "val_acc": vacc,
                           "best_val_bce": best_val})
            logf.flush()
            print(f"step {step:6d}/{total_steps}  lr {lr:.2e}  train {tl:.4f}  "
                  f"val {vl:.4f}  acc {vacc:.4f}  best {best_val:.4f}"
                  f"{'  *' if improved else ''}")

            if not smoke and since_improve >= int(cfg.train.patience_evals):
                stop_reason = f"early stop: no val improvement for {since_improve} evals"
                break
        if time_budget is not None and (time.time() - t0) > time_budget:
            stop_reason = "time budget reached"
            break
    else:
        stop_reason = "reached total_steps"

    torch.save({"model": model.state_dict(), "model_cfg": dict(cfg.model),
                "step": step, "n_params": n_params, "smoke": smoke},
               run / "ckpt" / "last.pt")
    logf.close()
    wall = time.time() - t0

    info = {
        "run_dir": str(run), "smoke": smoke, "n_params": n_params,
        "n_train": len(tr), "n_val": len(va), "batch_shapes": batch_shapes,
        "queries_per_shape": nq, "steps_run": step, "total_steps_planned": total_steps,
        "warmup": warmup, "wall_s": wall, "steps_per_s": step / max(1e-9, wall),
        "best_val_bce": best_val, "best_step": best_step, "stop_reason": stop_reason,
        "amp_dtype": str(cfg.train.amp_dtype), "seed": cfg.seed,
        "schedule": fit_info,
        # how far the cosine actually got: 1.0 means it finished
        "cosine_fraction_completed": (
            min(1.0, (step - fit_info["probe_step"])
                / max(1, total_steps - fit_info["probe_step"])) if fit_info else None),
        "final_lr": lr_at(max(0, step - 1), float(cfg.train.lr), float(cfg.train.min_lr),
                          warmup, total_steps, cos_start),
    }
    with open(run / "logs" / "run_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"\ndone in {wall/60:.1f} min | {step} steps | best val BCE {best_val:.4f} "
          f"@ step {best_step} | {stop_reason}")

    if smoke:
        return smoke_gate(cfg, run, va, device)

    with open(resolve(cfg.paths.data) / "last_run.txt", "w") as f:
        f.write(str(run))
    return 0


def smoke_gate(cfg, run: Path, data: OccData, device: str) -> int:
    """GATE: meshify one overfit shape. A recognizable watertight surface or STOP.

    Pre-registered, so it cannot be argued after the fact:
        watertight == True  AND  IoU@128 against the GT mesh >= 0.5
    """
    import csv as _csv

    from pc2mesh.geom import grid_iou
    from pc2mesh.meshify import load_checkpoint, meshify_bounds, meshify_one
    from pc2mesh.common import load_mesh, repair_for_labeling
    from pc2mesh.dataset import load_cloud

    model, _ = load_checkpoint(run / "ckpt" / "best.pt", cfg, device)
    bounds = meshify_bounds(cfg)
    manifest = {r["stem"]: r for r in
                _csv.DictReader(open(resolve(cfg.paths.data) / "manifest.csv"))}

    def score(stem):
        pc = load_cloud(resolve(cfg.paths.cache), stem)
        m = meshify_one(model, pc, cfg, bounds, device, drop_floaters=False)
        if len(m.faces):
            m.export(run / "meshes" / f"smoke_{stem}.stl")
        gt, _, _ = repair_for_labeling(load_mesh(manifest[stem]["mesh_path"], process=True))
        return (m, bool(len(m.faces) and m.is_watertight),
                float(grid_iou(m, gt, bounds, int(cfg.eval.iou_resolution)))
                if len(m.faces) else 0.0, manifest[stem]["folder"])

    # Every probe shape is meshified and reported. The GATE is still the first
    # one and only the first one -- widening it to a mean or a min would be
    # changing a pre-registered threshold -- but with nine source folders the
    # other seven are the only evidence that the other domains decode at all.
    per_shape = [(s,) + score(s) for s in data.stems]
    stem, mesh, wt, iou, _ = per_shape[0]
    out = run / "meshes" / f"smoke_{stem}.stl"
    need_wt = bool(cfg.train.smoke.gate_watertight)
    need_iou = float(cfg.train.smoke.gate_iou128)

    print("\n================ SMOKE GATE ================")
    print(f"{'probe shape':44s} {'folder':32s} {'wt':>4s} {'IoU@128':>8s}")
    for s_, _m, w_, i_, f_ in per_shape:
        print(f"{s_[:44]:44s} {f_[:32]:32s} {str(w_):>4s} {i_:8.4f}"
              + ("   <- GATE" if s_ == stem else ""))
    print(f"\nGATED shape    : {stem}")
    print(f"faces          : {len(mesh.faces)}")
    print(f"watertight     : {wt}")
    print(f"winding cons.  : {bool(len(mesh.faces) and mesh.is_winding_consistent)}")
    print(f"IoU@128 vs GT  : {iou:.4f}   (pre-registered gate: watertight and >= {need_iou:.2f})")
    print(f"mesh           : {out}")
    passed = (wt or not need_wt) and iou >= need_iou
    print("verdict        :", "PASS" if passed else "STOP -- do not proceed to the full run")
    print("============================================")
    with open(run / "logs" / "smoke_gate.json", "w") as f:
        json.dump({"stem": stem, "faces": int(len(mesh.faces)), "watertight": wt,
                   "iou128": float(iou), "passed": bool(passed),
                   "gate": f"watertight={need_wt} and iou128 >= {need_iou}",
                   "probe_shapes_stratified_by_folder": True,
                   "all_probe_shapes": [
                       {"stem": s_, "folder": f_, "faces": int(len(m_.faces)),
                        "watertight": w_, "iou128": i_}
                       for s_, m_, w_, i_, f_ in per_shape]}, f, indent=2)
    return 0 if passed else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true", help="overfit 8 shapes and meshify one")
    ap.add_argument("--steps", type=int, default=0, help="override total steps")
    ap.add_argument("--run-dir", default=None,
                    help="pin the run directory (so the caller can nohup into its logs/)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    return train(load_config(args.config), args)


if __name__ == "__main__":
    raise SystemExit(main())
