"""Stage 7 — figures into runs/<ts>/figs/.

    loss_curves.png            train/val BCE and val accuracy
    gate_bars.png              every gate against its pre-registered threshold
    iou_hist.png               per-shape IoU@128 distribution
    chamfer_vs_facecount.png   is the error driven by GT complexity?
    qualitative.png            6 held-out shapes x (input cloud | predicted | GT)

Colour follows the reference data-viz palette. Loss and accuracy are separate
panels rather than a dual axis, and every gate bar carries its verdict as text so
meaning never rests on colour alone (the status red/green pair is not
CVD-separable, ΔE 4.1).

    python pc2mesh/plots.py --run runs/<ts>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pc2mesh.common import load_config, load_mesh, repair_for_labeling, resolve  # noqa: E402

warnings.filterwarnings("ignore")

# ---- reference palette (light surface)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # categorical slots 1-3
# single-hue sequential ramp (blue 100->700), light = low, dark = high
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
STATUS = {"PASS": "#0ca30c", "PARTIAL": "#fab219", "FAIL": "#d03b3b"}


def style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=1.0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    return ax


def newfig(w, h):
    fig = plt.figure(figsize=(w, h), facecolor=SURFACE)
    return fig


def save(fig, path, dpi):
    fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# --------------------------------------------------------------------------- 1


def plot_loss_curves(run: Path, cfg):
    rows = list(csv.DictReader(open(run / "logs" / "train_log.csv")))
    if not rows:
        return
    step = np.array([int(r["step"]) for r in rows])
    tr = np.array([float(r["train_bce"]) for r in rows])
    va = np.array([float(r["val_bce"]) for r in rows])
    acc = np.array([float(r["val_acc"]) for r in rows])
    best_i = int(np.argmin(va))

    fig = newfig(9, 6.6)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.45, 1], hspace=0.32)

    ax = style(fig.add_subplot(gs[0]), "Occupancy BCE", None, "BCE (nats)")
    ax.plot(step, tr, color=S1, lw=2, label="train")
    ax.plot(step, va, color=S2, lw=2, label="val")
    ax.axvline(step[best_i], color=MUTED, lw=1, ls="--")
    ax.plot([step[best_i]], [va[best_i]], "o", ms=7, color=S2,
            mec=SURFACE, mew=2, zorder=5)
    # keep the annotation clear of the direct labels parked at the right edge
    late = step[best_i] > 0.65 * step[-1]
    ax.annotate(f"best val {va[best_i]:.4f}\n@ step {step[best_i]:,}",
                (step[best_i], va[best_i]), textcoords="offset points",
                xytext=(-12 if late else 10, 26), ha="right" if late else "left",
                color=INK, fontsize=8.5)
    ax.set_xlim(0, step[-1] * 1.14)
    ax.text(step[-1], tr[-1], "  train", color=INK2, fontsize=9, va="center")
    ax.text(step[-1], va[-1], "  val", color=INK2, fontsize=9, va="center")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")

    ax2 = style(fig.add_subplot(gs[1]), "Validation accuracy", "step", "accuracy")
    ax2.plot(step, acc, color=S3, lw=2)
    ax2.set_xlim(0, step[-1] * 1.14)
    ax2.text(step[-1], acc[-1], f"  {acc[-1]:.3f}", color=INK2, fontsize=9, va="center")
    ax2.set_ylim(min(0.7, acc.min() - 0.02), 1.0)

    fig.suptitle("Training curves", color=INK, fontsize=13, x=0.125, ha="left", y=0.97)
    save(fig, run / "figs" / "loss_curves.png", int(cfg.plots.dpi))


# --------------------------------------------------------------------------- 2


def plot_gate_bars(run: Path, cfg, tag: str):
    gpath = run / "logs" / f"gates_{tag}.json"
    if not gpath.exists():
        print(f"  skip gate_bars: {gpath} missing")
        return
    res = json.load(open(gpath))
    ov = res["overall"]
    unit = [(k, v) for k, v in ov.items() if v["direction"] == "higher"]
    other = [(k, v) for k, v in ov.items() if v["direction"] != "higher"]

    fig = newfig(9.5, 5.8)
    gs = fig.add_gridspec(2, 1, height_ratios=[len(unit), max(1, len(other))], hspace=0.45)

    def draw(ax, items, xmax, title, xlabel):
        style(ax, title, xlabel, None)
        names = [k for k, _ in items]
        y = np.arange(len(items))
        for i, (k, v) in enumerate(items):
            ax.barh(i, v["value"], height=0.55, color=STATUS[v["verdict"]],
                    edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.plot([v["pass_at"], v["pass_at"]], [i - 0.36, i + 0.36],
                    color=INK, lw=2, zorder=5)
            # verdict as TEXT: the status palette is not CVD-separable.
            # Park it clear of the threshold tick, whichever of the two is further right.
            x = max(v["value"], v["pass_at"]) + xmax * 0.02
            ax.text(min(x, xmax * 0.99), i, f"{v['value']:.3f}  {v['verdict']}",
                    va="center", ha="left", color=INK, fontsize=8.5, zorder=6)
        ax.set_yticks(y)
        ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=9, color=INK2)
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.grid(axis="y", visible=False)

    draw(fig.add_subplot(gs[0]), unit, 1.18,
         "Rates and quality metrics  (higher is better)", None)
    if other:
        xm = max(max(v["value"] for _, v in other) * 1.45,
                 max(v["pass_at"] for _, v in other) * 1.8)
        draw(fig.add_subplot(gs[1]), other, xm,
             "Chamfer-L2 x1e3  (lower is better)", "value")

    handles = [plt.Line2D([0], [0], color=INK, lw=2, label="pre-registered threshold")]
    fig.legend(handles=handles, frameon=False, fontsize=8.5, labelcolor=INK2,
               loc="lower right", bbox_to_anchor=(0.98, -0.02))
    fig.suptitle(f"Evaluation gates — {res['n_shapes']} held-out shapes ({tag})",
                 color=INK, fontsize=13, x=0.125, ha="left", y=0.99)
    save(fig, run / "figs" / "gate_bars.png", int(cfg.plots.dpi))


# --------------------------------------------------------------------------- 3,4


def load_json(p):
    p = Path(p)
    return json.load(open(p)) if p.exists() else None


def _per_shape(run: Path, tag: str):
    p = run / "logs" / f"eval_per_shape_{tag}.csv"
    if not p.exists():
        return []
    return [r for r in csv.DictReader(open(p)) if not r["error"]]


def plot_iou_hist(run: Path, cfg, tag: str):
    rows = _per_shape(run, tag)
    if not rows:
        print("  skip iou_hist: no per-shape metrics")
        return
    iou = np.array([float(r["iou128"]) for r in rows])
    thr = float(cfg.eval.gates["iou128_mean"]["pass"])

    fig = newfig(8, 4.4)
    ax = style(fig.add_subplot(111), f"IoU@128 across {len(iou)} held-out shapes",
               "IoU@128", "shapes")
    ax.hist(iou, bins=np.linspace(0, 1, 41), color=S1, edgecolor=SURFACE, linewidth=1.2)
    ax.axvline(iou.mean(), color=S2, lw=2)
    ax.axvline(thr, color=INK, lw=2, ls="--")
    top = ax.get_ylim()[1]
    ax.text(iou.mean(), top * 0.97, f" mean {iou.mean():.3f}", color=S2,
            fontsize=9, va="top")
    ax.text(thr, top * 0.86, f" gate {thr:.2f}", color=INK, fontsize=9, va="top")
    ax.text(0.02, top * 0.97, f"median {np.median(iou):.3f}\n"
                              f">= gate: {(iou >= thr).mean():.0%} of shapes",
            color=INK2, fontsize=8.5, va="top")
    ax.set_xlim(0, 1)
    save(fig, run / "figs" / "iou_hist.png", int(cfg.plots.dpi))


def plot_chamfer_vs_facecount(run: Path, cfg, tag: str):
    rows = _per_shape(run, tag)
    if not rows:
        print("  skip chamfer_vs_facecount: no per-shape metrics")
        return
    nf = np.array([float(r["gt_faces"]) for r in rows])
    ch = np.array([float(r["chamfer_l2_x1e3"]) for r in rows])
    ok = np.isfinite(ch) & (nf > 0)
    nf, ch = nf[ok], ch[ok]
    thr = float(cfg.eval.gates["chamfer_l2_x1e3_mean"]["pass"])

    fig = newfig(8, 4.6)
    ax = style(fig.add_subplot(111), "Chamfer-L2 vs ground-truth complexity",
               "GT faces (log scale)", "Chamfer-L2 x1e3")
    ax.scatter(nf, ch, s=26, color=S1, edgecolor=SURFACE, linewidth=1.0, alpha=0.9, zorder=3)
    ax.axhline(thr, color=INK, lw=2, ls="--", zorder=4)
    ax.set_xscale("log")
    ax.text(nf.min(), thr, f" gate {thr:.2f}", color=INK, fontsize=9, va="bottom")
    if len(nf) > 2:
        r = float(np.corrcoef(np.log10(nf), ch)[0, 1])
        ax.text(0.98, 0.95, f"Pearson r (log faces, chamfer) = {r:+.2f}",
                transform=ax.transAxes, ha="right", va="top", color=INK2, fontsize=8.5)
    save(fig, run / "figs" / "chamfer_vs_facecount.png", int(cfg.plots.dpi))


# --------------------------------------------------------------------------- 5


def _decimate(mesh, target=6000):
    if len(mesh.faces) <= target:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(target)
    except Exception:
        return mesh


def _gt_mesh(cfg, manifest, stem):
    g, _, _ = repair_for_labeling(load_mesh(manifest[stem]["mesh_path"], process=True))
    return _decimate(g)


def _pred_mesh(run, tag, stem):
    import trimesh

    p = run / "meshes" / tag / f"{stem}.stl"
    if not p.exists():
        return None
    return _decimate(trimesh.load(p, process=True, force="mesh"))


def _shared_camera(arrays):
    """One cubic box covering everything drawn, so all panels are comparable."""
    allpts = np.concatenate([np.asarray(a) for a in arrays if a is not None and len(a)])
    ctr = (allpts.min(0) + allpts.max(0)) / 2
    half = float((allpts.max(0) - allpts.min(0)).max()) / 2 * 1.05
    return [(ctr[d] - half, ctr[d] + half) for d in range(3)]


def _panel(fig, n_rows, n_cols, idx, cfg, lim, kind, payload, title=None, row_label=None):
    ax = fig.add_subplot(n_rows, n_cols, idx, projection="3d")
    ax.set_facecolor(SURFACE)
    if kind == "cloud" and payload is not None:
        p = np.asarray(payload)
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=2.0, c=S1, depthshade=True, linewidths=0)
    elif payload is not None and len(getattr(payload, "faces", [])):
        v = np.asarray(payload.vertices)
        ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=payload.faces,
                        color=kind, edgecolor="none", linewidth=0,
                        shade=True, antialiased=False)
    ax.set_xlim(*lim[0])
    ax.set_ylim(*lim[1])
    ax.set_zlim(*lim[2])
    ax.set_box_aspect((1, 1, 1))
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.5)
    except TypeError:
        pass
    ax.view_init(elev=float(cfg.plots.elev), azim=float(cfg.plots.azim))
    ax.set_axis_off()
    if title:
        ax.set_title(title, color=INK, fontsize=10, pad=-2)
    if row_label:
        ax.text2D(-0.02, 0.5, row_label, transform=ax.transAxes, rotation=90,
                  va="center", ha="center", color=INK2, fontsize=7.5)
    return ax


def pick_shapes(run: Path, cfg, pick_tag: str, n=None):
    """Deterministic selection: evenly spaced across the IoU ranking, best to worst.

    Selection is driven by ONE tag so the same shapes can be rendered at any
    resolution — a side-by-side comparison is only meaningful on an identical set.
    """
    rows = [r for r in _per_shape(run, pick_tag) if float(r["pred_faces"] or 0) > 0]
    if not rows:
        return []
    rows.sort(key=lambda r: -float(r["iou128"]))
    n = min(n or int(cfg.plots.qualitative_shapes), len(rows))
    return [rows[int(round(i * (len(rows) - 1) / max(1, n - 1)))] for i in range(n)]


def plot_qualitative(run: Path, cfg, tag: str, pick_tag: str | None = None,
                     out_name: str = "qualitative.png", label: str = "",
                     mesh_tag: str | None = None):
    """`tag` names the per-shape CSV; `mesh_tag` names meshes/<dir> when they differ."""
    pick_tag = pick_tag or tag
    mesh_tag = mesh_tag or tag
    pick = pick_shapes(run, cfg, pick_tag)
    if not pick:
        print(f"  skip {out_name}: no per-shape metrics for {pick_tag}")
        return
    by_stem = {r["stem"]: r for r in _per_shape(run, tag)}
    manifest = {r["stem"]: r for r in csv.DictReader(
        open(resolve(cfg.paths.data) / "manifest.csv"))}
    cache = resolve(cfg.paths.cache)

    clouds, preds, gts = [], [], []
    for r in pick:
        s = r["stem"]
        clouds.append(np.load(cache / f"{s}.npz", allow_pickle=True)["pc"].astype(np.float64))
        preds.append(_pred_mesh(run, mesh_tag, s))
        gts.append(_gt_mesh(cfg, manifest, s))
    lim = _shared_camera(clouds + [m.vertices for m in preds + gts if m is not None])

    n = len(pick)
    fig = newfig(9.5, 2.15 * n)
    for i, r in enumerate(pick):
        s = r["stem"]
        iou = float(by_stem.get(s, r)["iou128"])
        for j, (kind, payload, title) in enumerate((
                ("cloud", clouds[i], "input cloud (1000 pts)"),
                (S2, preds[i], f"predicted{label}"),
                (MUTED, gts[i], "ground truth"))):
            _panel(fig, n, 3, i * 3 + j + 1, cfg, lim, kind, payload,
                   title=title if i == 0 else None,
                   row_label=f"{s[:24]}\nIoU {iou:.3f}" if j == 0 else None)
    fig.suptitle(f"Held-out reconstructions{label} — sampled evenly across the IoU "
                 f"ranking (best to worst), shared camera",
                 color=INK, fontsize=12, x=0.5, y=0.995)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.955, bottom=0.005,
                        wspace=-0.02, hspace=-0.02)
    save(fig, run / "figs" / out_name, int(cfg.plots.dpi))


def plot_qualitative_compare(run: Path, cfg, tag_a: str, tag_b: str, pick_tag: str,
                             out_name: str, label_a="R=128", label_b="R=256",
                             stems=None, subtitle: str = "",
                             mesh_tag_a: str | None = None, mesh_tag_b: str | None = None):
    """cloud | tag_a | tag_b | GT for one FIXED shape set, one shared camera.

    tag_* name the per-shape CSVs; mesh_tag_* name meshes/<dir> when they differ
    (the R=256 CSV is eval_per_shape_r256.csv but its meshes live in
    meshes/r256_floaters_dropped/).
    """
    mesh_tag_a = mesh_tag_a or tag_a
    mesh_tag_b = mesh_tag_b or tag_b
    a_rows = {r["stem"]: r for r in _per_shape(run, tag_a)}
    b_rows = {r["stem"]: r for r in _per_shape(run, tag_b)}
    if not a_rows or not b_rows:
        print(f"  skip {out_name}: missing per-shape metrics")
        return
    if stems is None:
        stems = [r["stem"] for r in pick_shapes(run, cfg, pick_tag)]
    stems = [s for s in stems if s in a_rows and s in b_rows]
    if not stems:
        print(f"  skip {out_name}: no overlapping shapes")
        return

    manifest = {r["stem"]: r for r in csv.DictReader(
        open(resolve(cfg.paths.data) / "manifest.csv"))}
    cache = resolve(cfg.paths.cache)
    clouds = [np.load(cache / f"{s}.npz", allow_pickle=True)["pc"].astype(np.float64)
              for s in stems]
    A = [_pred_mesh(run, mesh_tag_a, s) for s in stems]
    B = [_pred_mesh(run, mesh_tag_b, s) for s in stems]
    G = [_gt_mesh(cfg, manifest, s) for s in stems]
    lim = _shared_camera(clouds + [m.vertices for m in A + B + G if m is not None])

    n = len(stems)
    fig = newfig(12.5, 2.15 * n)
    for i, s in enumerate(stems):
        ra, rb = a_rows[s], b_rows[s]
        row_label = (f"{s[:22]}\nIoU@128grid {float(ra['iou128']):.3f} -> "
                     f"{float(rb['iou128']):.3f}\n"
                     f"Euler gt {ra['euler_gt']} | pred {ra['euler_pred']} -> {rb['euler_pred']}")
        for j, (kind, payload, title) in enumerate((
                ("cloud", clouds[i], "input cloud (1000 pts)"),
                (S2, A[i], label_a),
                (S3, B[i], label_b),
                (MUTED, G[i], "ground truth"))):
            _panel(fig, n, 4, i * 4 + j + 1, cfg, lim, kind, payload,
                   title=title if i == 0 else None,
                   row_label=row_label if j == 0 else None)
    fig.suptitle(subtitle or f"{label_a} vs {label_b} on identical held-out shapes "
                             f"— shared camera and axis limits",
                 color=INK, fontsize=12, x=0.5, y=0.995)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.005,
                        wspace=-0.02, hspace=-0.02)
    save(fig, run / "figs" / out_name, int(cfg.plots.dpi))


def euler_transitions(run: Path, tag_a: str, tag_b: str):
    """Per-shape Euler-match transition between two resolutions."""
    a = {r["stem"]: r for r in _per_shape(run, tag_a)}
    b = {r["stem"]: r for r in _per_shape(run, tag_b)}
    out = {"fixed": [], "broken": [], "unchanged_correct": [], "unchanged_wrong": []}
    for s in sorted(set(a) & set(b)):
        ma, mb = int(a[s]["euler_match"]), int(b[s]["euler_match"])
        key = {(0, 0): "unchanged_wrong", (0, 1): "fixed",
               (1, 0): "broken", (1, 1): "unchanged_correct"}[(ma, mb)]
        out[key].append(s)
    return out


def plot_euler_changes(run: Path, cfg, tag_a: str, tag_b: str, n=3,
                       out_name="euler_changes.png",
                       mesh_tag_a: str | None = None, mesh_tag_b: str | None = None):
    """The shapes whose Euler verdict actually moved — they explain the topology result."""
    tr = euler_transitions(run, tag_a, tag_b)
    changed = tr["fixed"] + tr["broken"]
    if not changed:
        print("  skip euler_changes: no shape changed Euler verdict")
        return
    # interleave both directions so the figure is not one-sided
    pick, i = [], 0
    target = min(n, len(changed))
    while len(pick) < target:
        for grp in (tr["fixed"], tr["broken"]):
            if i < len(grp) and len(pick) < target:
                pick.append(grp[i])
        i += 1
    plot_qualitative_compare(
        run, cfg, tag_a, tag_b, tag_a, out_name, stems=pick,
        mesh_tag_a=mesh_tag_a, mesh_tag_b=mesh_tag_b,
        subtitle=f"Shapes whose Euler verdict CHANGED between resolutions "
                 f"({len(tr['fixed'])} fixed, {len(tr['broken'])} broken of "
                 f"{len(set(tr['fixed'])|set(tr['broken'])|set(tr['unchanged_correct'])|set(tr['unchanged_wrong']))})")


# --------------------------------------------------------------------------- ablation


def plot_resolution_ablation(run: Path, cfg, tag_a: str, tag_b: str,
                             ps_a: str, ps_b: str, label_a="R=128", label_b="R=256",
                             out_name="resolution_ablation.png"):
    """(a) every gate at both resolutions, (b) paired per-shape IoU, (c) Euler transitions.

    (b) and (c) are PAIRED: the same 104 held-out shapes at both resolutions, so a
    point-for-point comparison is valid.
    """
    ga = load_json(run / "logs" / f"gates_{tag_a}.json")
    gb = load_json(run / "logs" / f"gates_{tag_b}.json")
    if not ga or not gb:
        print(f"  skip {out_name}: need gates_{tag_a}.json and gates_{tag_b}.json")
        return
    ra = {r["stem"]: r for r in _per_shape(run, ps_a)}
    rb = {r["stem"]: r for r in _per_shape(run, ps_b)}
    shared = sorted(set(ra) & set(rb))

    fig = newfig(12.6, 11.4)
    gs = fig.add_gridspec(3, 2, height_ratios=[2.5, 0.72, 2.25], hspace=0.42, wspace=0.22)

    # ---- (a) grouped bars, unit-scale gates
    unit = [k for k, v in ga["overall"].items() if v["direction"] == "higher"]
    other = [k for k, v in ga["overall"].items() if v["direction"] != "higher"]

    def grouped(ax, keys, xmax, title, xlabel):
        style(ax, title, xlabel, None)
        h = 0.34
        for i, k in enumerate(keys):
            for d, (g, col, lab) in enumerate(((ga, S1, label_a), (gb, S2, label_b))):
                v = g["overall"][k]
                y = i + (d - 0.5) * h * 1.06
                ax.barh(y, v["value"], height=h, color=col, edgecolor=SURFACE,
                        linewidth=1.5, zorder=3, label=lab if i == 0 else None)
                x = max(v["value"], v["pass_at"]) + xmax * 0.02
                ax.text(min(x, xmax * 0.995), y, f"{v['value']:.3f} {v['verdict']}",
                        va="center", ha="left", color=INK, fontsize=7.6, zorder=6)
            t = ga["overall"][k]["pass_at"]
            ax.plot([t, t], [i - 0.44, i + 0.44], color=INK, lw=2, zorder=5)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([k.replace("_", " ") for k in keys], fontsize=8.5, color=INK2)
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.grid(axis="y", visible=False)

    ax_a = fig.add_subplot(gs[0, :])
    grouped(ax_a, unit, 1.22, "(a) Every gate at both resolutions  (higher is better)", None)
    # legend above the axes: inside it would sit on the bars' value labels
    ax_a.legend(frameon=False, fontsize=8.5, labelcolor=INK2, ncol=2,
                loc="lower right", bbox_to_anchor=(1.0, 1.005))
    if other:
        ax_b = fig.add_subplot(gs[1, :])
        xm = max(max(g["overall"][other[0]]["value"] for g in (ga, gb)) * 1.5,
                 ga["overall"][other[0]]["pass_at"] * 1.9)
        grouped(ax_b, other, xm, "Chamfer-L2 x1e3  (lower is better)", "value")

    # ---- (b) paired per-shape IoU scatter
    ax = style(fig.add_subplot(gs[2, 0]),
               f"(b) Per-shape IoU on the {ga['gate_iou_resolution']}-grid, paired "
               f"({len(shared)} shapes)",
               f"IoU at {label_a}", f"IoU at {label_b}")
    xa = np.array([float(ra[s]["iou128"]) for s in shared])
    xb = np.array([float(rb[s]["iou128"]) for s in shared])
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.5, ls="--", zorder=2)
    ax.scatter(xa, xb, s=24, color=S1, edgecolor=SURFACE, linewidth=0.8, alpha=0.9, zorder=3)
    d = xb - xa
    # only the few biggest movers, alternating sides, so labels do not pile up
    movers = [j for j in np.argsort(-np.abs(d))[:4] if abs(d[j]) >= 1e-4]
    for rank, j in enumerate(movers):
        dx, dy = (8, 10) if rank % 2 == 0 else (8, -14)
        ax.annotate(f"{shared[j][:18]} {d[j]:+.3f}", (xa[j], xb[j]),
                    textcoords="offset points", xytext=(dx, dy),
                    fontsize=6.8, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.text(0.03, 0.97,
            f"mean {xa.mean():.4f} → {xb.mean():.4f}  ({xb.mean()-xa.mean():+.5f})\n"
            f"by sign: {(d > 0).sum()} up / {(d < 0).sum()} down / {(d == 0).sum()} equal "
            f"(= {len(d)})\n"
            f"|Δ|>1e-4: {(np.abs(d) > 1e-4).sum()}    |Δ|>0.02: {(np.abs(d) > 0.02).sum()}",
            transform=ax.transAxes, va="top", ha="left", color=INK2, fontsize=8)

    # ---- (c) Euler transition counts, confusion style
    tr = euler_transitions(run, ps_a, ps_b)
    M = np.array([[len(tr["unchanged_wrong"]), len(tr["fixed"])],
                  [len(tr["broken"]), len(tr["unchanged_correct"])]], dtype=float)
    ax = style(fig.add_subplot(gs[2, 1]),
               "(c) Euler match transitions (paired)", f"{label_b} (columns)",
               f"{label_a} (rows)")
    ax.grid(False)
    vmax = max(M.max(), 1)
    for i in range(2):
        for j in range(2):
            frac = M[i, j] / vmax
            # single-hue sequential ramp: light = few, dark = many
            shade = SEQ[min(len(SEQ) - 1, int(frac * (len(SEQ) - 1)))]
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=shade,
                                       edgecolor=SURFACE, linewidth=3, zorder=2))
            ax.text(j, i, f"{int(M[i, j])}\n{M[i,j]/max(1,M.sum()):.0%}",
                    ha="center", va="center", zorder=4, fontsize=13,
                    color="#ffffff" if frac > 0.55 else INK, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["no match", "match"], fontsize=9, color=INK2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["no match", "match"], fontsize=9, color=INK2)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # axes-fraction coords: independent of the inverted y-axis, so it lands BELOW
    ax.text(0.5, -0.16, f"fixed {len(tr['fixed'])}   broken {len(tr['broken'])}   "
                        f"net {len(tr['fixed'])-len(tr['broken']):+d} of {int(M.sum())} shapes",
            transform=ax.transAxes, ha="center", va="top", color=INK, fontsize=9)

    cost = (f"cost: {label_a} meshify {ga.get('meshify_seconds',0):.0f}s, "
            f"{ga.get('pred_faces_mean',0):.0f} faces/mesh   |   "
            f"{label_b} meshify {gb.get('meshify_seconds',0):.0f}s, "
            f"{gb.get('pred_faces_mean',0):.0f} faces/mesh")
    fig.suptitle(f"Resolution ablation — same checkpoint, no retraining\n{cost}",
                 color=INK, fontsize=12.5, x=0.5, y=0.995)
    save(fig, run / "figs" / out_name, int(cfg.plots.dpi))


# --------------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--run", default=None)
    ap.add_argument("--tag", default="floaters_dropped")
    ap.add_argument("--ablation", action="store_true",
                    help="also render the R=128 vs R=256 resolution ablation figures")
    ap.add_argument("--tag-b", default="r256_floaters_dropped")
    ap.add_argument("--ps-b", default="r256")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.run:
        run = Path(args.run)
    else:
        run = Path(open(resolve(cfg.paths.data) / "last_run.txt").read().strip())
    (run / "figs").mkdir(parents=True, exist_ok=True)
    print(f"figures for {run}")

    plot_loss_curves(run, cfg)
    plot_gate_bars(run, cfg, args.tag)
    plot_iou_hist(run, cfg, args.tag)
    plot_chamfer_vs_facecount(run, cfg, args.tag)
    plot_qualitative(run, cfg, args.tag)

    if args.ablation:
        plot_resolution_ablation(run, cfg, args.tag, args.tag_b, args.tag, args.ps_b)
        # SAME shape set as qualitative.png: selection is pinned to the R=128
        # ranking, only the meshes change. Re-picking would break the comparison.
        plot_qualitative(run, cfg, args.ps_b, pick_tag=args.tag,
                         out_name="qualitative_r256.png", label=" @ R=256",
                         mesh_tag=args.tag_b)
        plot_qualitative_compare(run, cfg, args.tag, args.ps_b, args.tag,
                                 "qualitative_r128_vs_r256.png",
                                 mesh_tag_a=args.tag, mesh_tag_b=args.tag_b)
        plot_euler_changes(run, cfg, args.tag, args.ps_b, n=3,
                           mesh_tag_a=args.tag, mesh_tag_b=args.tag_b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
