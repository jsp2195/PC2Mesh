# Figures

Every figure here was produced from the shipped checkpoint's own run
(`pc2mesh_v3`, 584 held-out shapes) by the code in this repository. None is a
diagram; all of them are measurements.

| file | what it shows | produced by |
|---|---|---|
| `qualitative.png` | 6 held-out shapes sampled evenly across the IoU ranking, best (0.992) to worst (0.263). Cloud, prediction, ground truth, shared camera. | `main.py figures` → `qualitative.png` |
| `per_folder_gates.png` | All seven gates, per source folder, each bar against its pre-registered threshold and labelled with its verdict. | per-folder aggregation of `eval_gates` output |
| `iou_by_folder.png` | The IoU@128 distribution per source folder. | per-folder aggregation of `eval_gates` output |
| `remesh_tradeoff.png` | What the face budget costs: gates at the marching-cubes output vs 4,000 / 8,000 / 16,000 faces. | `remesh` sweep scored by `eval_gates` |
| `remesh_qualitative.png` | The same shapes before and after decimation, so the budget can be judged by eye and not only by IoU. | `remesh` sweep |

## Regenerating them from your own run

```bash
python main.py figures --run runs/<ts>
```

writes five figures into `runs/<ts>/figs/`: `loss_curves.png`, `gate_bars.png`,
`iou_hist.png`, `chamfer_vs_facecount.png` and `qualitative.png`.

It needs that run's `eval_gates` output — `logs/gates_<tag>.json`,
`logs/eval_per_shape_<tag>.csv` and `meshes/<tag>/`. Four of the five figures
score a prediction against its ground truth, which is exactly what does not exist
at inference time, so this is a reporting tool and not part of `infer`. Run:

```bash
python main.py train
python -m pc2mesh.eval_gates --ckpt runs/<ts>/ckpt/best.pt --drop-floaters
python main.py figures --run runs/<ts>
```

`--tag` selects which evaluation to plot (default `floaters_dropped`).
`--ablation` additionally renders an R=128 vs R=256 comparison, and needs a second
evaluation at that resolution:

```bash
python -m pc2mesh.eval_gates --ckpt runs/<ts>/ckpt/best.pt --drop-floaters \
    --resolution 256 --out-tag r256_floaters_dropped
python main.py figures --run runs/<ts> --ablation
```

Colour follows a fixed palette, loss and accuracy are separate panels rather than
a dual axis, and every gate bar carries its verdict as text — the status red/green
pair is not distinguishable under common colour-vision deficiencies (ΔE 4.1), so
no figure here lets meaning rest on colour alone.
