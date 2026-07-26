# Data and architecture — `runs/20260725_131613`, checkpoint `ckpt/best.pt`

Every number below is read from an artifact on disk. The file it came from is named
in the "source" column or in the paragraph. Anything the artifacts do not record is
printed as **NOT RECORDED** rather than inferred.

The shipped checkpoint `../pc2mesh-release/checkpoints/pc2mesh_v3.pt` carries weights
that are bit-identical to `runs/20260725_131613/ckpt/best.pt` (all 104 tensors compare
equal under `torch.equal`); it differs only by two added metadata keys, `global_bounds`
and `provenance`. So this document describes both files.

---

## 1. Corpus

### 1.1 All 22 discovered folders

`data/corpus_inventory.csv` has one row per discovered `.stl` folder — 22 rows.
`data/corpus_inventory_summary.json` totals them at **89,874 files**, of which
**89,869** loaded (`n_error` 4, `n_oversize` 1).

| # | discovered folder | `n_stl` | `n_loaded` | disposition |
|---|---|---:|---:|---|
| 1 | `1551_original_grains` | 1,545 | 1,545 | EXCLUDED — duplicate |
| 2 | `I41.02_4x_1ero_1vss_STLs` | 11,355 | 11,355 | INCLUDED |
| 3 | `I43.01_BH_2xDS_STLs` | 10,692 | 10,692 | INCLUDED |
| 4 | `I43.05_BH_4xDS_STLs` | 1,482 | 1,482 | INCLUDED |
| 5 | `IDOX_prill_1_STLs` | 29,696 | 29,692 | INCLUDED (4 load errors) |
| 6 | `IP_01_STLs` | 14,161 | 14,160 | INCLUDED (1 oversize) |
| 7 | `PointCloud_Remesh_PostProcess` | 2 | 2 | EXCLUDED — pipeline output |
| 8 | `STLs` | 3,314 | 3,314 | EXCLUDED — un-normalized source |
| 9 | `STLs_normalized` | 1,250 | 1,250 | INCLUDED (legacy) |
| 10 | `STLs_normalized_repeat` | 2,064 | 2,064 | EXCLUDED — duplicate |
| 11 | `alshibli_1551_STLs` | 1,551 | 1,551 | INCLUDED |
| 12 | `prill_1_0.25-scale_STLs` | 7,690 | 7,690 | INCLUDED |
| 13 | `prill_1_0.25-scale_subvolumes/00_subvol_downscale-5/00_subvol_downscale-5_STLs` | 104 | 104 | INCLUDED — pooled |
| 14 | `prill_1_0.25-scale_subvolumes/01_subvol_downscale-2/01_subvol_downscale-2_STLs` | 1,624 | 1,624 | INCLUDED — pooled |
| 15 | `prill_1_0.25-scale_subvolumes/02_subvol_downscale-2/02_subvol_downscale-2_STLs` | 1,508 | 1,508 | INCLUDED — pooled |
| 16 | `prill_1_0.25-scale_subvolumes/03_subvol_downscale-10/03_subvol_downscale-10_STLs` | 26 | 26 | INCLUDED — pooled |
| 17 | `prill_1_0.25-scale_subvolumes/04_subvol_downscale-5/04_subvol_downscale-5_STLs` | 102 | 102 | INCLUDED — pooled |
| 18 | `prill_1_0.25-scale_subvolumes/05_subvol_downscale-2/05_subvol_downscale-2_STLs` | 1,557 | 1,557 | INCLUDED — pooled |
| 19 | `prill_1_0.25-scale_subvolumes/06_subvol_downscale-5/06_subvol_downscale-5_STLs` | 95 | 95 | INCLUDED — pooled |
| 20 | `prill_1_0.25-scale_subvolumes/07_subvol_downscale-10/07_subvol_downscale-10_STLs` | 18 | 18 | INCLUDED — pooled |
| 21 | `prill_1_0.25-scale_subvolumes/08_subvol_downscale-10/08_subvol_downscale-10_STLs` | 20 | 20 | INCLUDED — pooled |
| 22 | `prill_1_0.25-scale_subvolumes/09_subvol_downscale-10/09_subvol_downscale-10_STLs` | 18 | 18 | INCLUDED — pooled |
| | **total** | **89,874** | **89,869** | |

Rows 13–22 are ten subdirectories that `data/corpus_plan.json` pools into one source
folder, `prill_1_0.25-scale_subvolumes` (104 + 1,624 + 1,508 + 26 + 102 + 1,557 + 95 +
18 + 20 + 18 = **5,072** files). So the 22 discovered folders become **9 source
folders**: 18 rows included, 4 rows excluded.

`data/corpus_plan.json` also lists `runs/**` as an exclusion ("this pipeline's own
predicted meshes"). It is not one of the 22 inventory rows and has no file count.

### 1.2 Exclusions and the measured evidence

All overlap fractions below are from `data/corpus_duplicates.json`, whose geometric
signature is `(n_faces, sorted bbox extents to 1e-4)` after `process=True`
(`tools/dup_check.py:13`).

**`1551_original_grains` — duplicate of `alshibli_1551_STLs`.**
`geom_shared` = 1,530. `frac_of_a_in_b` = **0.9902912621359223** (99.03% of
`1551_original_grains`) and `frac_of_b_in_a` = **0.9864603481624759** (98.65% of
`alshibli_1551_STLs`). `sha1_shared` = **0** — the same grains re-exported, so byte
identity finds nothing and only the geometric signature does. Both folders are
internally clean (`n_internal_geometric_duplicates` = 0 in each). Neither shares any
geometry with the other six scanned folders (all remaining pair fractions are 0.0,
except a single mesh shared with `STLs_normalized` and one with
`STLs_normalized_repeat`, fractions 0.00065 / 0.00065).

**`STLs_normalized_repeat` — duplicate of `STLs_normalized`.**
`sha1_shared` = **1,028** and `geom_shared` = **1,028**; `frac_of_b_in_a` =
**0.9980582524271845**, i.e. 99.81% of this folder's 1,030 distinct meshes are
byte-identical to a file in `STLs_normalized`. `frac_of_a_in_b` = **0.8846815834767642**.
Internally it is 2,064 files over 1,030 distinct meshes —
`n_internal_geometric_duplicates` = **1,034**.

**`STLs` — the un-normalized source of `STLs_normalized`.**
Its measured cross-folder overlap with `STLs_normalized` is **`geom_shared` = 0,
`frac_of_a_in_b` = 0.0, `frac_of_b_in_a` = 0.0** — normalization rescales every mesh
to max-extent 1, so the extent-based signature cannot match across the two folders by
construction. The exclusion therefore does **not** rest on an overlap fraction. The
measured evidence that does support it is internal: 3,314 files over 1,168 distinct
geometries, `n_internal_geometric_duplicates` = **2,146**. `data/corpus_plan.json`
states the remaining basis as stem identity ("all 1250 `STLs_normalized` stems appear
here"); the per-stem comparison behind that claim is **NOT RECORDED** as a fraction in
`data/corpus_duplicates.json`.

**`PointCloud_Remesh_PostProcess` — pipeline output.**
2 files. It is not present in `data/corpus_duplicates.json` at all, so its duplicate
overlap fraction is **NOT RECORDED**. `data/corpus_inventory.csv` records the two files
at 1,200 and 21,654 faces with `n_components_med` 1.

**Coverage limit of the duplicate scan.** `data/corpus_duplicates.json` covers 8
folders. `I41.02_4x_1ero_1vss_STLs`, `I43.01_BH_2xDS_STLs`, `I43.05_BH_4xDS_STLs`,
`IP_01_STLs` and `PointCloud_Remesh_PostProcess` were not scanned, so their
cross-folder overlap with anything is **NOT RECORDED**.

**One included pair does overlap.** `prill_1_0.25-scale_STLs` × `IDOX_prill_1_STLs`:
`geom_shared` = 1,226, `frac_of_a_in_b` = **0.234730997511009**, `frac_of_b_in_a` =
**0.05293609671848014**. Both folders are included. `data/corpus_plan.json` cites the
0.053 direction ("only 5.3% signature overlap"); the other direction is 0.235, i.e.
23.5% of `prill_1_0.25-scale_STLs`'s distinct geometries also appear in
`IDOX_prill_1_STLs`. Both are stated here because only one appears in the plan.

### 1.3 Included folders, end to end

Sources: `n_stl`/`n_loaded` from `data/corpus_inventory.csv`; `n_loadable`,
`n_unique_geometry`, `n_selected` from `data/corpus_build_summary.json`; selected count
cross-checked against row counts in `data/corpus_build.csv` and `data/manifest.csv`;
rejected from `data/rejects.csv`; cached from `data/prep_summary.json`; train/val from
`data/split.json`.

| source folder | files | loadable | unique after geom dedup | selected | rejected | cached | train | val | cap applied |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `STLs_normalized` (legacy) | 1,250 | 1,250 | see note | 1,250 | 209 | 1,041 | 937 | 104 | none — kept whole |
| `alshibli_1551_STLs` | 1,551 | 1,551 | 1,551 | 600 | 0 | 600 | 540 | 60 | 600 |
| `IDOX_prill_1_STLs` | 29,696 | 29,692 | 27,218 | 600 | 0 | 600 | 540 | 60 | 600 |
| `prill_1_0.25-scale_STLs` | 7,690 | 7,690 | 6,079 | 600 | 0 | 600 | 540 | 60 | 600 |
| `prill_1_0.25-scale_subvolumes` | 5,072 | 5,072 | 2,967 | 600 | 0 | 600 | 540 | 60 | 600 |
| `I41.02_4x_1ero_1vss_STLs` | 11,355 | 11,355 | 8,832 | 600 | 0 | 600 | 540 | 60 | 600 |
| `I43.01_BH_2xDS_STLs` | 10,692 | 10,692 | 10,645 | 600 | 0 | 600 | 540 | 60 | 600 |
| `I43.05_BH_4xDS_STLs` | 1,482 | 1,482 | 1,477 | 600 | 0 | 600 | 540 | 60 | 600 |
| `IP_01_STLs` | 14,161 | 14,160 | 11,905 | 600 | 0 | 600 | 540 | 60 | 600 |
| **total** | **82,949** | **82,944** | — | **6,050** | **209** | **5,841** | **5,257** | **584** | |

The cap is `cap_per_new_folder: 600` in `data/corpus_build_summary.json`, applied to
each of the eight new folders after deduplication. Every one of the eight had more than
600 unique meshes, so the cap bound in all eight cases. The legacy folder is uncapped:
`n_legacy_symlinked` = 1,250.

Notes on the two fields that are not a plain count:

- **`STLs_normalized` unique after geometric dedup: NOT RECORDED.**
  `data/corpus_build_summary.json` has `"n_unique_geometry": null` for this folder, with
  `"note": "symlinked as-is, not rebuilt"` — deduplication was never run on it. The
  nearest recorded measurement is `data/corpus_duplicates.json`, which puts it at 1,250
  files / 1,187 distinct sha1 / **1,162 distinct geometries**, i.e.
  `n_internal_geometric_duplicates` = 88. That measurement did not gate anything: all
  1,250 entered.
- **`STLs_normalized` selected = 1,250.** The `n_selected` field reads `0` for this
  folder because it counts meshes *rebuilt*, and the legacy corpus was symlinked. The
  1,250 figure is `n_legacy_symlinked` in the same file, and is confirmed independently
  by 1,250 `STLs_normalized` rows in `data/manifest.csv`.

**Two dedup numbers exist and they differ.** `data/corpus_build_summary.json`
(`n_unique_geometry`) and `data/corpus_duplicates.json` (`n_distinct_geometry`) disagree
where both are present — IDOX 27,218 vs 23,160; `prill_1_0.25-scale_STLs` 6,079 vs
5,223; subvolumes 2,967 vs 2,534; `alshibli_1551_STLs` 1,551 vs 1,551. The signatures
are different: selection keyed on `(n_faces, ext_x, ext_y, ext_z)` rounded to 4 dp,
**unsorted** (`tools/build_corpus.py:219-220`), while the duplicate scan keys on
`(n_faces, sorted extents to 1e-4)` after `process=True` (`tools/dup_check.py:13`).
Sorting the extents merges axis-permuted copies, so it always finds fewer distinct
meshes. The table above reports the **selection** number, because that is the one the
cap was applied to.

### 1.4 Reconciliation

Required identity: selected − rejected = cached = train + val.

| | value | source |
|---|---:|---|
| selected | 6,050 | `n_shapes_total`, `data/corpus_build_summary.json`; also 6,050 rows in `data/manifest.csv` and `n_pairs`, `data/prep_summary.json` |
| rejected | 209 | `n_rejected`, `data/prep_summary.json`; 209 data rows in `data/rejects.csv` |
| selected − rejected | **5,841** | |
| cached | **5,841** | `n_cached_total`, `data/prep_summary.json` |
| train | 5,257 | `n_train`, `data/prep_summary.json`; `len(split["train"])` = 5,257 |
| val | 584 | `n_val`, `data/prep_summary.json`; `len(split["val"])` = 584 |
| train + val | **5,841** | |

**The totals reconcile exactly**, and they reconcile per folder as well: 1,250 − 209 =
1,041 = 937 + 104 for `STLs_normalized`, and 600 − 0 = 600 = 540 + 60 for each of the
eight particle folders. Summing `data/split.json`'s `by_folder` lists independently
gives 5,257 train and 584 val, matching the top-level lists.

`data/prep_summary.json` records `pinned_not_in_cache: []` — the 104 legacy validation
stems pinned from `data/legacy_val.json` were all present in the cache, so the legacy
val set is intact and the split is 90/10 within every folder
(`val_frac_actual` 0.1 everywhere; 0.09990393852065321 for the legacy folder).

---

## 2. Rejections

**209 rejections**, `reject_rate` **0.034545454545454546** (`data/prep_summary.json`).

**From which folders — all 209 from `STLs_normalized`.** Grouping the 209 data rows of
`data/rejects.csv` by the `folder` column gives exactly one group:
`{"STLs_normalized": 209}`. The eight particle folders contributed zero rejections;
`data/corpus_build_summary.json` independently records `n_reject: 0` and `n_error: 0`
across the 4,800 build jobs.

**Reason distribution — one reason, 209/209 (100%).**

| reason (verbatim from `rejects.csv`) | n | fraction |
|---|---:|---:|
| `[reject] not watertight/winding-consistent after fill_holes+fix_normals` | 209 | 1.000 |

Breaking the same rows down by the four state columns:

| `watertight_before` | `winding_before` | `watertight_after` | `winding_after` | n |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 1 | 13 |
| 0 | 1 | 0 | 1 | 196 |

Every rejected mesh failed on **watertightness**, not winding: `watertight_after` is 0
for all 209, while `fix_normals` succeeded in making winding consistent for all 209
(including the 13 that started winding-inconsistent). The repair budget is one hole-fill
pass (`prep.max_hole_fill_passes: 1`, `runs/20260725_131613/config.yaml`) and it is
applied for labeling only. `data/prep_summary.json` records
`n_repaired_for_labeling: 14` — the meshes the repair rescued, as against the 209 it
could not.

---

## 3. Supervision

From `runs/20260725_131613/config.yaml` (`prep:` and `train:`) and
`data/prep_summary.json`.

| quantity | value | source |
|---|---|---|
| queries cached per shape | **100,000** | `prep.n_queries` |
| queries drawn per shape per training step | **2,048** | `train.queries_per_shape`; `queries_per_shape` in `run_info.json` |
| near-surface fraction | **0.75** | `prep.near_frac` — 75,000 near / 25,000 uniform per shape (derived from `n_queries × near_frac`; the split is not stored as two counts) |
| uniform fraction | **0.25** | complement of the above; drawn uniformly in the padded global bbox |
| near-surface sigmas | **[0.003, 0.01]**, mixed 50/50 within the near fraction | `prep.near_sigmas` and its config comment |
| labeling method | **igl fast winding number** | `prep.wn_threshold` comment: "igl fast winding number -> occupancy" |
| occupancy threshold | **0.5** | `prep.wn_threshold` |
| labeling repair | fill_holes → fix_normals, **1 pass max**, labeling only — never applied to the geometry scored against | `prep.max_hole_fill_passes: 1` and its comment |
| shapes repaired for labeling | **14** | `n_repaired_for_labeling`, `data/prep_summary.json` |
| val fraction | **0.1**, taken within each folder | `prep.val_frac` |

The same `near_frac` is read by both the cache builder and `dataset.sample_batch`, so
batch composition matches cache composition (config comment at `prep.near_frac`).

**Measured occupancy of the resulting labels** (`data/prep_summary.json`):

| pool | mean occupancy rate |
|---|---:|
| all queries | 0.3783484711522 |
| near-surface | 0.48930106488614955 |
| uniform | 0.045490689950350965 |

**Global bbox.** `data/global_bounds.json` — "union of mesh and point-cloud bounds over
all matched pairs; consumers apply their own padding":

| axis | min | max | extent |
|---|---:|---:|---:|
| x | −0.703830897808075 | 0.7588662505149841 | 1.4626971483230591 |
| y | −0.6663352847099304 | 0.6971907615661621 | 1.3635260462760925 |
| z | −0.6953426599502563 | 0.7275621891021729 | 1.4229048490524292 |

**Padding: `prep.bbox_pad: 0.05`** — 5% of each axis's extent, added at both ends (each
axis widens by 10%). The padded box actually used, from
`data/prep_summary.json.padded_global_bbox`:

| axis | min | max | extent |
|---|---:|---:|---:|
| x | −0.776965755224228 | 0.8320011079311371 | 1.6089668631553651 |
| y | −0.7345115870237351 | 0.7653670638799668 | 1.4998786509037019 |
| z | −0.7664879024028778 | 0.7987074315547943 | 1.5651953339576721 |

`meshify.bbox_pad` is the same 0.05, so the decode grid at inference covers exactly the
box the uniform queries were drawn from.

---

## 4. Architecture

**Every value in this section is read from `model_cfg` inside
`runs/20260725_131613/ckpt/best.pt`**, not from `config.yaml`. The two happen to agree
today, but the checkpoint is the authority. The release checkpoint carries an identical
`model_cfg`.

| hyperparameter | value |
|---|---|
| `n_points` | 1000 |
| `n_centers` | 64 |
| `use_normals` | True |
| `n_freq` | 32 |
| `d_model` | 256 |
| `n_heads` | 4 |
| `n_layers` | 6 |
| `mlp_ratio` | 4 |
| `dropout` | 0.0 |
| `fourier_max_freq` | 64.0 |

That is the complete `model_cfg` dict — ten keys, no others.

**Exact parameter count: 5,893,889.** This is `n_params` in the checkpoint, it matches
`n_params` in `runs/20260725_131613/logs/run_info.json`, and summing `.numel()` over all
104 tensors in the checkpoint's `model` state dict reproduces it exactly.

| part | parameters | share | tensors |
|---|---:|---:|---:|
| encoder | **5,053,440** | 85.74% | 84 |
| decoder | **840,449** | 14.26% | 20 |
| **total** | **5,893,889** | 100% | 104 |

Encoder, by tensor family:

| block | parameters |
|---|---:|
| `embed.proj` (195 → 256) | 50,176 |
| `cross` (LN-q, LN-k, MHA 256, out-proj) | 264,192 |
| `blocks.0`–`blocks.5`, 6 × 789,760 | 4,738,560 |
| `norm` | 512 |
| **encoder total** | **5,053,440** |

Each of the 6 PreLN blocks is: `norm` 512 + self-attention (`in_proj` 196,608 + 768,
`out_proj` 65,536 + 256) + `mlp.norm` 512 + `fc1` 256→1024 (262,144 + 1,024) + `fc2`
1024→256 (262,144 + 256) = 789,760.

Decoder, by tensor family:

| block | parameters |
|---|---:|
| `embed.proj` (192 → 256) | 49,408 |
| `cross` (LN-q, LN-k, MHA 256, out-proj) | 264,192 |
| `mlp` (norm + 256→1024→256) | 526,080 |
| `norm` | 512 |
| `head` (256 → 1) | 257 |
| **decoder total** | **840,449** |

Two input widths are worth reading off the weight shapes, because they confirm two
config flags independently of the config file:

- `encoder.embed.proj.weight` is **(256, 195)**. 195 = 3 axes × 32 frequencies × 2
  (sin, cos) = 192, plus 3 raw normal components. The extra 3 columns are `use_normals:
  True` visible in the weights.
- `decoder.embed.proj.weight` is **(256, 192)** — query points carry position only, no
  normal.

**M = 64 is a latent-set size and does not change the parameter count.** `n_centers`
sets how many FPS centres the encoder's cross-attention pools the 1,000 input points
into, i.e. the *sequence length* of the latent set. It is a shape of an activation, not
of a weight. No tensor in the 104-tensor state dict has 64 in its shape — the only
dimensions present are 1, 195, 192, 256, 768 and 1024. Changing M changes cost and
capacity per shape; it leaves all 5,893,889 parameters exactly where they are. (This
matters here because a second run at M = 128 was planned and cancelled; had it produced
a checkpoint, its parameter count would have been the same 5,893,889.)

---

## 5. Training

Sources: `runs/20260725_131613/logs/run_info.json`, `runs/20260725_131613/config.yaml`,
and the checkpoint's own scalars.

**Optimizer**

| | value | source |
|---|---|---|
| optimizer | **AdamW** | `pc2mesh/train.py:179` |
| learning rate (base) | 1.0e-3 | `train.lr` |
| betas | (0.9, 0.99) | `train.betas` |
| weight decay | 0.01 | `train.weight_decay` |
| eps | **NOT RECORDED** | not set in `config.yaml`; `train.py` does not pass it |
| gradient clip | 1.0 | `train.grad_clip` |
| loss | `BCEWithLogitsLoss` | `pc2mesh/train.py:182` |
| autocast dtype | bf16 | `amp_dtype`, `run_info.json` |
| seed | 0 | `seed`, `run_info.json` |
| batch | 32 shapes × 2,048 queries = 65,536 queries/step | `batch_shapes`, `queries_per_shape`, `run_info.json` |

**Schedule** — linear warmup → constant plateau → cosine to `min_lr`.

| | value |
|---|---|
| warmup | 300 steps (linear from 0 to base lr) |
| plateau | held at base lr until the probe fires |
| probe trigger | wall clock, `schedule_probe_seconds: 240` (not the step-count fallback `schedule_probe_step: 2000`) |
| probe fired at | step **1,222**, wall **240.11 s** |
| steady-state rate measured by the probe | **4.715 steps/s** |
| fitted total steps | **16,554** |
| wall-clock budget / margin | 60 min × 0.97 |
| cosine | from step 1,222 to step 16,554 — 15,332 decaying steps |
| min lr | 1.0e-5 |
| early stopping | `patience_evals: 15` on val BCE — did not trigger |
| hard cap | `max_steps: 200000` — not binding |

**Augmentation** (`train.augment`), applied to the training clouds:

| | value |
|---|---|
| rotation | `rotate_mode: yaw_tilt` — free yaw about z, plus ±15° tilt about a random horizontal axis (`tilt_deg: 15.0`, `up_axis: 2`) |
| anisotropic scale | ±10% per axis (`scale_aniso: 0.10`) |
| point jitter | σ = 0.003 (`jitter_sigma`) |
| point dropout | keep 90–100% of points, pad by resampling (`dropout_keep_min: 0.9`, `dropout_keep_max: 1.0`) |
| legacy `rotate: true` | unused while `rotate_mode` is set |

Full SO(3) is available (`rotate_mode: so3`) and was not used.

**Run**

| | value | source |
|---|---:|---|
| steps run / planned | **16,554 / 16,554** | `steps_run`, `total_steps_planned` |
| wall clock | **2,739.4652936458588 s** = 45.66 min | `wall_s` |
| throughput | **6.042785078678204 steps/s** | `steps_per_s` |
| — shapes/s | 193.4 (= 6.0428 × 32) | derived |
| — queries/s | ≈ 3.96 × 10⁵ (= 6.0428 × 65,536) | derived |
| total shape-presentations | 529,728 (= 16,554 × 32) ≈ 100.8 epochs over 5,257 train shapes | derived |
| total queries seen | 1,084,882,944 | derived |
| best val BCE | **0.3344535529613495** | `best_val_bce`; `val_bce` in the checkpoint agrees |
| best step | **16,500** of 16,554 | `best_step`; `step` in the checkpoint agrees |
| val accuracy at best | 0.8316707611083984 | `val_acc`, checkpoint |
| stop reason | **`reached total_steps`** | `stop_reason` |
| cosine fraction completed | **1.0** | `cosine_fraction_completed` |
| final LR | **1.0000010391478069e-05** | `final_lr` — the cosine landed on `min_lr` |
| smoke run | False | `smoke`, `run_info.json` and checkpoint |

The run finished 753 s under its 3,492 s budget because the steady-state rate it
actually sustained (6.043 steps/s) was higher than the 4.715 steps/s the probe measured,
so the fitted length was conservative. The cosine still completed — `stop_reason` is
`reached total_steps`, not a wall-clock truncation, and the final LR sits on `min_lr`.

---

## 6. Inference operating point

`runs/20260725_131613/config.yaml` is the snapshot taken at training time and has **no
`infer:` block** — under it, `pc2mesh/infer.py:148-154` would fall back to
`remesh.target_faces`, which in that snapshot is 8,000 (the sweep budget). The shipped
operating point was chosen afterwards and lives in `pc2mesh/config.yaml`, which is the
file the release copy ships. Both files are cited per row.

| | value | source |
|---|---|---|
| grid resolution | **128³** | `meshify.resolution`, both configs |
| grid bounds | global bbox padded by **0.05** — the same padded box the queries were drawn from | `meshify.bbox_pad` |
| decode chunk | 200,000 query points per forward pass | `meshify.chunk` |
| marching-cubes level | **0.0**, on the logit field | `meshify.level` |
| **shell clamp** | outer 1-voxel shell forced to **−1.0e4** | `meshify.shell_logit` |
| **floater policy** | drop connected components smaller than **0.001 (0.1%)** of the largest; on by default, disabled with `--no-drop-floaters` | `meshify.floater_volume_frac`; `pc2mesh/infer.py:399` |
| **face budget** | **4,000** faces, quadric decimation, revert-whole-on-break | `infer.target_faces`, `pc2mesh/config.yaml:180-181` (absent from the run snapshot) |
| — sweep budget in the run snapshot | 8,000 | `remesh.target_faces`, `runs/20260725_131613/config.yaml:167` |
| — revert log | `logs/remesh_reverts.csv` | `remesh.reverts_csv` |
| **tet backend** | **gmsh** | `backend`, `logs/tets_remesh_4000.json` |
| **tet backend version** | **4.15.2** | `backend_version`, same file |
| backends probed | `tetgen: None`, `gmsh: 4.15.2`, `wildmeshing: None` — gmsh was the only one importable | `backends_probed`, same file |
| tet algorithm | Delaunay 3D (`Mesh.Algorithm3D = 1`), run per closed shell, `Mesh.Optimize` and `Mesh.OptimizeNetgen` on | `pc2mesh/tetrahedralize.py:88-133` |
| tet size factor / optimize / volume tolerance | 1.0 / True / 0.01 | `size_factor`, `optimize`, `volume_tol`, `logs/tets_remesh_4000.json` |

**Measured at this operating point**, on the 584 held-out shapes:

| | value | source |
|---|---:|---|
| mean faces per shape | 4,000.0 (2,336,000 total) | `pred_faces_mean`, `runs/20260725_131613/logs/gates_remesh_4000.json` |
| floaters dropped during evaluation | True | `drop_floaters`, same file |
| IoU grid used for scoring | 128³ | `gate_iou_resolution`, same file |
| tetrahedralization success | **577 / 584 = 0.988013698630137** (7 failures) | `logs/tets_remesh_4000.json`, `overall` |
| median tets per shape | 8,383 | same |
| volume agreement (surface vs tets) | `volume_ok_rate` 1.0 at tolerance 0.01 | same |
| tet wall clock | 61.05 s over 28 workers | same |

---

## Source index

| file | used for |
|---|---|
| `data/corpus_inventory.csv` | the 22 discovered folders, file counts, load errors |
| `data/corpus_inventory_summary.json` | 89,874 / 89,869 totals |
| `data/corpus_duplicates.json` | duplicate overlap fractions, internal duplicate counts |
| `data/corpus_plan.json` | include/exclude decisions and the pooling of the 10 subvolume dirs |
| `data/corpus_build.csv` | 4,800 built rows, all `status: ok` |
| `data/corpus_build_summary.json` | `n_loadable`, `n_unique_geometry`, `n_selected`, the 600 cap, `n_legacy_symlinked`, `n_shapes_total` |
| `data/manifest.csv` | 6,050 pairs actually staged, per folder |
| `data/prep_summary.json` | rejections, cache totals, occupancy rates, padded bbox, per-folder train/val |
| `data/rejects.csv` | the 209 rejections, folder and reason distribution |
| `data/split.json` | per-folder train/val stem lists |
| `data/global_bounds.json` | the unpadded global bbox |
| `runs/20260725_131613/logs/run_info.json` | steps, wall clock, throughput, best val BCE, stop reason, schedule fit, final LR |
| `runs/20260725_131613/config.yaml` | supervision recipe, optimizer, schedule, augmentation, meshify constants |
| `runs/20260725_131613/ckpt/best.pt` | `model_cfg`, parameter count, encoder/decoder split, best step, val BCE, val accuracy |
| `runs/20260725_131613/logs/gates_remesh_4000.json` | face count and evaluation settings at the shipped budget |
| `logs/tets_remesh_4000.json` | tet backend, version, probe results, success rate |
| `pc2mesh/config.yaml` | `infer.target_faces` — the shipped face budget, absent from the run snapshot |
| `pc2mesh/train.py`, `pc2mesh/infer.py`, `pc2mesh/tetrahedralize.py`, `tools/build_corpus.py`, `tools/dup_check.py` | optimizer class, flag defaults, dedup signatures, tet algorithm |
