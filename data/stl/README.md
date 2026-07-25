# Drop STLs here

`main.py prepare` reads every `.stl` under this directory, in any units and any
frame, and writes:

    data/corpus/<subdir>/<stem>.stl    normalized: centroid at the origin, max extent 1.0
    data/clouds/<subdir>/<stem>.npy    (1000, 6) float64, xyz ++ unit face normal

**Subdirectories are folder labels.** One subdirectory per source keeps them
separable: the 90/10 train/val split is taken *within* each folder, and the gate
report breaks out a row per folder, which is the only way to see one domain being
carried by another. Files placed directly here get the label `stl`.

**Stems must be unique across subdirectories.** The query cache is flat, so
`prepare` stops with the two colliding paths rather than letting one overwrite
the other. Comparison is case- and whitespace-insensitive.

**What gets rejected, with the reason recorded in `data/corpus_build.csv`:**
meshes with 0 faces, and meshes that are still not watertight and
winding-consistent after the labeling repair (`fill_holes` → `fix_normals`) —
the winding number cannot label those, so they cannot supervise anything.

This directory is otherwise empty and stays that way in git: `.gitignore`
excludes `*.stl` everywhere except `examples/`.
