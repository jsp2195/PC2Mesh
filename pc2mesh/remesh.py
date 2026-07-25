"""Stage 5b — decimate a directory of surfaces to a fixed face budget, unbroken.

Marching cubes at R=128 emits ~36k faces per shape, laid out along the grid rather
than along the surface. This module asks one question and only one: what does a
uniform quadric decimation to a fixed budget cost, measured by the same
pre-registered gates that scored the undecimated output? Nothing here retrains,
re-decodes, or touches a threshold; the input is STLs that already exist.

`pc2mesh.infer` calls `remesh_one` directly, at the `infer.target_faces` operating
point. This CLI is for sweeping the budget over a whole directory.

Three steps, no smoothing and no subdivision:

    quadric decimation -> trimesh(process=True) -> repair.fix_normals

`is_watertight` and `is_winding_consistent` are re-checked after EVERY step. If
either regresses, the shape reverts WHOLE — the input mesh is written out
untouched — and the stem is appended to the reverts CSV with the step that broke
it. Reverting only the offending step would leave half-remeshed geometry in the
output directory and make the revert count mean nothing, so the unit of revert is
the shape, not the step. Every step runs on a copy, so a revert really is a
revert: the last good state is never mutated in place.

Decimation removes geometry and cannot add any. IoU and normal consistency below
the input are the price of the face budget, not a regression to explain away. The
number that would be a finding is the revert count; it was 0 of 584 shapes at
every budget tried (4,000 / 8,000 / 16,000).

    python -m pc2mesh.remesh --in-dir runs/<ts>/meshes/floaters_dropped \
        --out-dir runs/<ts>/meshes/remesh_4000 --target-faces 4000
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pc2mesh.common import load_config, load_mesh, resolve, set_seed  # noqa: E402

REVERT_FIELDS = [
    "stem", "step", "reason", "faces_in", "faces_step",
    "watertight_after", "winding_after",
]


def flags(mesh) -> tuple[bool, bool]:
    """(is_watertight, is_winding_consistent) — the two properties that must survive.

    A property that raises is treated as False: an unanswerable mesh is not a
    passing mesh.
    """
    try:
        return bool(mesh.is_watertight), bool(mesh.is_winding_consistent)
    except Exception:
        return False, False


def decimate(mesh, target_faces: int):
    """Quadric decimation to `target_faces`. A mesh already under budget is returned
    unchanged rather than being resampled up to it."""
    if len(mesh.faces) <= int(target_faces):
        return mesh
    return mesh.simplify_quadric_decimation(int(target_faces))


def reprocess(mesh):
    """`trimesh process=True` on the current geometry — the vertex merge that makes
    watertightness meaningful after decimation has rewritten the triangle soup."""
    import trimesh

    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices).copy(),
                           faces=np.asarray(mesh.faces).copy(), process=True)


def fix_normals(mesh):
    """`repair.fix_normals` on a COPY, so the pre-step mesh stays revertible."""
    import trimesh

    m = mesh.copy()
    trimesh.repair.fix_normals(m)
    return m


STEPS = (("decimate", decimate), ("process", reprocess), ("fix_normals", fix_normals))


def remesh_one(mesh, target_faces: int):
    """Run the three steps, stopping at the first one that breaks the mesh.

    Returns (mesh_to_write, failure). `failure` is None when all three steps held
    the invariant; otherwise it describes the step that regressed and the mesh
    returned is the INPUT, untouched.
    """
    faces_in = len(mesh.faces)
    wt, wc = flags(mesh)
    if not (wt and wc):
        # Nothing to protect and nothing this module can fix: pass it through so the
        # output directory still holds one mesh per input, and say so.
        return mesh, {"step": "input", "reason": "input already not watertight/winding consistent",
                      "faces_in": faces_in, "faces_step": faces_in,
                      "watertight_after": int(wt), "winding_after": int(wc)}

    cur = mesh
    for name, fn in STEPS:
        try:
            cand = fn(cur, target_faces) if name == "decimate" else fn(cur)
        except Exception as e:
            return mesh, {"step": name, "reason": f"{type(e).__name__}: {e}",
                          "faces_in": faces_in, "faces_step": "",
                          "watertight_after": "", "winding_after": ""}
        wt, wc = flags(cand)
        if len(cand.faces) == 0 or not (wt and wc):
            reason = ("empty mesh" if len(cand.faces) == 0
                      else "not watertight" if not wt else "winding inconsistent")
            return mesh, {"step": name, "reason": reason, "faces_in": faces_in,
                          "faces_step": len(cand.faces),
                          "watertight_after": int(wt), "winding_after": int(wc)}
        cur = cand
    return cur, None


_G = {}


def _init(target, out_dir):
    _G.update(target=int(target), out_dir=Path(out_dir))


def _remesh_file(path_str):
    """One shape, start to finish. Returns (stem, faces_in, faces_out, under, failure).

    Split out of the loop so the shapes can run in a Pool: with ~600 held-out
    shapes and three face budgets the serial version is over an hour, and every
    shape is independent -- the only shared state is the reverts list, which is
    collected in the parent and sorted there.
    """
    p = Path(path_str)
    stem = p.stem
    target = _G["target"]
    base = load_mesh(p, process=True)
    faces_in = len(base.faces)
    under = int(faces_in <= target)
    out, failure = remesh_one(base, target)
    dest = _G["out_dir"] / f"{stem}.stl"
    out.export(dest)

    # The artifact is the FILE, not the object. STL is a float32 triangle soup,
    # so the invariant is re-checked on what was actually written — export is a
    # step like any other, and it is the last one that can break a mesh.
    if failure is None:
        wt, wc = flags(load_mesh(dest, process=True))
        if not (wt and wc):
            failure = {"step": "export",
                       "reason": "not watertight" if not wt else "winding inconsistent",
                       "faces_in": faces_in, "faces_step": len(out.faces),
                       "watertight_after": int(wt), "winding_after": int(wc)}
            out = base
            base.export(dest)

    wt, wc = flags(load_mesh(dest, process=True))
    emitted_broken = not (wt and wc)
    if failure is not None:
        failure["stem"] = stem
        failure.setdefault("faces_in", faces_in)
    return stem, faces_in, len(out.faces), under, failure, emitted_broken


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--in-dir", required=True, help="directory of STLs to decimate")
    ap.add_argument("--out-dir", default=None,
                    help="defaults to <in-dir>/../remesh_<target_faces>")
    ap.add_argument("--target-faces", type=int, default=0,
                    help="override cfg.remesh.target_faces")
    ap.add_argument("--reverts-csv", default=None,
                    help="override cfg.remesh.reverts_csv")
    ap.add_argument("--glob", default="*.stl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    workers = args.workers or int(cfg.scan.workers)
    target = int(args.target_faces or cfg.remesh.target_faces)
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir.parent / f"remesh_{target}"
    out_dir.mkdir(parents=True, exist_ok=True)
    reverts_csv = Path(args.reverts_csv) if args.reverts_csv else resolve(cfg.remesh.reverts_csv)
    reverts_csv.parent.mkdir(parents=True, exist_ok=True)

    paths = sorted(in_dir.glob(args.glob))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no meshes matching {args.glob} under {in_dir}")

    reverts, faces_in, faces_out, n_under = [], [], [], 0
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(target, str(out_dir))) as pool:
        for stem, fin, fout, under, failure, broken in tqdm(
                pool.imap_unordered(_remesh_file, [str(p) for p in paths], chunksize=1),
                total=len(paths), desc=f"remesh[{target}]"):
            faces_in.append(fin)
            faces_out.append(fout)
            n_under += under
            if broken:
                # Only reachable when the reverted-to input was itself broken. Writing
                # it keeps one mesh per shape, but it is not something to discover
                # later in a metrics table.
                print(f"!! {stem}: emitted mesh is not watertight/winding consistent "
                      f"— the INPUT was already broken")
            if failure is not None:
                reverts.append(failure)
    dt = time.time() - t0

    with open(reverts_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVERT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(reverts, key=lambda r: r["stem"]))

    n = len(paths)
    by_step: dict[str, int] = {}
    for r in reverts:
        by_step[r["step"]] = by_step.get(r["step"], 0) + 1
    print(f"\n================ REMESH (target {target} faces) ================")
    print(f"in       : {in_dir}")
    print(f"out      : {out_dir}")
    print(f"shapes   : {n}  ({dt:.1f}s over {workers} workers, "
          f"{dt*workers/max(1,n):.2f}s of work each)")
    print(f"faces    : {np.mean(faces_in):.0f} mean in -> {np.mean(faces_out):.0f} mean out "
          f"({np.sum(faces_in)} -> {np.sum(faces_out)} total)")
    print(f"already under budget: {n_under}/{n} shapes (decimation is a no-op for these)")
    print(f"reverts  : {len(reverts)}/{n} = {100.0*len(reverts)/n:.1f}%"
          + (f"  ({', '.join(f'{k} {v}' for k, v in sorted(by_step.items()))})" if by_step else ""))
    print(f"logged to: {reverts_csv}")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
