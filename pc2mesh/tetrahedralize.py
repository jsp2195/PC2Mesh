"""Stage 6 — fill a watertight triangle surface with tetrahedra.

Input is Stage 5's remeshed output, because a decimated surface is what a solver
would actually be handed: marching cubes at R=128 emits ~34k grid-aligned
triangles per shape and tetrahedralizing that produces an element count set by
the isosurfacing grid rather than by the geometry.

BACKEND ORDER, tried in this order and RECORDED in the output:

    1. tetgen        (python `tetgen`)      — the reference constrained Delaunay
    2. gmsh          (python `gmsh`)        — Delaunay 3D on a discrete surface
    3. wildmeshing   (python `wildmeshing`) — fTetWild, tolerant of bad input

Whichever is importable first is used, and its name and version go into every
output file. If none is importable this stage reports that and stops — it does
not fall back to a convex hull, a voxelization, or anything else that would put
a differently-defined mesh under the same column heading.

GATES REPORTED (this stage pre-registers none of its own; the numbers are
reported as measured):

    generation success       a tet mesh was produced at all
    orientation              signed volumes, counted by sign; "inverted" means a
                             tet whose sign disagrees with the mesh's own
                             convention, and a mesh where every tet is negative
                             is reported as negatively oriented rather than as
                             100% inverted, because that is an ordering
                             convention and not a broken element
    min dihedral angle       per tet, the smallest of its 6 dihedral angles;
                             the distribution over all tets is what is reported
    element count            tets per shape
    volume error             |sum|V_tet| - V_surface| / V_surface, target < 0.01
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
from pc2mesh.common import load_config, load_mesh, resolve, set_seed  # noqa: E402

warnings.filterwarnings("ignore")

BACKENDS = ("tetgen", "gmsh", "wildmeshing")

PER_SHAPE_FIELDS = [
    "stem", "folder", "backend", "status",
    "surf_faces", "surf_vertices", "surf_watertight", "surf_volume",
    "n_shells", "n_nested_shells", "shells_watertight",
    "n_nodes", "n_tets", "tet_volume",
    "n_negative", "n_positive", "n_degenerate", "orientation",
    "n_inverted", "all_one_sign",
    "min_dihedral_deg", "p01_dihedral_deg", "p50_dihedral_deg", "mean_min_dihedral_deg",
    "max_dihedral_deg", "n_tets_below_10deg", "frac_tets_below_10deg",
    "volume_rel_error", "volume_ok", "seconds", "error",
]


def available_backend(preferred: str | None = None):
    """(name, version) of the first importable backend, or (None, None)."""
    order = [preferred] if preferred else list(BACKENDS)
    for name in order:
        try:
            m = __import__(name)
        except Exception:
            continue
        v = getattr(m, "__version__", None)
        if name == "gmsh" and v is None:
            try:
                v = m.GMSH_API_VERSION
            except Exception:
                v = "?"
        return name, str(v or "?")
    return None, None


# ------------------------------------------------------------------ backends

def _gmsh_one_shell(V: np.ndarray, F: np.ndarray, size_factor: float, optimize: bool):
    """Delaunay 3D inside ONE closed shell. Returns (nodes, tets)."""
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.model.add("s")
        surf = gmsh.model.addDiscreteEntity(2)
        tags = np.arange(1, len(V) + 1, dtype=np.int64)
        gmsh.model.mesh.addNodes(2, surf, tags, V.reshape(-1))
        gmsh.model.mesh.addElementsByType(
            surf, 2, [], (F.astype(np.int64) + 1).reshape(-1))
        # The surface is already a closed manifold: reclassify so gmsh treats it as
        # the boundary of a volume instead of a free-floating element soup.
        gmsh.model.mesh.createTopology()
        loop = gmsh.model.geo.addSurfaceLoop([surf])
        gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        # 1 = Delaunay: the robust choice on an isosurface, and the one that does
        # not try to improve the boundary triangles Stage 5 fixed the budget of.
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFactor", float(size_factor))
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1 if optimize else 0)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1 if optimize else 0)
        gmsh.model.mesh.generate(3)

        ntags, coords, _ = gmsh.model.mesh.getNodes()
        nodes = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        remap = np.zeros(int(np.max(ntags)) + 1, dtype=np.int64)
        remap[np.asarray(ntags, dtype=np.int64)] = np.arange(len(ntags))
        etypes, _, enodes = gmsh.model.mesh.getElements(3)
        tets = np.zeros((0, 4), dtype=np.int64)
        for et, en in zip(etypes, enodes):
            if int(et) == 4:  # 4-node tetrahedron
                tets = remap[np.asarray(en, dtype=np.int64).reshape(-1, 4)]
        return nodes, tets
    finally:
        gmsh.finalize()


def _tet_gmsh(V: np.ndarray, F: np.ndarray, size_factor: float, optimize: bool = True):
    """Fill every connected shell of the surface, then concatenate.

    Marching cubes at R=128 routinely emits several disjoint shells even after
    --drop-floaters (7 of them on 30_construction_set), and a single gmsh surface
    loop over all of them defines no volume: gmsh meshes one shell and silently
    reports a solid whose volume is a fraction of the real one. That is exactly
    the failure mode a relative-volume gate exists to catch, so the shells are
    separated here and filled one at a time.
    """
    import trimesh

    m = trimesh.Trimesh(vertices=V, faces=F, process=True)
    parts = m.split(only_watertight=False)
    if len(parts) <= 1:
        parts = [m]
    nodes_all, tets_all, off = [], [], 0
    for p in parts:
        pv = np.ascontiguousarray(p.vertices, dtype=np.float64)
        pf = np.ascontiguousarray(p.faces, dtype=np.int64)
        n, t = _gmsh_one_shell(pv, pf, size_factor, optimize)
        if len(t) == 0:
            continue
        nodes_all.append(n)
        tets_all.append(t + off)
        off += len(n)
    if not tets_all:
        return np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int64)
    return np.concatenate(nodes_all, 0), np.concatenate(tets_all, 0)


def _tet_tetgen(V: np.ndarray, F: np.ndarray, size_factor: float,
                optimize: bool = True):
    import tetgen

    t = tetgen.TetGen(V, F)
    nodes, tets = t.tetrahedralize(order=1, mindihedral=10, minratio=1.5)
    return np.asarray(nodes, dtype=np.float64), np.asarray(tets, dtype=np.int64)


def _tet_wildmeshing(V: np.ndarray, F: np.ndarray, size_factor: float,
                     optimize: bool = True):
    import wildmeshing

    t = wildmeshing.Tetrahedralizer(stop_quality=10)
    t.set_mesh(V, F)
    t.tetrahedralize()
    nodes, tets = t.get_tet_mesh()
    return np.asarray(nodes, dtype=np.float64), np.asarray(tets, dtype=np.int64)


TETRAHEDRALIZE = {"gmsh": _tet_gmsh, "tetgen": _tet_tetgen, "wildmeshing": _tet_wildmeshing}


# ------------------------------------------------------------------ measurement

def signed_volumes(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    a, b, c, d = (nodes[tets[:, i]] for i in range(4))
    return np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6.0


def min_dihedrals(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Smallest of each tet's 6 dihedral angles, in degrees.

    The dihedral along the edge shared by two faces is pi minus the angle between
    their outward normals. Faces are taken as the four vertex triples with the
    opposite vertex left out; the sign convention cancels in the |cos| used here
    because the pairs are enumerated consistently.
    """
    v = nodes[tets]                                   # (T,4,3)
    faces = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    n = np.stack([np.cross(v[:, f[1]] - v[:, f[0]], v[:, f[2]] - v[:, f[0]])
                  for f in faces], axis=1)            # (T,4,3) outward
    n = n / np.maximum(np.linalg.norm(n, axis=2, keepdims=True), 1e-300)
    out = np.full(len(tets), np.inf)
    for i in range(4):
        for j in range(i + 1, 4):
            c = np.clip(-np.einsum("ij,ij->i", n[:, i], n[:, j]), -1.0, 1.0)
            out = np.minimum(out, np.degrees(np.arccos(c)))
    return out


_G = {}


def _init(backend, size_factor, vol_tol, out_dir, write_mesh, optimize):
    _G.update(backend=backend, size_factor=size_factor, vol_tol=vol_tol,
              out_dir=Path(out_dir), write=bool(write_mesh), optimize=bool(optimize))


def shell_report(mesh):
    """(n_shells, n_nested, all_watertight, reference_volume).

    The reference the tet volume is compared against is the sum of the shells'
    own |volume|, which is what filling each shell separately produces. A shell
    that sits INSIDE another is a cavity, and then that sum is wrong -- it adds
    the cavity instead of subtracting it -- so nesting is counted and written to
    the per-shape CSV rather than left to distort a volume error silently.
    """
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return 1, 0, bool(mesh.is_watertight), abs(float(mesh.volume))
    import igl

    cen = np.array([np.asarray(p.bounds, dtype=np.float64).mean(0) for p in parts])
    nested = 0
    for i, p in enumerate(parts):
        others = np.delete(np.arange(len(parts)), i)
        w = igl.fast_winding_number(
            np.ascontiguousarray(p.vertices, dtype=np.float64),
            np.ascontiguousarray(p.faces, dtype=np.int64),
            np.ascontiguousarray(cen[others]))
        nested += int((np.abs(w) > 0.5).sum())
    return (len(parts), nested, bool(all(p.is_watertight for p in parts)),
            float(sum(abs(float(p.volume)) for p in parts)))


def _one(job):
    stem, folder, path = job
    row = {k: "" for k in PER_SHAPE_FIELDS}
    row.update(stem=stem, folder=folder, backend=_G["backend"])
    t0 = time.time()
    try:
        m = load_mesh(path, process=True)
        V = np.ascontiguousarray(m.vertices, dtype=np.float64)
        F = np.ascontiguousarray(m.faces, dtype=np.int64)
        row.update(surf_faces=len(F), surf_vertices=len(V),
                   surf_watertight=int(m.is_watertight))
        if not m.is_watertight:
            row.update(status="skip", error="input surface is not watertight",
                       seconds=time.time() - t0)
            return row
        n_shell, n_nested, shells_wt, vs = shell_report(m)
        row.update(n_shells=n_shell, n_nested_shells=n_nested,
                   shells_watertight=int(shells_wt), surf_volume=vs)

        nodes, tets = TETRAHEDRALIZE[_G["backend"]](V, F, _G["size_factor"],
                                                    _G["optimize"])
        row.update(n_nodes=len(nodes), n_tets=len(tets))
        if len(tets) == 0:
            row.update(status="fail", error="backend produced 0 tetrahedra",
                       seconds=time.time() - t0)
            return row

        sv = signed_volumes(nodes, tets)
        scale = float(np.abs(sv).max())
        deg = int((np.abs(sv) <= 1e-12 * max(scale, 1e-30)).sum())
        neg = int((sv < 0).sum()) - int(((sv < 0) & (np.abs(sv) <= 1e-12 * scale)).sum())
        pos = len(sv) - neg - deg
        # A mesh where every tet is negative is negatively ORIENTED, not 100%
        # broken; what makes an element inverted is disagreeing with its own mesh.
        orientation = "positive" if pos >= neg else "negative"
        n_inv = neg if orientation == "positive" else pos
        row.update(n_negative=neg, n_positive=pos, n_degenerate=deg,
                   orientation=orientation, n_inverted=n_inv + deg,
                   all_one_sign=int(n_inv + deg == 0))

        dih = min_dihedrals(nodes, tets)
        row.update(min_dihedral_deg=float(dih.min()),
                   p01_dihedral_deg=float(np.percentile(dih, 1)),
                   p50_dihedral_deg=float(np.percentile(dih, 50)),
                   mean_min_dihedral_deg=float(dih.mean()),
                   max_dihedral_deg=float(dih.max()),
                   n_tets_below_10deg=int((dih < 10).sum()),
                   frac_tets_below_10deg=float((dih < 10).mean()))

        vt = float(np.abs(sv).sum())
        rel = abs(vt - vs) / max(vs, 1e-30)
        row.update(tet_volume=vt, volume_rel_error=rel,
                   volume_ok=int(rel < _G["vol_tol"]), status="ok",
                   seconds=time.time() - t0)

        if _G["write"]:
            import meshio

            out = _G["out_dir"] / f"{stem}.vtu"
            meshio.write_points_cells(out, nodes, [("tetra", tets)])
        return row
    except Exception as e:
        row.update(status="error", error=f"{type(e).__name__}: {e}",
                   seconds=time.time() - t0)
        return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--in-dir", required=True, help="watertight surfaces to fill")
    ap.add_argument("--out-dir", default=None, help="where .vtu tet meshes go")
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--tag", default="")
    ap.add_argument("--backend", default=None, choices=list(BACKENDS))
    ap.add_argument("--size-factor", type=float, default=1.0,
                    help="gmsh Mesh.MeshSizeFactor; 1.0 lets the surface set the size")
    ap.add_argument("--volume-tol", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-write", action="store_true", help="measure only, emit no .vtu")
    ap.add_argument("--no-optimize", action="store_true",
                    help="skip the backend's own tet-quality optimization pass")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed, torch_too=False)
    in_dir = Path(args.in_dir)
    tag = args.tag or in_dir.name
    out_dir = Path(args.out_dir) if args.out_dir else in_dir.parent / f"tets_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = Path(args.logs_dir)
    logs.mkdir(parents=True, exist_ok=True)
    workers = args.workers or int(cfg.scan.workers)

    name, version = available_backend(args.backend)
    print(f"backend search order: {' -> '.join(BACKENDS)}")
    for b in BACKENDS:
        got, v = available_backend(b)
        print(f"  {b:12s} {'available ' + str(v) if got else 'NOT AVAILABLE'}")
    if name is None:
        msg = ("no tetrahedralization backend is importable offline "
               f"({', '.join(BACKENDS)}); Stage 6 stops here and nothing is written. "
               "No substitute (convex hull, voxelization, surface-only export) is "
               "emitted, because it would not be a tetrahedral mesh of this surface.")
        print(f"\n!! STOP (this stage only): {msg}")
        with open(logs / f"tets_{tag}.json", "w") as f:
            json.dump({"backend": None, "status": "unavailable", "reason": msg}, f, indent=2)
        return 2
    print(f"\nusing backend: {name} {version}")

    manifest = {r["stem"]: r for r in csv.DictReader(
        open(resolve(cfg.paths.data) / "manifest.csv"))}
    paths = sorted(in_dir.glob("*.stl"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no .stl under {in_dir}")
    jobs = [(p.stem, manifest.get(p.stem, {}).get("folder", "?"), str(p)) for p in paths]

    t0 = time.time()
    rows = []
    with Pool(workers, initializer=_init,
              initargs=(name, args.size_factor, args.volume_tol, str(out_dir),
                        not args.no_write, not args.no_optimize)) as pool:
        for r in tqdm(pool.imap_unordered(_one, jobs, chunksize=1),
                      total=len(jobs), desc=f"tet[{name}]"):
            rows.append(r)
    rows.sort(key=lambda r: r["stem"])
    wall = time.time() - t0

    with open(logs / f"tets_per_shape_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PER_SHAPE_FIELDS)
        w.writeheader()
        w.writerows(rows)

    def agg(rs):
        ok = [r for r in rs if r["status"] == "ok"]
        if not rs:
            return {}
        d = {"n": len(rs), "n_ok": len(ok), "success_rate": len(ok) / len(rs),
             "n_skip": sum(1 for r in rs if r["status"] == "skip"),
             "n_fail": sum(1 for r in rs if r["status"] in ("fail", "error"))}
        if ok:
            inv = np.array([r["n_inverted"] for r in ok])
            dih = np.array([r["min_dihedral_deg"] for r in ok])
            nt = np.array([r["n_tets"] for r in ok], dtype=np.float64)
            ve = np.array([r["volume_rel_error"] for r in ok])
            d.update(
                n_meshes_multi_shell=int(sum(1 for r in ok if r["n_shells"] > 1)),
                n_meshes_with_nested_shells=int(sum(1 for r in ok if r["n_nested_shells"])),
                shells_median=float(np.median([r["n_shells"] for r in ok])),
                n_meshes_all_one_sign=int(sum(r["all_one_sign"] for r in ok)),
                frac_meshes_all_one_sign=float(np.mean([r["all_one_sign"] for r in ok])),
                inverted_elements_total=int(inv.sum()),
                n_meshes_negatively_oriented=int(
                    sum(1 for r in ok if r["orientation"] == "negative")),
                min_dihedral_min=float(dih.min()), min_dihedral_p01=float(np.percentile(dih, 1)),
                min_dihedral_median=float(np.median(dih)), min_dihedral_max=float(dih.max()),
                mean_of_mean_min_dihedral=float(
                    np.mean([r["mean_min_dihedral_deg"] for r in ok])),
                frac_tets_below_10deg=float(np.mean([r["frac_tets_below_10deg"] for r in ok])),
                tets_min=int(nt.min()), tets_median=float(np.median(nt)),
                tets_mean=float(nt.mean()), tets_max=int(nt.max()),
                tets_total=int(nt.sum()),
                volume_rel_error_mean=float(ve.mean()),
                volume_rel_error_median=float(np.median(ve)),
                volume_rel_error_max=float(ve.max()),
                volume_ok_rate=float(np.mean([r["volume_ok"] for r in ok])),
                seconds_mean=float(np.mean([r["seconds"] for r in ok])))
        return d

    folders = sorted({r["folder"] for r in rows})
    result = {
        "backend": name, "backend_version": version,
        "backends_probed": {b: (available_backend(b)[1] or None) for b in BACKENDS},
        "in_dir": str(in_dir), "out_dir": str(out_dir) if not args.no_write else None,
        "tag": tag, "volume_tol": args.volume_tol, "size_factor": args.size_factor,
        "optimize": not args.no_optimize,
        "wall_seconds": wall, "workers": workers,
        "overall": agg(rows),
        "per_folder": {f: agg([r for r in rows if r["folder"] == f]) for f in folders},
    }
    with open(logs / f"tets_{tag}.json", "w") as f:
        json.dump(result, f, indent=2)

    o = result["overall"]
    print(f"\n================ TETRAHEDRALIZE ({tag}) ================")
    print(f"backend            : {name} {version}")
    print(f"shapes             : {o['n']}  ok {o['n_ok']}  skip {o['n_skip']}  "
          f"fail {o['n_fail']}   success {o['success_rate']:.4f}")
    if o.get("n_ok"):
        print(f"all-one-sign       : {o['frac_meshes_all_one_sign']:.4f} of meshes "
              f"({o['inverted_elements_total']} inverted elements in total; "
              f"{o['n_meshes_negatively_oriented']} meshes negatively oriented)")
        print(f"min dihedral (deg) : min {o['min_dihedral_min']:.3f}  p01 "
              f"{o['min_dihedral_p01']:.3f}  median {o['min_dihedral_median']:.3f}  "
              f"max {o['min_dihedral_max']:.3f}")
        print(f"  mean over tets   : {o['mean_of_mean_min_dihedral']:.3f}   "
              f"tets below 10 deg: {o['frac_tets_below_10deg']:.4f}")
        print(f"elements           : median {o['tets_median']:.0f}  mean "
              f"{o['tets_mean']:.0f}  min {o['tets_min']}  max {o['tets_max']}  "
              f"total {o['tets_total']:,}")
        print(f"volume rel error   : mean {o['volume_rel_error_mean']:.2e}  median "
              f"{o['volume_rel_error_median']:.2e}  max {o['volume_rel_error_max']:.2e}"
              f"   < {args.volume_tol}: {o['volume_ok_rate']:.4f}")
    print(f"\n{'folder':34s} {'n':>5s} {'succ':>6s} {'1sign':>6s} {'mindih':>7s} "
          f"{'tets':>8s} {'volerr':>9s} {'vol_ok':>7s}")
    for f in folders:
        a = result["per_folder"][f]
        if not a.get("n_ok"):
            print(f"{f[:34]:34s} {a['n']:5d} {a['success_rate']:6.3f}       -       - "
                  f"       -         -       -")
            continue
        print(f"{f[:34]:34s} {a['n']:5d} {a['success_rate']:6.3f} "
              f"{a['frac_meshes_all_one_sign']:6.3f} {a['min_dihedral_median']:7.3f} "
              f"{a['tets_median']:8.0f} {a['volume_rel_error_median']:9.2e} "
              f"{a['volume_ok_rate']:7.3f}")
    print(f"\nwall {wall:.1f}s over {workers} workers -> {logs / f'tets_{tag}.json'}")
    print("=" * 56)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
