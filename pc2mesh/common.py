"""Shared helpers: config, seeding, path/stem handling, mesh loading.

Everything tunable lives in config.yaml; this module only knows how to find and
read it.
"""
from __future__ import annotations

import hashlib
import os
import random
import re
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


class Cfg(dict):
    """dict with attribute access, recursively."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v


def load_config(path: str | os.PathLike | None = None) -> Cfg:
    with open(path or CONFIG_PATH) as f:
        return Cfg(yaml.safe_load(f))


def resolve(p: str | os.PathLike) -> Path:
    """Resolve a config-relative path against the repo root."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def set_seed(seed: int, torch_too: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch_too:
        try:
            import torch

            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass


# ----------------------------------------------------------------- stems / files

_WS = re.compile(r"\s+")


def norm_stem(name: str) -> str:
    """Case- and whitespace-normalized stem used to pair the two trees.

    Runs of whitespace collapse to a single space, ends are stripped, and the
    result is lowercased. Extension case is handled by the caller (Path.stem).
    """
    return _WS.sub(" ", str(name).strip()).lower()


def stable_seed(text: str, mod: int = 1 << 31) -> int:
    """A per-name seed that is the SAME in every process.

    Python salts `hash()` for str with PYTHONHASHSEED, which is read at
    interpreter start — so `os.environ["PYTHONHASHSEED"] = ...` inside a running
    process (as set_seed does) has no effect on it, and `abs(hash(stem))` gives a
    different number every run. Anything that seeds sampling off a shape's name
    has to use this instead, or the run is not reproducible and nothing says so.
    """
    return int.from_bytes(hashlib.sha1(str(text).encode()).digest()[:8], "big") % mod


def sha1_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def index_tree(root: Path, pattern: str) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    """Map normalized stem -> path for every file under `root` matching `pattern`.

    Returns (index, collisions). Collisions are stems claimed by more than one
    file; they are reported rather than silently resolved.
    """
    index: dict[str, Path] = {}
    collisions: dict[str, list[Path]] = {}
    for p in sorted(root.glob(pattern)):
        if not p.is_file():
            continue
        key = norm_stem(p.stem)
        if key in index:
            collisions.setdefault(key, [index[key]]).append(p)
        else:
            index[key] = p
    return index, collisions


def rel_folder(path: Path, root: Path) -> str:
    """Folder label for split/report grouping: the sub-directory under `root`.

    Normalized with abspath rather than resolve(): a file under `root` may be a
    symlink to a tree outside it, and resolve() would follow the link back out of
    `root` and raise. The label wanted here is where the file sits on the shelf,
    not where it points.
    """
    rel = Path(os.path.abspath(path)).relative_to(os.path.abspath(root)).parent
    return root.name if str(rel) == "." else str(rel)


# ----------------------------------------------------------------- meshes

def load_mesh(path, process: bool = True):
    """Load an STL as a single Trimesh.

    STL is an unindexed triangle soup: `process=True` merges coincident vertices,
    which is what makes watertightness meaningful at all. Every stage that reads a
    mesh uses this function so the topology is identical everywhere.
    """
    import trimesh

    m = trimesh.load(str(path), process=process, force="mesh")
    if isinstance(m, trimesh.Scene):  # defensive; force="mesh" should prevent this
        m = m.dump(concatenate=True)
    return m


def repair_for_labeling(mesh):
    """LABELING-ONLY repair. Returns (mesh, was_repaired, ok).

    Operates on a copy: the geometry used for scoring is never modified. Order is
    fill_holes -> fix_normals, per spec.
    """
    import trimesh

    if mesh.is_watertight and mesh.is_winding_consistent:
        return mesh, False, True
    m = mesh.copy()
    try:
        trimesh.repair.fill_holes(m)
    except Exception:
        pass
    try:
        trimesh.repair.fix_normals(m)
    except Exception:
        pass
    ok = bool(m.is_watertight and m.is_winding_consistent)
    return m, True, ok


def padded_bounds(bounds: np.ndarray, pad: float) -> np.ndarray:
    """Expand an axis-aligned box by `pad` (fraction of each axis extent)."""
    bounds = np.asarray(bounds, dtype=np.float64)
    ext = bounds[1] - bounds[0]
    return np.stack([bounds[0] - pad * ext, bounds[1] + pad * ext])


def load_global_bounds(cfg: Cfg) -> np.ndarray:
    """The single padded bbox shared by queries, the meshify grid and IoU grids."""
    import json

    p = resolve(cfg.paths.data) / "global_bounds.json"
    with open(p) as f:
        d = json.load(f)
    return np.array([d["min"], d["max"]], dtype=np.float64)
