"""Stage 5 — decode a latent set into a watertight surface.

Grid over the padded global bbox at R^3, encode z ONCE per shape, decode the grid
in chunks, force the outer one-voxel shell to a large negative logit so the level
set cannot run off the grid, then marching cubes at level 0.

Usage
    python pc2mesh/meshify.py --ckpt runs/<ts>/ckpt/best.pt --split val --limit 20 \
        --out runs/<ts>/meshes --drop-floaters
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pc2mesh.common import (  # noqa: E402
    load_config, load_global_bounds, padded_bounds, resolve, set_seed,
)
from pc2mesh.model import build_model  # noqa: E402


def grid_coords(bounds: np.ndarray, res: int):
    """Axis samples and the (res^3, 3) world-space grid, C-ordered as (x,y,z)."""
    lo, hi = bounds
    axes = [np.linspace(lo[d], hi[d], res, dtype=np.float64) for d in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    spacing = tuple(float(a[1] - a[0]) for a in axes)
    return g, spacing, lo


@torch.no_grad()
def logit_volume(model, pc: torch.Tensor, bounds: np.ndarray, res: int, chunk: int,
                 shell_logit: float, device="cuda", amp_dtype=torch.bfloat16) -> np.ndarray:
    """(res,res,res) float32 logit field for one shape. Encodes z exactly once."""
    grid, spacing, lo = grid_coords(bounds, res)
    gt = torch.from_numpy(grid).float().to(device)

    with torch.autocast(device, dtype=amp_dtype, enabled=device == "cuda"):
        z = model.encode(pc.unsqueeze(0) if pc.dim() == 2 else pc)

    out = torch.empty(gt.shape[0], dtype=torch.float32, device=device)
    for s in range(0, gt.shape[0], chunk):
        q = gt[s:s + chunk].unsqueeze(0)
        with torch.autocast(device, dtype=amp_dtype, enabled=device == "cuda"):
            out[s:s + chunk] = model.decode(z, q)[0].float()
    vol = out.view(res, res, res).cpu().numpy().astype(np.float32)

    # Outer one-voxel shell forced outside: the isosurface must close inside the grid.
    vol[0, :, :] = shell_logit
    vol[-1, :, :] = shell_logit
    vol[:, 0, :] = shell_logit
    vol[:, -1, :] = shell_logit
    vol[:, :, 0] = shell_logit
    vol[:, :, -1] = shell_logit
    return vol


def volume_to_mesh(vol: np.ndarray, bounds: np.ndarray, level: float = 0.0,
                   drop_floaters: bool = False, floater_frac: float = 0.001):
    """Marching cubes on the logit field -> a processed, outward-oriented Trimesh."""
    import trimesh
    from skimage import measure

    res = vol.shape[0]
    _, spacing, lo = grid_coords(bounds, res)
    if vol.max() <= level or vol.min() >= level:
        return trimesh.Trimesh()  # empty field: no level set at all

    verts, faces, _, _ = measure.marching_cubes(vol, level=level, spacing=spacing)
    verts = verts + np.asarray(lo)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)

    if drop_floaters and len(mesh.faces):
        parts = mesh.split(only_watertight=False)
        if len(parts) > 1:
            def vol_of(c):
                try:
                    v = abs(float(c.volume))
                except Exception:
                    v = 0.0
                return v if v > 0 else float(np.prod(c.extents))
            vols = np.array([vol_of(c) for c in parts])
            keep = [p for p, v in zip(parts, vols) if v >= floater_frac * vols.max()]
            if keep:
                mesh = trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]

    if len(mesh.faces):
        mesh.remove_unreferenced_vertices()
        trimesh.repair.fix_normals(mesh)
    return mesh


@torch.no_grad()
def meshify_one(model, pc_np: np.ndarray, cfg, bounds: np.ndarray, device="cuda",
                drop_floaters: bool = False, resolution: int | None = None):
    """`resolution` overrides cfg.meshify.resolution for a single call.

    Used by the resolution ablation so the committed config keeps its
    pre-registered R=128 while a run can be reproduced from the command line.

    `pc_np` is (N,3) or (N,6) = xyz ++ unit normal; the encoder slices off the
    normals when the checkpoint was trained without them.
    """
    pc = torch.from_numpy(np.asarray(pc_np, dtype=np.float32)).to(device)
    vol = logit_volume(
        model, pc, bounds,
        res=int(resolution or cfg.meshify.resolution), chunk=int(cfg.meshify.chunk),
        shell_logit=float(cfg.meshify.shell_logit), device=device,
    )
    return volume_to_mesh(vol, bounds, level=float(cfg.meshify.level),
                          drop_floaters=drop_floaters,
                          floater_frac=float(cfg.meshify.floater_volume_frac))


def load_checkpoint(ckpt_path, cfg, device="cuda"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = ck.get("model_cfg", dict(cfg.model))
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def meshify_bounds(cfg) -> np.ndarray:
    """The single grid every stage shares."""
    return padded_bounds(load_global_bounds(cfg), float(cfg.meshify.bbox_pad))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--stems", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--drop-floaters", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resolution", type=int, default=0,
                    help="override cfg.meshify.resolution for this run")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    from pc2mesh.dataset import load_split

    stems = args.stems or load_split(cfg)[args.split]
    if args.limit:
        stems = stems[: args.limit]
    out_dir = Path(args.out) if args.out else Path(args.ckpt).resolve().parent.parent / "meshes"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, _ = load_checkpoint(args.ckpt, cfg, args.device)
    bounds = meshify_bounds(cfg)
    cache = resolve(cfg.paths.cache)
    res = args.resolution or int(cfg.meshify.resolution)
    print(f"grid {res}^3 over "
          f"{np.round(bounds[0],4).tolist()} .. {np.round(bounds[1],4).tolist()}")

    from pc2mesh.dataset import load_cloud

    t0 = time.time()
    n_wt = 0
    for stem in tqdm(stems, desc="meshify"):
        pc = load_cloud(cache, stem)
        mesh = meshify_one(model, pc, cfg, bounds, args.device, args.drop_floaters,
                           resolution=res)
        n_wt += int(len(mesh.faces) > 0 and mesh.is_watertight)
        mesh.export(out_dir / f"{stem}.stl")
    dt = time.time() - t0
    print(f"wrote {len(stems)} meshes to {out_dir} in {dt:.1f}s "
          f"({dt/max(1,len(stems)):.2f}s each); watertight {n_wt}/{len(stems)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
