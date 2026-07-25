"""pc2mesh — point cloud -> occupancy field -> marching cubes -> watertight surface.

    python main.py verify                          self-check this clone
    python main.py infer  --in <dir> --out <dir>   clouds or meshes -> surfaces
    python main.py prepare                         data/stl/ -> normalized corpus + cache
    python main.py train                           train on that cache

`train`, `infer` and `verify` forward their remaining arguments to the module that
implements them, so `python main.py infer --help` is the real option list.
`prepare` runs three stages in sequence and has its own small option set, below.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# verb -> (help line, module). One module, arguments forwarded verbatim.
PASSTHROUGH = {
    "train": ("train the occupancy net on the cache built by `prepare` "
              "(--smoke runs the 8-shape overfit gate first)", "pc2mesh.train"),
    "infer": ("a directory of .npy clouds and/or .stl meshes -> watertight surfaces "
              "(--tets also fills them with tetrahedra)", "pc2mesh.infer"),
    "verify": ("self-check: environment, gate thresholds, checkpoint, examples, "
               "and one real end-to-end decode", "pc2mesh.verify"),
}
PREPARE_HELP = ("normalize data/stl/ -> data/corpus + data/clouds, audit the pairs, "
                "then build the query cache and the 90/10 split")


def run(module: str, argv: list[str]) -> int:
    """Import `module` and call its main() with `argv` as the command line."""
    mod = importlib.import_module(module)
    saved = sys.argv
    try:
        sys.argv = [module.replace(".", "/") + ".py"] + argv
        return int(mod.main() or 0)
    finally:
        sys.argv = saved


def prepare(argv: list[str]) -> int:
    """normalize+sample -> pair audit -> query cache, stopping at the first failure.

    Chaining on past a failed stage would build a cache out of whatever happened to
    land on disk and report a shape count that means nothing, so a non-zero exit
    from any stage ends the whole verb.
    """
    ap = argparse.ArgumentParser(prog="main.py prepare", description=PREPARE_HELP)
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=0, help="only process N shapes (debug)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild shapes that are already normalized/cached")
    ap.add_argument("--workers", type=int, default=0,
                    help="normalize+sample pool size; the other two stages read "
                         "scan.workers / prep.workers from config.yaml")
    ap.add_argument("--pin-val", default=None,
                    help="JSON with a 'val' list of stems that MUST be held out")
    args = ap.parse_args(argv)

    common = []
    if args.config:
        common += ["--config", args.config]
    if args.limit:
        common += ["--limit", str(args.limit)]

    stages = [
        ("pc2mesh.prepare", common
         + (["--force"] if args.force else [])
         + (["--workers", str(args.workers)] if args.workers else [])),
        ("pc2mesh.pair_scan", common),
        ("pc2mesh.prep_queries", common
         + (["--force"] if args.force else [])
         + (["--pin-val", args.pin_val] if args.pin_val else [])),
    ]
    for i, (module, sub) in enumerate(stages, 1):
        print(f"\n########## prepare [{i}/{len(stages)}] {module} ##########")
        rc = run(module, sub)
        if rc:
            print(f"\n!! {module} exited {rc}; stopping here rather than running the "
                  f"rest of `prepare` on a half-built input.")
            return rc
    return 0


def usage() -> int:
    print(__doc__.strip())
    print("\nverbs:")
    print(f"  {'prepare':9s} {PREPARE_HELP}")
    for verb, (help_, _) in PASSTHROUGH.items():
        print(f"  {verb:9s} {help_}")
    print("\n`python main.py <verb> --help` for a verb's own options.")
    return 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        return usage()
    verb, rest = argv[0], argv[1:]
    if verb == "prepare":
        return prepare(rest)
    if verb in PASSTHROUGH:
        return run(PASSTHROUGH[verb][1], rest)
    print(f"unknown verb {verb!r}; expected one of prepare, "
          f"{', '.join(PASSTHROUGH)}\n")
    return usage() or 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
