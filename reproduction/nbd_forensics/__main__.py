"""Command-line entrypoint for non-destructive NBD forensic stages."""

from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["all", "inventory", "tests", "scores", "geometry", "report"])
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--artifacts", type=Path, default=Path("results/mvtec-author-encoder-budgeted-v3"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    artifacts = args.artifacts if args.artifacts.is_absolute() else root / args.artifacts
    stages = ["tests", "scores", "geometry", "report"] if args.stage == "all" else [args.stage]
    run(root, artifacts.resolve(), args.output.resolve(), stages)


if __name__ == "__main__":
    main()
