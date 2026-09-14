"""Fetch exactly the reviewed author source commits, retaining their licenses."""

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
lock = json.loads((ROOT / "source_lock.json").read_text())
for name, source in lock["sources"].items():
    destination = ROOT / "vendor" / name
    env = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1")
    if not destination.exists():
        destination.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(destination)], check=True, env=env)
        subprocess.run(
            ["git", "-C", str(destination), "remote", "add", "origin", source["url"]],
            check=True,
            env=env,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(destination),
                "fetch",
                "--depth=1",
                "origin",
                source["commit"],
            ],
            check=True,
            env=env,
        )
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "--detach", source["commit"]],
            check=True,
            env=env,
        )
    actual = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != source["commit"]:
        raise RuntimeError(
            f"Existing checkout {name} has another revision; refusing to overwrite"
        )
    print(name, actual)
