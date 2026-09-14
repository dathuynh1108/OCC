"""Publish small numeric/runtime evidence; keep large fixtures and logs in backup."""

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/reproduction-2026-09-14/evidence"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    files = [
        ("fixtures", name)
        for name in [
            "deep_parity.json",
            "drocc_parity.json",
            "patchcore_parity.json",
            "patchcore_selection_parity.json",
            "shallow_parity.json",
            "native_data_audit.json",
            "inference_replay.json",
        ]
    ] + [("runtime", "pip-versions.txt")]
    for folder, name in files:
        source = ROOT / "artifacts/reproduction" / folder / name
        shutil.copyfile(source, OUT / name)
        manifest[name] = {
            "source": source.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
