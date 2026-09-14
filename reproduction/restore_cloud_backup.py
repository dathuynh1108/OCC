"""Restore manifest-bound fitted state from previously downloaded release ZIPs."""

import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
        value.update(block)
    return value.hexdigest()


def restore(manifest, archives_dir, root):
    if not manifest["complete"]:
        raise ValueError("Cloud backup is incomplete")
    root = root.resolve()
    # Verify every downloaded archive before changing any fitted state.
    for entry in manifest["archives"]:
        path = archives_dir / entry["name"]
        with path.open("rb") as stream:
            if path.stat().st_size != entry["bytes"] or digest(stream) != entry["sha256"]:
                raise ValueError(f"Archive checksum mismatch: {path.name}")
        for rel in entry["files"]:
            target = (root / rel).resolve()
            if Path(rel).is_absolute() or not target.is_relative_to(root):
                raise ValueError(f"Unsafe archive path: {rel}")
    restored = 0
    for entry in manifest["archives"]:
        with zipfile.ZipFile(archives_dir / entry["name"]) as archive:
            for rel, expected in entry["files"].items():
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                # Only replace this declared file after its decoded bytes pass.
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as out:
                    temporary = Path(out.name)
                    try:
                        with archive.open(rel) as stream:
                            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                                out.write(block)
                        out.close()
                        with temporary.open("rb") as stream:
                            valid = (
                                temporary.stat().st_size == expected["bytes"]
                                and digest(stream) == expected["sha256"]
                            )
                        if not valid:
                            raise ValueError(f"Decoded file checksum mismatch: {rel}")
                        temporary.replace(target)
                        restored += 1
                    finally:
                        temporary.unlink(missing_ok=True)
    return restored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "results/reproduction-2026-09-14/cloud_backup_manifest.json",
    )
    args = parser.parse_args()
    count = restore(json.loads(args.manifest.read_text()), args.archives_dir, ROOT)
    print(f"Restored and verified {count} files")


if __name__ == "__main__":
    main()
