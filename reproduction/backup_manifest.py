"""Create/verify a portable off-box manifest before deleting the rented instance."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCOPES = [
    "results/native",
    "results/mvtec-author-encoder-budgeted-v3",
    "artifacts/reproduction/fixtures",
    "artifacts/reproduction/runtime",
    "artifacts/reproduction/smoke",
    "data/native/MNIST/raw",
    "data/native/cifar-10-batches-py",
    "data/native/cifar-10-python.tar.gz",
    "artifacts/backbone/wide_resnet50_2-95faca4d.pth",
    "run_plan.json",
    "run_plan_budgeted.json",
    "source_lock.json",
    "reproduction",
    "nbdbench",
]


def sha(path):
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), (
        f"File changed while hashing: {path}"
    )
    return {"bytes": after.st_size, "sha256": h.hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["create", "verify"])
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "create":
        files = {}
        for scope in SCOPES:
            base = ROOT / scope
            assert base.exists(), scope
            for p in sorted(base.rglob("*") if base.is_dir() else [base]):
                if (
                    not p.is_file()
                    or "__pycache__" in p.parts
                    or p.suffix in {".tmp", ".pyc"}
                ):
                    continue
                files[p.relative_to(ROOT).as_posix()] = sha(p)
        value = {
            "schema": 1,
            "files": files,
            "file_count": len(files),
            "total_bytes": sum(v["bytes"] for v in files.values()),
            "regenerable_exclusions": [
                "native preprocessed tensor caches",
                "MVTec extracted images (original archive separately verified)",
                "vendor checkouts (pinned source lock; local clones retained)",
            ],
            "mvtec_archive_sha256": "cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d",
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(value, indent=2) + "\n")
    else:
        value = json.loads(args.manifest.read_text())
        errors = []
        for rel, expected in value["files"].items():
            path = ROOT / rel
            if not path.exists() or sha(path) != expected:
                errors.append(rel)
        archive = ROOT / "data/mvtec_anomaly_detection.tar.xz"
        if (
            not archive.exists()
            or sha(archive)["sha256"] != value["mvtec_archive_sha256"]
        ):
            errors.append("original MVTec archive")
        if errors:
            raise RuntimeError(f"Backup mismatch: {errors}")
        result = {
            "verified": True,
            "file_count": value["file_count"],
            "total_bytes": value["total_bytes"],
            "manifest_sha256": sha(args.manifest)["sha256"],
            "original_MVTec_archive_verified": True,
        }
        args.manifest.with_name("backup_verified.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(json.dumps(result))


if __name__ == "__main__":
    main()
