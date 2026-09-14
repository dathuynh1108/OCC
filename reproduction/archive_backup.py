"""Archive selected backup files on GitHub; token is read once from stdin."""

import argparse
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import quote
import requests

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--files", type=Path, required=True)
    args = parser.parse_args()
    token = sys.stdin.readline().strip()
    assert token
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": "Bearer " + token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    response = session.get(
        f"https://api.github.com/repos/{args.repo}/releases/tags/{args.tag}", timeout=30
    )
    response.raise_for_status()
    release = response.json()
    expected = json.loads(
        (ROOT / "results/reproduction-2026-09-14/backup_manifest.json").read_text()
    )["files"]
    selected = args.files.read_text().splitlines()
    assert selected and len(selected) == len(set(selected))
    groups = [[]]
    size = 0
    for rel in selected:
        assert rel in expected and rel.startswith(
            ("results/", "artifacts/", "data/", "reproduction/")
        )
        if size + expected[rel]["bytes"] > 500 * 1024**2 and groups[-1]:
            groups.append([])
            size = 0
        groups[-1].append(rel)
        size += expected[rel]["bytes"]
    out = ROOT / "results/reproduction-2026-09-14/cloud_backup_manifest.json"
    result = {
        "repo": args.repo,
        "tag": args.tag,
        "release_url": release["html_url"],
        "complete": False,
        "archives": [],
    }
    for index, files in enumerate(groups, 1):
        name = f"checkpoint-backup-{index:02d}.zip"
        with tempfile.TemporaryDirectory(prefix="occ-backup-") as tmp:
            path = Path(tmp) / name
            with zipfile.ZipFile(
                path, "w", zipfile.ZIP_DEFLATED, compresslevel=1
            ) as archive:
                for license_name in ["MVTEC_LICENSE.txt", "BACKUP_ASSET_LICENSES.md"]:
                    archive.write(
                        ROOT / "docs/reproduction" / license_name, license_name
                    )
                for rel in files:
                    assert sha(ROOT / rel) == expected[rel]["sha256"], rel
                    archive.write(ROOT / rel, rel)
            # Check the archive's actual decoded members before uploading.
            with zipfile.ZipFile(path) as archive:
                for rel in files:
                    with archive.open(rel) as f:
                        h = hashlib.sha256()
                        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                            h.update(block)
                    assert h.hexdigest() == expected[rel]["sha256"], rel
            digest = sha(path)
            existing = next((a for a in release["assets"] if a["name"] == name), None)
            if existing is None:
                with path.open("rb") as f:
                    uploaded = session.post(
                        release["upload_url"].split("{")[0] + "?name=" + quote(name),
                        headers={"Content-Type": "application/zip"},
                        data=f,
                        timeout=(30, 600),
                    )
                uploaded.raise_for_status()
                asset = uploaded.json()
            else:
                asset = existing
            readback = session.get(asset["url"], timeout=30)
            readback.raise_for_status()
            asset = readback.json()
            assert asset["size"] == path.stat().st_size
            assert asset["digest"] == "sha256:" + digest
            result["archives"].append(
                {
                    "name": name,
                    "asset_id": asset["id"],
                    "url": asset["browser_download_url"],
                    "bytes": asset["size"],
                    "sha256": digest,
                    "github_digest_verified": True,
                    "files": {rel: expected[rel] for rel in files},
                }
            )
            out.write_text(json.dumps(result, indent=2) + "\n")
            print(
                f"Verified archive {index}/{len(groups)}: {asset['size']} bytes",
                flush=True,
            )
    result["complete"] = True
    result["file_count"] = len(selected)
    out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
