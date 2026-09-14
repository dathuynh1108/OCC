"""Download the full original MVTec AD archive, check its known SHA-256."""

import argparse
import subprocess
import tarfile
from pathlib import Path

from .data import sha256_file

URL = (
    "https://www.mydrive.ch/shares/150996/b52ecdcbf521176e9db9c731f2304b27/"
    "download/420938113-1629960298/mvtec_anomaly_detection.tar.xz"
)
SHA256 = "cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data")
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "mvtec_anomaly_detection.tar.xz"
    if not archive.exists() or sha256_file(archive) != SHA256:
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "5",
                "--continue-at",
                "-",
                "--output",
                str(archive),
                URL,
            ],
            check=True,
        )
    if sha256_file(archive) != SHA256:
        raise RuntimeError("original MVTec archive checksum mismatch")
    destination = root / "mvtec_ad"
    destination.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter="data")
    print(f"Original full MVTec AD extracted into {destination}; SHA256 verified")


if __name__ == "__main__":
    main()
