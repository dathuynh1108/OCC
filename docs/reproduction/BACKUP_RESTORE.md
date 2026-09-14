# Backup and restore

Code, raw predictions, configuration, histories, audit evidence and final slides
are committed on `main`. Original datasets, backbone weights and fitted binary
state are excluded from Git history.

The rented GPU's final snapshot is recorded in
`results/reproduction-2026-09-14/backup_manifest.json`. Its 5,999 files are
preserved across the local checkout and the
[GitHub Release](https://github.com/dathuynh1108/OCC/releases/tag/reproduction-2026-09-14).
The release contains the 422 fitted-state files that had not completed local
transfer, grouped into ten ZIPs. It is an incremental archive, not a complete
dataset/checkpoint bundle for a fresh clone. Local source, predictions, native
datasets, the original MVTec archive and backbone weights were retained.

`backup_offbox_verified.json` records full snapshot coverage. Local files were
SHA-256 checked. Every cloud ZIP member was decoded and checked against the
snapshot before upload, and GitHub's independently reported archive digest and
size were checked after upload. `cloud_backup_manifest.json` lists the exact
members and hashes; `cloud_backup_provider_readback.json` records provider proof.

To complete the original local checkout after the network recovers, run from
the repository root (GitHub CLI required; allow roughly 10 GB free space):

```bash
mkdir -p artifacts/cloud-backup
gh release download reproduction-2026-09-14 --repo dathuynh1108/OCC \
  --pattern 'checkpoint-backup-*.zip' --dir artifacts/cloud-backup
python3 -m reproduction.restore_cloud_backup --archives-dir artifacts/cloud-backup
python3 -m reproduction.backup_manifest verify \
  --manifest results/reproduction-2026-09-14/backup_manifest.json
```

The restore command verifies ZIP and decoded member checksums before replacing
the declared files. It leaves unrelated files alone. Only the last command can
create `backup_verified.json`, which means the entire original snapshot and
MVTec archive are verified locally. This file is deliberately absent until then.
The snapshot describes training-time source; later delivery utilities and
closure documents are versioned separately in Git.

For a fresh clone, follow [RERUN.md](RERUN.md) to fetch the original datasets,
pinned author sources and backbone, then run the declared matrix. The GitHub
release also includes the final self-contained Overleaf ZIP.

MVTec-derived fitted assets retain attribution and the dataset license in each
ZIP; see [BACKUP_ASSET_LICENSES.md](BACKUP_ASSET_LICENSES.md).
