"""Verify the exact imported Git blobs and arithmetic used in the slide update.

This is NOT a training, checkpoint replay, or independent prediction-level audit.
The public repository supplies those reports; this script checks the imported
summary/category/paired CSV snapshots and the locked configuration.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '91223d63adc3289786bbdeff16d5feeb59aeb3d7'
BLOBS = {
    'results/summary.csv': '69592dd1ed2ecdfe28f6a8209a95eab6c222cb3d',
    'results/category_summary.csv': '72d09e7235b2be17950601afc24e2d2dee78e728',
    'results/paired_deltas.csv': 'cd31ee412dde6730e52c4260078e614a5e1d5169',
    'configs/full.json': '5e9f73841151dc7ec4144291b9b675fdee9aacb4',
}


def main() -> None:
    checks = {}
    for relative, expected in BLOBS.items():
        data = (ROOT / relative).read_bytes()
        git_sha = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if git_sha != expected:
            raise ValueError(f'Imported source changed: {relative}; {git_sha} != {expected}')
        checks[relative] = {'git_blob_sha1': git_sha,
                            'sha256': hashlib.sha256(data).hexdigest(),
                            'bytes': len(data), 'matches_published_git_blob': True}

    summary = pd.read_csv(ROOT / 'results/summary.csv').set_index('method')
    cats = pd.read_csv(ROOT / 'results/category_summary.csv')
    paired = pd.read_csv(ROOT / 'results/paired_deltas.csv')
    cfg = json.loads((ROOT / 'configs/full.json').read_text())
    assert len(summary) == 10 and len(cats) == 150 and len(paired) == 45
    assert cats.category.nunique() == 15 and cats.method.nunique() == 10
    assert set(cats.category) == set(cfg['categories'])
    assert sorted(paired.seed.unique().tolist()) == cfg['seeds'] == [0, 1, 2]
    assert not cats.duplicated(['category', 'method']).any()
    assert not paired.duplicated(['category', 'seed']).any()
    means = cats.groupby('method').auroc_mean.mean().reindex(summary.index)
    np.testing.assert_allclose(means, summary.auroc_mean, atol=1e-14, rtol=0)
    macro = paired.groupby('seed').mean(numeric_only=True)
    delta = {}
    for col, baseline in [('NBD_minus_same_centers', 'PatchScore_same_centers'),
                          ('NBD_minus_byte_budget', 'PatchScore_byte_budget')]:
        values = 100 * macro[col]
        np.testing.assert_allclose(values.mean(),
            100 * (summary.loc['NBD', 'auroc_mean'] - summary.loc[baseline, 'auroc_mean']),
            atol=1e-12, rtol=0)
        delta[col] = {'mean_pp': float(values.mean()), 'sample_sd_pp': float(values.std(ddof=1))}
    result = {
        'status': 'passed', 'repository': 'dathuynh1108/OCC', 'commit': COMMIT,
        'scope': 'Imported blob identities, row coverage and aggregate arithmetic only.',
        'benchmark_rerun_here': False, 'checkpoint_replay_here': False,
        'prediction_level_audit_here': False,
        'source_files': checks, 'row_counts': {'summary': 10, 'category_summary': 150, 'paired_deltas': 45},
        'category_means_match_published_macro_means': True, 'paired_deltas': delta,
        'limitations': ['AUROC/AP/SD are taken from the published benchmark.',
                       'Only paired-delta SD is recomputed here from its three seed macro means.',
                       'Per-epoch/variance/replay diagnostics are reported by the repository, not rerun here.']
    }
    (ROOT / 'results/import_verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: 4 Git blobs; 10 summary rows; 150 category rows; 45 paired rows; macro means and paired deltas.')


if __name__ == '__main__':
    main()
