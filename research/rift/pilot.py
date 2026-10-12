"""Graph-level mechanism probe. No rendered image or MVTec claim is made here."""
from __future__ import annotations
import argparse
import csv
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from .core import describe, NormalMemory, metrics, normal_pvalues


def make_graph(seed: int, kind: str = 'normal', noise: float = .0):
    """Two routes with neutral buffers and role-labelled intermediate guides.

    Routing: change the guide role, keeping adjacency, endpoints, counts, and
    aggregated ONE-HOP role-pair counts fixed. All-role diffusion and shortest
    paths are intentionally allowed to detect this; this is not a winning demo.
    Connection: swap which named output is connected to each input.
    Conductance nuisance is independently generated and label-independent.
    """
    if kind not in ('normal', 'routing', 'connection', 'normal_shift') or noise < 0:
        raise ValueError('invalid graph condition')
    rng = np.random.default_rng(seed)
    w = torch.zeros((18, 18), dtype=torch.float64)
    for offset in (0, 9):
        for i in range(8):
            u, v = offset + i, offset + i + 1
            weight = np.exp(rng.normal(0, noise))
            w[u, v] = w[v, u] = weight
    a = torch.eye(18, dtype=torch.float64)[:, [0, 9, 8, 17]]
    labels = torch.full((18,), 4)
    labels[[0, 9, 8, 17]] = torch.arange(4)
    labels[3:6] = 5
    labels[12:15] = 6
    if kind == 'routing':
        labels[3:6], labels[12:15] = 6, 5
    if kind == 'connection':
        left, right = w[7, 8].clone(), w[16, 17].clone()
        w[7, 8] = w[8, 7] = w[16, 17] = w[17, 16] = 0
        w[7, 17] = w[17, 7] = left
        w[16, 8] = w[8, 16] = right
    c = torch.nn.functional.one_hot(labels, 7).double()
    if kind == 'normal_shift':
        # Beyond node permutation: a held-out change in absolute conductance.
        w *= 1.7
        p = torch.as_tensor(rng.permutation(18))
        w, a, c = w[p][:, p], a[p], c[p]
    return w, a, c


def environment():
    import scipy, sklearn
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
            'cuda_available': torch.cuda.is_available()}


def prepare_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'Output is not empty: {path}. Choose a new directory.')
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(path, rows):
    if not rows:
        raise ValueError('nothing to write')
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def run_probe(out, seeds=(0, 1, 2), noises=(0., .35), train=32, cal=64, test=64):
    out = prepare_output(out)
    started = time.time()
    reports, predictions = [], []
    for seed in seeds:
        for noise in noises:
            base = 1_000_000 * (seed + 1)
            fit = [describe(*make_graph(base+i, noise=noise)) for i in range(train)]
            valid = [describe(*make_graph(base+100_000+i, noise=noise)) for i in range(cal)]
            kinds = ['normal', 'routing', 'connection', 'normal_shift']
            examples = [(kind, i, describe(*make_graph(base+200_000+i, kind, noise)))
                        for kind in kinds for i in range(test)]
            for method in fit[0]:
                bank = NormalMemory(np.stack([x[method] for x in fit]))
                cal_scores = bank.score(np.stack([x[method] for x in valid]))
                scores = bank.score(np.stack([x[2][method] for x in examples]))
                pv = normal_pvalues(scores, cal_scores)
                for (kind, i, _), score, pvalue in zip(examples, scores, pv):
                    predictions.append({'seed': seed, 'noise': noise, 'method': method,
                                        'kind': kind, 'sample': i, 'score': float(score),
                                        'normal_pvalue': float(pvalue)})
                for anomaly in ('routing', 'connection'):
                    selected = np.array([x[0] in ('normal', anomaly) for x in examples])
                    labels = np.array([x[0] == anomaly for x in examples])[selected]
                    report = metrics(labels, scores[selected], cal_scores)
                    shifted = np.array([x[0] == 'normal_shift' for x in examples])
                    report['shift_fpr_at_5pct_cal'] = float((pv[shifted] <= .05).mean())
                    reports.append({'seed': seed, 'noise': noise, 'method': method,
                                    'task': anomaly, **report})
            print(f'graph probe seed={seed}, conductance noise={noise}: complete', flush=True)
    save_csv(out/'metrics.csv', reports)
    save_csv(out/'scores.csv', predictions)
    summary = {'scope': 'synthetic graph mechanism probe, not image AD',
               'environment': environment(), 'seconds': time.time()-started,
               'config': {'seeds': list(seeds), 'noises': list(noises), 'train': train, 'cal': cal, 'test_per_kind': test},
               'claims': ['No real images evaluated.', 'No hyperparameters selected on test labels.',
                          'Shortest-path and all-role-transfer baselines have the same graph and role evidence.']}
    (out/'run.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--noises', type=float, nargs='+', default=[0., .35])
    parser.add_argument('--train', type=int, default=32)
    parser.add_argument('--cal', type=int, default=64)
    parser.add_argument('--test', type=int, default=64)
    args = parser.parse_args()
    if min(args.train, args.cal, args.test) < 2:
        parser.error('each split needs at least two samples')
    torch.set_num_threads(1)
    run_probe(args.out, args.seeds, args.noises, args.train, args.cal, args.test)


if __name__ == '__main__':
    main()
