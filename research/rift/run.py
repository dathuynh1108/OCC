"""Run an image proxy-graph or supplied observed-graph experiment."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from .core import describe, NormalMemory, metrics, normal_pvalues
from .images import (ImageConfig, Sample, load_loco, audit_samples, cache_features,
                     RoleGraph, sha256_file)
from .pilot import environment, prepare_output, save_csv


def evaluate(samples, descriptors, out, pca_dim=0, roles=None, times=None):
    fit = np.array([i for i, s in enumerate(samples) if s.split == 'fit'])
    cal = np.array([i for i, s in enumerate(samples) if s.split == 'cal'])
    test = np.array([i for i, s in enumerate(samples) if s.split == 'test'])
    if min(len(fit), len(cal), len(test)) < 2:
        raise ValueError('fit, calibration and test each need at least two samples')
    if any(s.label != 0 for s in samples if s.split != 'test'):
        raise ValueError('fit and calibration must contain normal samples only')
    if 1/(len(cal)+1) > .05:
        print('WARNING: calibration too small to reject at 5%; p-value resolution is '
              f'{1/(len(cal)+1):.3f}', flush=True)
    rows, reports, audits = [], [], []
    arrays = {key: np.stack([x[key] for x in descriptors]) for key in descriptors[0]}
    np.savez_compressed(out/'descriptors.npz', **arrays, fit=fit, cal=cal, test=test,
                        ids=np.array([s.id for s in samples]))
    labels = np.array([samples[i].label for i in test])
    kinds = np.array([samples[i].kind for i in test])
    for method, matrix in arrays.items():
        model = NormalMemory(matrix[fit], pca_dim=pca_dim)
        calibration = model.score(matrix[cal])
        scores, nearest = model.score(matrix[test], return_nearest=True)
        pv = normal_pvalues(scores, calibration)
        for index, score in zip(cal, calibration):
            s = samples[index]
            rows.append({'id': s.id, 'split': 'cal', 'kind': s.kind, 'label': 0,
                         'method': method, 'score': float(score), 'normal_pvalue': '', 'nearest_fit_id': ''})
        for index, score, pvalue, reference in zip(test, scores, pv, nearest):
            s = samples[index]
            rows.append({'id': s.id, 'split': 'test', 'kind': s.kind, 'label': s.label,
                         'method': method, 'score': float(score), 'normal_pvalue': float(pvalue),
                         'nearest_fit_id': samples[fit[reference]].id})
        groups = ['all'] + sorted(set(kinds[labels == 1]))
        for group in groups:
            selected = np.ones(len(test), dtype=bool) if group == 'all' else ((labels == 0) | (kinds == group))
            reports.append({'method': method, 'group': group,
                            **metrics(labels[selected], scores[selected], calibration)})
        state = {'center': model.center, 'scale': model.scale, 'bank': model.bank, 'calibration': calibration}
        if model.projection is not None:
            state.update(pca_components=model.projection.components_, pca_mean=model.projection.mean_)
        np.savez_compressed(out/f'memory_{method}.npz', **state)
        if method == 'rift':
            standardized = model.transform(matrix[test])
            for row, index in enumerate(test):
                audit = {'id': samples[index].id, 'nearest_normal': samples[fit[nearest[row]]].id,
                         'scope': 'descriptor discrepancy, not verified pixel-level causal evidence'}
                if pca_dim or roles is None:
                    audit['top_mediator_triples'] = None
                else:
                    pairs = [(r, s) for r in range(roles) for s in range(r, roles)]
                    offset = 2*roles + len(times)*len(pairs)
                    delta = standardized[row] - model.bank[nearest[row]]
                    response = delta[offset:]
                    triples = []
                    for j in np.argsort(np.abs(response))[-3:][::-1]:
                        t_index, rest = divmod(int(j), roles*len(pairs))
                        mediator, pair = divmod(rest, len(pairs))
                        r, s = pairs[pair]
                        triples.append({'tau': float(times[t_index]), 'source_role': r,
                                        'target_role': s, 'mediator_role': mediator,
                                        'standardized_difference': float(response[j])})
                    audit['top_mediator_triples'] = triples
                audits.append(audit)
    save_csv(out/'scores.csv', rows)
    save_csv(out/'metrics.csv', reports)
    with (out/'audit.jsonl').open('w', encoding='utf-8') as f:
        for audit in audits:
            f.write(json.dumps(audit)+'\n')
    return reports


def _source_hashes():
    return {p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def run_images(root, category, out, cache, cfg, device='cuda', seed=0):
    cfg.validate()
    out = prepare_output(out)
    started = time.time()
    samples = load_loco(root, category, seed)
    audit = audit_samples(samples)
    (out/'manifest.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    paths = cache_features(samples, cache, cfg, device)
    fit_features = [np.load(paths[s.id], allow_pickle=False)['features'] for s in samples if s.split == 'fit']
    graph = RoleGraph(cfg, seed).fit(fit_features)
    graph.save(out/'role_graph.npz')
    descriptors = []
    for i, sample in enumerate(samples):
        with np.load(paths[sample.id], allow_pickle=False) as stored:
            f = stored['features']
        w, a, c = graph.transform(f, device)
        descriptor = describe(w, a, c, cfg.times, cfg.eta)
        descriptor['appearance'] = np.concatenate([f.mean(0), f.std(0)]).astype(np.float64)
        descriptors.append(descriptor)
        if i % 50 == 0:
            print(f'{category}, seed={seed}: graph {i+1}/{len(samples)}', flush=True)
    reports = evaluate(samples, descriptors, out, cfg.pca_dim, cfg.roles, cfg.times)
    info = {'scope': 'image_proxy_graph', 'dataset': 'MVTec LOCO layout (verify manifest provenance)',
            'category': category, 'seed': seed, 'device': device, 'config': asdict(cfg),
            'environment': environment(), 'seconds': time.time()-started, 'source_sha256': _source_hashes(),
            'fit_images': sum(s.split == 'fit' for s in samples),
            'calibration_images': sum(s.split == 'cal' for s in samples),
            'test_images': sum(s.split == 'test' for s in samples),
            'warnings': ['Roles are learned visual/spatial proxies, not annotated endpoints.',
                         'No test masks, test-label tuning, anomaly training, or test graph transduction.',
                         'Full-image square resize changes aspect ratio; no center crop.',
                         'Feature grid may miss thin wires or create spurious feature connectivity.',
                         'Frozen CNN extraction only; there are no neural training epochs.']}
    if cfg.backbone == 'rgb':
        info['warnings'].append('RGB debug backbone: not a pretrained-feature benchmark result.')
    (out/'run.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    return reports


def run_graphs(manifest, out, cfg, device):
    """JSONL records: id, graph (relative NPZ path), split, kind, label.

    NPZ: w [N,N], anchors [N,R], roles [N,G]. Role-column meanings must be shared
    across images. Graphs can have different N but must have fixed R and G.
    Exact graph duplicates are allowed: distinct images may induce the same graph.
    """
    out = prepare_output(out)
    manifest = Path(manifest)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    samples, descriptors, audit = [], [], []
    seen_ids = set()
    for row in rows:
        if row['id'] in seen_ids:
            raise ValueError('duplicate graph sample id')
        seen_ids.add(row['id'])
        split = {'train': 'fit', 'validation': 'cal'}.get(row['split'], row['split'])
        if split not in ('fit', 'cal', 'test') or row['label'] not in (0, 1):
            raise ValueError('invalid graph split or label')
        if split != 'test' and row['label'] != 0:
            raise ValueError('anomalous graph in fit/calibration split')
        path = (manifest.parent / row['graph']).resolve()
        with np.load(path, allow_pickle=False) as loaded:
            values = [torch.as_tensor(loaded[key], dtype=torch.float64, device=device)
                      for key in ('w', 'anchors', 'roles')]
        descriptors.append(describe(*values, cfg.times, cfg.eta))
        samples.append(Sample(row['id'], path, split, row['kind'], row['label']))
        audit.append({**row, 'sha256': sha256_file(path)})
    evaluate(samples, descriptors, out, cfg.pca_dim)
    (out/'manifest.json').write_text(json.dumps(audit, indent=2))
    (out/'run.json').write_text(json.dumps({'scope': 'supplied_graphs', 'config': asdict(cfg),
                                          'device': device, 'environment': environment(),
                                          'source_sha256': _source_hashes()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--root', type=Path, help='MVTec LOCO dataset root')
    source.add_argument('--graph-manifest', type=Path, help='JSONL of supplied observed graphs')
    parser.add_argument('--category', nargs='+', default=['splicing_connectors'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=Path('research/rift/cache'))
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable; use --device cpu explicitly')
    cfg = ImageConfig(**json.loads(args.config.read_text())) if args.config else ImageConfig()
    cfg.validate()
    if args.graph_manifest:
        run_graphs(args.graph_manifest, args.out, cfg, args.device)
        return
    aggregate = []
    for category in args.category:
        for seed in args.seeds:
            reports = run_images(args.root, category, args.out/category/f'seed_{seed}',
                                 args.cache, cfg, args.device, seed)
            aggregate.extend({'category': category, 'seed': seed, **r} for r in reports)
    save_csv(args.out/'all_metrics.csv', aggregate)
    # Report seed mean/SD within each category; no test-driven method ranking or selection.
    summary = []
    for category in args.category:
        keys = sorted({(r['method'], r['group']) for r in aggregate if r['category'] == category})
        for method, group in keys:
            values = [r['auroc'] for r in aggregate if r['category'] == category and
                      r['method'] == method and r['group'] == group and r['auroc'] is not None]
            summary.append({'category': category, 'method': method, 'group': group,
                            'seeds': len(values), 'auroc_mean': float(np.mean(values)) if values else None,
                            'auroc_sample_sd': float(np.std(values, ddof=1)) if len(values) > 1 else None})
    save_csv(args.out/'summary.csv', summary)
    print(f'Results: {args.out}/summary.csv', flush=True)


if __name__ == '__main__':
    main()
