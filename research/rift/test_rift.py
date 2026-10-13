"""Numerical and protocol tests; no dataset or weight download."""
import numpy as np
import pytest
import torch


def core():
    from research.rift import core as module
    return module


def fixture():
    w = torch.zeros(5, 5, dtype=torch.float64)
    for u, v, x in [(0, 1, .8), (1, 2, 1.2), (2, 3, .7), (3, 4, 1.1), (1, 4, .3)]:
        w[u, v] = w[v, u] = x
    a = torch.eye(5, dtype=torch.float64)[:, [0, 4]]
    c = torch.nn.functional.one_hot(torch.tensor([0, 0, 1, 1, 1]), 2).double()
    return w, a, c


def test_core_module_exists():
    import importlib.util
    assert importlib.util.find_spec('research.rift.core') is not None


def test_finite_interventions_equal_recomputed_solve():
    m = core(); w, a, c = fixture()
    tau, eta = 2.3, .65
    transfer, delta, edges = m.intervene(w, a, tau, eta)
    for j, (u, v) in enumerate(edges.T.tolist()):
        perturbed = w.clone()
        perturbed[u, v] *= 1 - eta
        perturbed[v, u] *= 1 - eta
        actual = m.transfer(perturbed, a, tau) - transfer
        torch.testing.assert_close(delta[j], actual, atol=1e-12, rtol=1e-10)


def test_derivative_limit():
    m = core(); w, a, c = fixture()
    _, exact, _ = m.intervene(w, a, 2., 1e-5)
    _, linear, _ = m.intervene(w, a, 2., 1e-5, linear=True)
    torch.testing.assert_close(exact, linear, atol=1e-10, rtol=1e-4)


def test_node_permutation_invariance():
    m = core(); w, a, c = fixture()
    p = torch.tensor([3, 0, 4, 2, 1])
    x = m.describe(w, a, c, times=[.5, 2.])
    y = m.describe(w[p][:, p], a[p], c[p], times=[.5, 2.])
    for key in x:
        np.testing.assert_allclose(x[key], y[key], atol=1e-10, rtol=1e-9)


def test_disconnected_graph_and_missing_role_are_finite():
    m = core(); w, a, c = fixture()
    w.zero_(); a[:, 1] = 0
    for values in m.describe(w, a, c, times=[1., 10.]).values():
        assert np.isfinite(values).all()


def test_invalid_graph_rejected():
    m = core(); w, a, c = fixture()
    for bad in [w - 1., w + torch.eye(5), w.fill_(float('nan'))]:
        with pytest.raises(ValueError):
            m.describe(bad, a, c)


def test_mediator_swap_is_invisible_to_endpoint_transfer():
    from research.rift.pilot import make_graph
    m = core()
    normal = make_graph(4, 'normal', noise=0.)
    abnormal = make_graph(4, 'routing', noise=0.)
    x = m.describe(*normal); y = m.describe(*abnormal)
    np.testing.assert_allclose(x['transfer'], y['transfer'], atol=1e-12)
    np.testing.assert_allclose(x['direct'], y['direct'], atol=1e-12)
    assert np.linalg.norm(x['rift'] - y['rift']) > 1e-4
    # Strong baseline sees all roles. Do not hide its ability to catch this case.
    assert np.linalg.norm(x['allrole_transfer'] - y['allrole_transfer']) > 1e-4


def test_normal_memory_does_not_use_query_for_fit():
    m = core(); rng = np.random.default_rng(3)
    memory = m.NormalMemory(rng.normal(size=(20, 7)))
    mean = memory.center.copy()
    assert memory.score(np.ones((1, 7)) * 100)[0] > memory.score(np.zeros((1, 7)))[0]
    np.testing.assert_array_equal(memory.center, mean)


def test_no_single_class_auroc_fabricated():
    m = core()
    report = m.metrics(np.zeros(3), np.arange(3.), np.arange(5.))
    assert report['auroc'] is None and report['ap'] is None


def test_rank_calibration_is_normal_only():
    m = core()
    scores = np.array([0., 1., 9.]); calibration = np.arange(5.)
    np.testing.assert_allclose(m.normal_pvalues(scores, calibration), [1., 5/6, 1/6])


def image_fixture(root):
    from PIL import Image
    rng = np.random.default_rng(1)
    category = root / 'splicing_connectors'
    for split, kind, count in [('train', 'good', 8), ('validation', 'good', 4),
                               ('test', 'good', 3), ('test', 'logical_anomalies', 3),
                               ('test', 'structural_anomalies', 2)]:
        folder = category / split / kind; folder.mkdir(parents=True)
        for i in range(count):
            arr = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(folder / f'{i:03d}.png')
    return category


def test_loco_split_and_content_audit(tmp_path):
    from research.rift.images import load_loco, audit_samples
    image_fixture(tmp_path)
    samples = load_loco(tmp_path, 'splicing_connectors', seed=0)
    assert sum(s.split == 'fit' for s in samples) == 8
    assert sum(s.split == 'cal' for s in samples) == 4
    assert all(s.label == 0 for s in samples if s.split != 'test')
    assert len(audit_samples(samples)) == len(samples)


def test_duplicate_across_splits_rejected(tmp_path):
    from research.rift.images import load_loco, audit_samples
    import shutil
    category = image_fixture(tmp_path)
    shutil.copyfile(category/'train/good/000.png', category/'test/good/000.png')
    with pytest.raises(ValueError, match='duplicate'):
        audit_samples(load_loco(tmp_path, 'splicing_connectors', seed=0))


def test_role_fit_and_image_graph(tmp_path):
    from research.rift.images import ImageConfig, cache_features, load_loco, RoleGraph
    image_fixture(tmp_path)
    cfg = ImageConfig(backbone='rgb', image_size=32, grid_size=4, roles=3)
    samples = load_loco(tmp_path, 'splicing_connectors', seed=0)
    paths = cache_features(samples, tmp_path/'cache', cfg, device='cpu')
    fitted = [np.load(paths[s.id])['features'] for s in samples if s.split == 'fit']
    graph = RoleGraph(cfg, seed=3).fit(fitted)
    mean = graph.centers.copy()
    w, a, c = graph.transform(fitted[0], 'cpu')
    assert w.shape == (16, 16) and c.shape == (16, 3)
    core().validate(w, a, c)
    assert np.isfinite(core().describe(w, a, c)['rift']).all()
    np.testing.assert_array_equal(graph.centers, mean)
    # New image size must give a different cache identity.
    changed = ImageConfig(backbone='rgb', image_size=40, grid_size=4, roles=3)
    paths2 = cache_features(samples[:1], tmp_path/'cache', changed, device='cpu')
    assert paths2[samples[0].id] != paths[samples[0].id]


def test_cpu_resnet_forward_without_download():
    from research.rift.images import CNNFeatures
    model = CNNFeatures('resnet18', weights=None, feature_dim=16, grid_size=4)
    model.eval()
    with torch.no_grad():
        output = model(torch.rand(2, 3, 32, 32))
    assert output.shape == (2, 16, 16)
    assert torch.isfinite(output).all()


def test_roundoff_cannot_create_roleless_anomaly_score():
    from research.rift.pilot import make_graph
    m = core()
    x = m.describe(*make_graph(0, 'normal'))['rift_no_identity']
    y = m.describe(*make_graph(0, 'connection'))['rift_no_identity']
    memory = m.NormalMemory(np.stack([x] * 32))
    assert memory.score(np.stack([x, y])).tolist() == [0., 0.]


def test_image_runner_end_to_end(tmp_path):
    from research.rift.run import run_images
    from research.rift.images import ImageConfig
    image_fixture(tmp_path/'data')
    cfg = ImageConfig(backbone='rgb', image_size=32, grid_size=4, roles=3)
    run_images(tmp_path/'data', 'splicing_connectors', tmp_path/'out', tmp_path/'cache', cfg, 'cpu', seed=0)
    import json, csv
    summary = json.loads((tmp_path/'out/run.json').read_text())
    assert summary['scope'] == 'image_proxy_graph'
    assert summary['config']['backbone'] == 'rgb'
    with (tmp_path/'out/scores.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert rows and {'rift', 'allrole_transfer', 'appearance'} <= {r['method'] for r in rows}
    assert (tmp_path/'out/audit.jsonl').exists()
    with pytest.raises(FileExistsError):
        run_images(tmp_path/'data', 'splicing_connectors', tmp_path/'out', tmp_path/'cache', cfg, 'cpu', seed=0)


def test_graph_manifest_end_to_end(tmp_path):
    from research.rift.run import run_graphs
    from research.rift.images import ImageConfig
    from research.rift.pilot import make_graph
    import json
    rows = []
    for i, split in enumerate(['fit']*3 + ['cal']*3 + ['test']*4):
        kind = 'routing' if i >= 8 else 'normal'
        w, a, c = make_graph(i, kind, noise=.2)
        path = tmp_path/f'{i}.npz'
        np.savez(path, w=w.numpy(), anchors=a.numpy(), roles=c.numpy())
        rows.append({'id': str(i), 'graph': path.name, 'split': split,
                     'kind': kind, 'label': int(kind != 'normal')})
    manifest = tmp_path/'graphs.jsonl'
    manifest.write_text('\n'.join(json.dumps(r) for r in rows))
    run_graphs(manifest, tmp_path/'out', ImageConfig(), 'cpu')
    info = json.loads((tmp_path/'out/run.json').read_text())
    assert info['scope'] == 'supplied_graphs'
    assert (tmp_path/'out/metrics.csv').exists()


def test_missing_validation_uses_only_training_holdout(tmp_path):
    from research.rift.images import load_loco
    import shutil
    category = image_fixture(tmp_path)
    shutil.rmtree(category/'validation')
    samples = load_loco(tmp_path, 'splicing_connectors', seed=3)
    assert all(s.id.startswith('train/') for s in samples if s.split == 'cal')
    assert len({s.id for s in samples}) == len(samples)
    assert [s.id for s in samples if s.split == 'cal'] == [s.id for s in load_loco(tmp_path, 'splicing_connectors', seed=3) if s.split == 'cal']


def test_empty_output_guard(tmp_path):
    from research.rift.pilot import prepare_output
    out = prepare_output(tmp_path/'new'); (out/'result.txt').write_text('keep')
    with pytest.raises(FileExistsError):
        prepare_output(out)
    assert (out/'result.txt').read_text() == 'keep'


def test_scale_free_clustering_is_not_silently_assumed():
    # A global conductance change is a known failure, not a claimed invariance.
    from research.rift.pilot import make_graph
    m = core()
    x = m.describe(*make_graph(0, 'normal'))['rift']
    y = m.describe(*make_graph(0, 'normal_shift'))['rift']
    assert np.linalg.norm(x-y) > 1e-5
