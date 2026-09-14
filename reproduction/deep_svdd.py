"""Native author-PyTorch Deep SVDD with checkpoint and evidence instrumentation.

The source README schedule is an explicit replication target, not the different
paper schedule. Upstream modules/transforms/center/radius operations are imported.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from .common import (ROOT, environment, load_plan, restore_rng, rng_state,
                     save_checkpoint, seed_everything, sha256, verify_source, write_json)

sys.path.insert(0, str(ROOT / 'vendor/deep_svdd_torch/src'))
from datasets.cifar10 import CIFAR10_Dataset, MyCIFAR10
from datasets.mnist import MNIST_Dataset
from networks.main import build_autoencoder, build_network
from optim.ae_trainer import AETrainer
from optim.deepSVDD_trainer import DeepSVDDTrainer, get_radius

# Explicit torchvision API compatibility: values and ordering are unchanged.
MyCIFAR10.train_data = property(lambda self: self.data)
MyCIFAR10.test_data = property(lambda self: self.data)
MyCIFAR10.train_labels = property(lambda self: self.targets)
MyCIFAR10.test_labels = property(lambda self: self.targets)


class CachedDataset:
    """Materialize deterministic author transforms once, preserving loader RNG."""
    def __init__(self, dataset, normal_class, data_root, evidence_dir):
        original = (MNIST_Dataset if dataset == 'mnist' else CIFAR10_Dataset)(
            str(data_root), normal_class=normal_class)
        cache = Path(data_root) / 'author_torch_cache' / f'{dataset}_{normal_class}.pt'
        if cache.exists():
            value = torch.load(cache, weights_only=False, map_location='cpu')
        else:
            value = {}
            for split in ('train', 'test'):
                source = getattr(original, split + '_set')
                rows = [source[i] for i in range(len(source))]
                value[split] = (torch.stack([r[0] for r in rows]),
                                torch.tensor([int(r[1]) for r in rows]),
                                torch.tensor([int(r[2]) for r in rows]))
            value['original_labels'] = np.asarray(original.test_set.targets).tolist()
            save_checkpoint(cache, value)
        self.train_set = TensorDataset(*value['train'])
        self.test_set = TensorDataset(*value['test'])
        self.original_labels = value['original_labels']
        parity = {}
        for split in ('train', 'test'):
            source = getattr(original, split + '_set')
            cached = getattr(self, split + '_set')
            errors = []
            for i in (0, len(source) // 2, len(source) - 1):
                a, b = source[i], cached[i]
                assert int(a[1]) == int(b[1]) and int(a[2]) == int(b[2])
                errors.append(float((a[0] - b[0]).abs().max()))
            assert max(errors) == 0
            parity[split] = {'count': len(source), 'transform_max_abs_error': max(errors)}
        write_json(Path(evidence_dir) / f'{dataset}_class{normal_class}_data.json', {
            **parity, 'cache_sha256': sha256(cache), 'minmax_source': 'pinned_author_constants',
            'test_fitted': False, 'train_ids': value['train'][2].tolist(),
            'test_ids': value['test'][2].tolist()})

    def loaders(self, batch_size, shuffle_train=True, shuffle_test=False, num_workers=0):
        return (DataLoader(self.train_set, batch_size, shuffle=shuffle_train, num_workers=num_workers),
                DataLoader(self.test_set, batch_size, shuffle=shuffle_test, num_workers=num_workers))


def objective_loss(outputs, inputs, phase, objective, center, radius, nu):
    if phase == 'ae':
        return torch.mean(torch.sum((outputs - inputs) ** 2, dim=tuple(range(1, outputs.dim())))), None
    distances = torch.sum((outputs - center) ** 2, dim=1)
    if objective == 'soft-boundary':
        scores = distances - radius ** 2
        loss = radius ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
    else:
        loss = torch.mean(distances)
    return loss, distances


def feature_diagnostics(net, data):
    state = rng_state()
    was_training = net.training
    net.eval()
    with torch.no_grad():
        x = data.train_set.tensors[0][:200].cuda()
        z = net(x)
        value = {'sample_count': len(x), 'mean_feature_variance': float(z.var(0, unbiased=False).mean()),
                 'mean_norm': float(z.norm(dim=1).mean()), 'finite': bool(torch.isfinite(z).all())}
    net.train(was_training)
    restore_rng(state)
    return value


def train_phase(net, data, config, phase, objective, out, center=None,
                radius=None, epochs_override=None, stop_after=None):
    """Match upstream operation ordering, adding only passive diagnostics/checkpoints."""
    out.mkdir(parents=True, exist_ok=True)
    net = net.cuda()
    train_loader, _ = data.loaders(batch_size=config['batch_size'], num_workers=0)
    phase_config = config['ae_settings'] if phase == 'ae' else {
        'epochs': config['svdd_epochs'], 'milestone': config['svdd_milestone'],
        'weight_decay': config['weight_decay']}
    epochs = epochs_override or phase_config['epochs']
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'],
                                 weight_decay=phase_config['weight_decay'], amsgrad=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [phase_config['milestone']], gamma=config['lr_gamma'])
    checkpoint = out / 'latest.pt'
    history, start_epoch = [], 0
    if checkpoint.exists():
        previous = torch.load(checkpoint, weights_only=False, map_location='cuda')
        assert previous['config'] == config and previous['phase'] == phase
        assert previous['target_epochs'] == epochs and previous['objective'] == objective
        net.load_state_dict(previous['net'])
        optimizer.load_state_dict(previous['optimizer'])
        scheduler.load_state_dict(previous['scheduler'])
        center, radius = previous['center'], previous['radius']
        history, start_epoch = previous['history'], previous['epoch']
        restore_rng(previous['rng'])
    elif phase == 'svdd':
        trainer = DeepSVDDTrainer(objective, 0., None, config['nu'], device='cuda')
        center = trainer.init_center_c(train_loader, net, eps=config['center_epsilon'])
        radius = trainer.R
    net.train()
    for epoch in range(start_epoch, epochs):
        tick = time.perf_counter()
        scheduler.step()  # Author ordering retained; effective LR recorded each epoch.
        loss_sum, gradient_sum, batches, seen = 0., 0., 0, 0
        for inputs, _, _ in train_loader:
            inputs = inputs.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss, distances = objective_loss(outputs, inputs, phase, objective, center, radius, config['nu'])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss in {out}, epoch {epoch + 1}')
            loss.backward()
            gradient = torch.stack([p.grad.square().sum() for p in net.parameters() if p.grad is not None]).sum().sqrt()
            if not torch.isfinite(gradient):
                raise FloatingPointError(f'Nonfinite gradient in {out}')
            optimizer.step()
            if phase == 'svdd' and objective == 'soft-boundary' and epoch >= config['warmup_epochs']:
                radius.data = torch.tensor(get_radius(distances, config['nu']), device='cuda')
            loss_sum += loss.item()
            gradient_sum += gradient.item()
            batches += 1
            seen += len(inputs)
        torch.cuda.synchronize()
        row = {'epoch': epoch + 1, 'loss': loss_sum / batches, 'gradient_norm': gradient_sum / batches,
               'lr': optimizer.param_groups[0]['lr'], 'optimizer_steps': batches,
               'examples': seen, 'seconds': time.perf_counter() - tick,
               'radius': float(radius) if radius is not None else None}
        history.append(row)
        state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'rng': rng_state(), 'epoch': epoch + 1,
                 'target_epochs': epochs, 'phase': phase, 'objective': objective,
                 'center': center, 'radius': radius, 'config': config, 'history': history}
        save_checkpoint(checkpoint, state)
        write_json(out / 'history.json', history)
        print(json.dumps({'run': str(out.relative_to(ROOT)), 'phase': phase, **row}), flush=True)
        if stop_after is not None and epoch + 1 >= stop_after:
            break
    return net, center, radius, history


def predict(net, data, center, radius, objective):
    net.eval()
    _, loader = data.loaders(batch_size=200)
    rows = []
    with torch.no_grad():
        for inputs, targets, indices in loader:
            distances = torch.sum((net(inputs.cuda()) - center) ** 2, dim=1)
            if objective == 'soft-boundary':
                distances -= radius ** 2
            for i, y, score in zip(indices.tolist(), targets.tolist(), distances.cpu().tolist()):
                rows.append({'sample_id': i, 'original_label': data.original_labels[i],
                             'anomaly_label': y, 'score': score})
    return rows


def export_result(net, data, center, radius, objective, out, metadata):
    rows = predict(net, data, center, radius, objective)
    with (out / 'predictions.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    # Fresh model and persisted checkpoint, not the in-memory trained model.
    saved = torch.load(out / 'latest.pt', map_location='cuda', weights_only=False)
    replay = build_network(metadata['dataset'] + '_LeNet').cuda()
    replay.load_state_dict(saved['net'])
    replay_rows = predict(replay, data, saved['center'], saved['radius'], objective)
    error = float(np.max(np.abs(np.array([r['score'] for r in rows]) - np.array([r['score'] for r in replay_rows]))))
    assert error <= 1e-6
    # Reread CSV as independent metric input.
    with (out / 'predictions.csv').open() as f:
        imported = list(csv.DictReader(f))
    labels = [int(r['anomaly_label']) for r in imported]
    scores = [float(r['score']) for r in imported]
    result = {**metadata, 'objective': objective, 'selection': 'fixed_final_epoch',
              'auroc': roc_auc_score(labels, scores), 'average_precision': average_precision_score(labels, scores),
              'test_count': len(rows), 'checkpoint_sha256': sha256(out / 'latest.pt'),
              'prediction_sha256': sha256(out / 'predictions.csv'), 'checkpoint_replay_max_abs_error': error,
              'feature_after': feature_diagnostics(net, data), 'complete': True}
    write_json(out / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mnist', 'cifar10'], required=True)
    parser.add_argument('--normal-class', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/native')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'results/native/deep_svdd/DeepSVDD_author_torch_readme')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    verify_source('deep_svdd_torch')
    config = copy.deepcopy(load_plan()['native']['deep_svdd'])
    config['ae_settings'] = config['ae'][args.dataset]
    config['dataset'], config['normal_class'], config['seed'] = args.dataset, args.normal_class, args.seed
    if args.smoke:
        args.output_root = ROOT / 'artifacts/reproduction/smoke/deep_svdd'
    output = args.output_root / args.dataset / f'class_{args.normal_class}' / f'seed_{args.seed}'
    seed_everything(args.seed)
    data = CachedDataset(args.dataset, args.normal_class, args.data_root, args.output_root / 'data_evidence')
    net = build_network(args.dataset + '_LeNet')  # Author builds encoder BEFORE AE.
    ae = build_autoencoder(args.dataset + '_LeNet')
    write_json(output / 'config.json', {**config, 'smoke': args.smoke, 'runtime': environment(),
               'encoder_parameters': sum(p.numel() for p in net.parameters()),
               'ae_parameters': sum(p.numel() for p in ae.parameters())})
    ae, _, _, ae_history = train_phase(ae, data, config, 'ae', 'shared_initialization', output / 'ae',
                                       epochs_override=1 if args.smoke else None)
    # Reload persisted end-of-AE RNG even on resumed execution.
    ae_saved = torch.load(output / 'ae/latest.pt', weights_only=False, map_location='cuda')
    restore_rng(ae_saved['rng'])
    # Source pretrain() evaluates the AE once before transferring encoder weights.
    # This also consumes the test DataLoader's RNG base seed, so retain it exactly.
    AETrainer(batch_size=config['batch_size'], device='cuda').test(data, ae)
    encoder_start_rng = rng_state()
    net_dict = net.state_dict()
    net_dict.update({k: v for k, v in ae.state_dict().items() if k in net_dict})
    net.load_state_dict(net_dict)
    pretrained = copy.deepcopy(net.state_dict())
    for objective in config['objectives']:
        out = output / objective
        if (out / 'result.json').exists():
            continue
        net.load_state_dict(pretrained)
        restore_rng(encoder_start_rng)
        before = feature_diagnostics(net.cuda(), data)
        net, center, radius, history = train_phase(net, data, config, 'svdd', objective, out,
                                                  epochs_override=1 if args.smoke else None)
        result = export_result(net, data, center, radius, objective, out, {
            'target': config['target'], 'dataset': args.dataset, 'normal_class': args.normal_class,
            'seed': args.seed, 'smoke': args.smoke, 'feature_before': before,
            'ae_epochs': len(ae_history), 'svdd_epochs': len(history), 'nu': config['nu'],
            'ae_checkpoint_sha256': sha256(output / 'ae/latest.pt'),
            'run_plan_sha256': sha256(ROOT / 'run_plan.json'),
            'source_commit': verify_source('deep_svdd_torch')})
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
