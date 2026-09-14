"""Tiny deterministic source parity and interrupted-resume fixtures; not benchmarks."""
import copy
import shutil

import torch
from torch.utils.data import TensorDataset

from .common import ROOT, load_plan, rng_state, restore_rng, seed_everything, write_json, verify_source
from .deep_svdd import CachedDataset, build_network, build_autoencoder, train_phase
from optim.ae_trainer import AETrainer
from optim.deepSVDD_trainer import DeepSVDDTrainer


class FixtureDataset:
    loaders = CachedDataset.loaders

    def __init__(self, shape):
        inputs = torch.rand(6, *shape)
        self.train_set = self.test_set = TensorDataset(inputs, torch.zeros(6), torch.arange(6))


def state_errors(a, b):
    values = []
    for key in a.state_dict():
        error = (a.state_dict()[key].double() - b.state_dict()[key].double()).abs()
        values.append({'key': key, 'max': float(error.max()), 'mean': float(error.mean())})
    assert max(row['max'] for row in values) <= 1e-7, values
    return values


def main():
    verify_source('deep_svdd_torch')
    seed_everything(123)
    root = ROOT / 'artifacts/reproduction/fixtures/deep'
    # Refuse reuse of results so a fixture cannot pass from a stale checkpoint.
    root.mkdir(parents=True, exist_ok=True)
    output = []
    for dataset, shape in [('mnist', (1, 28, 28)), ('cifar10', (3, 32, 32))]:
        data = FixtureDataset(shape)
        config = copy.deepcopy(load_plan()['native']['deep_svdd'])
        config.update(batch_size=4, svdd_milestone=1, warmup_epochs=0)
        config['ae_settings'] = dict(config['ae'][dataset], milestone=1)
        for phase, objective in [('ae', 'initialization'), ('svdd', 'one-class'), ('svdd', 'soft-boundary')]:
            path = root / f'{dataset}_{phase}_{objective}'
            assert not path.exists(), f'Fixture output already exists: {path}'
            initial = (build_autoencoder if phase == 'ae' else build_network)(dataset + '_LeNet').cuda()
            reference = copy.deepcopy(initial)
            candidate = copy.deepcopy(initial)
            state = rng_state()
            if phase == 'ae':
                trainer = AETrainer(lr=config['lr'], n_epochs=1, lr_milestones=(1,),
                                    batch_size=4, weight_decay=config['ae_settings']['weight_decay'], device='cuda')
            else:
                trainer = DeepSVDDTrainer(objective, 0., None, config['nu'], lr=config['lr'], n_epochs=1,
                                         lr_milestones=(1,), batch_size=4, weight_decay=config['weight_decay'], device='cuda')
                trainer.warm_up_n_epochs = 0
            reference = trainer.train(data, reference)
            expected_rng = rng_state()
            restore_rng(state)
            candidate, center, radius, _ = train_phase(candidate, data, config, phase, objective, path, epochs_override=1)
            errors = state_errors(reference, candidate)
            gradients = []
            for a, b in zip(reference.parameters(), candidate.parameters()):
                gradients.append(float((a.grad-b.grad).abs().max()))
            assert max(gradients) <= 1e-7
            assert torch.equal(expected_rng['torch'], rng_state()['torch'])
            if phase == 'svdd':
                assert torch.equal(center, trainer.c) and torch.equal(radius, trainer.R)
            # Two epochs uninterrupted versus first epoch + reload, including scheduler/RNG.
            full, split = copy.deepcopy(initial), copy.deepcopy(initial)
            restore_rng(state)
            full, _, _, _ = train_phase(full, data, config, phase, objective, path / 'full', epochs_override=2)
            restore_rng(state)
            split, _, _, _ = train_phase(split, data, config, phase, objective, path / 'resume', epochs_override=2, stop_after=1)
            split, _, _, _ = train_phase(split, data, config, phase, objective, path / 'resume', epochs_override=2)
            resume = state_errors(full, split)
            output.append({'dataset': dataset, 'phase': phase, 'objective': objective,
                           'parameter_buffer_errors': errors, 'gradient_max_abs_error': max(gradients),
                           'resume_max_abs_error': max(r['max'] for r in resume),
                           'tolerance': 1e-7, 'upstream_same_runtime_parity': True,
                           'historical_theano_parity': 'not verified'})
    write_json(ROOT / 'artifacts/reproduction/fixtures/deep_parity.json', output)
    print('PASS: 6 source loss/gradient/state cases, 6 uninterrupted/resume cases')


if __name__ == '__main__':
    main()
