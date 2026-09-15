"""Verify every seed-0 test prediction before reporting the MVTec image comparison."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from sklearn.metrics import average_precision_score, roc_auc_score

from .common import ROOT, sha256, write_json
from .mvtec_image_baselines import CACHE, OUTPUT, SCHEDULES, output_for_schedule, protocol


def native_methods(schedule, output):
    """Map every row to the run that actually produced it.

    RBF SVDD has no epoch schedule, so a verified identical light export is
    reused in the full comparison instead of duplicating an unchanged fit.
    """
    return [
        ('RBF_SVDD_nu_0.01', OUTPUT, 'shallow/nu_0.01'),
        ('RBF_SVDD_nu_0.1', OUTPUT, 'shallow/nu_0.1'),
        ('DeepSVDD_one-class', output, 'deep/one-class'),
        ('DeepSVDD_soft-boundary', output, 'deep/soft-boundary'),
        ('DROCC', output, f"drocc/{SCHEDULES[schedule]['drocc_selection']}"),
    ]


def csv_rows(path):
    with path.open(newline='') as source:
        return list(csv.DictReader(source))


def export_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", choices=tuple(SCHEDULES), default="light")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or output_for_schedule(args.schedule)
    expected_protocol = protocol(args.schedule)
    assert json.loads((output / "protocol.json").read_text()) == expected_protocol
    manifest = json.loads((CACHE / 'manifest.json').read_text())
    rows = []
    for entry in manifest['categories']:
        category = entry['category']
        original = ROOT / 'results/native/patchcore/PatchCore_author_code_WR50_10pct' / category / 'seed-0'
        original_split = json.loads((original / 'splits.json').read_text())
        pc_predictions = csv_rows(original / 'predictions.csv')
        expected = {r['sample_id']: int(r['anomaly_label']) for r in pc_predictions}
        assert len(expected) == entry['test_count']
        assert set(expected) == set(original_split['test'])
        nbd = ROOT / 'results/lean-nbd-5070-20260915' / category / 'seed-0'
        adapted_predictions = csv_rows(nbd / 'predictions.csv')

        def measure(method, predictions, reference):
            actual = {r['sample_id']: int(r['anomaly_label']) for r in predictions}
            assert len(actual) == len(predictions) and actual == expected, (category, method)
            y = [int(r['anomaly_label']) for r in predictions]
            scores = [float(r['score']) for r in predictions]
            auroc, ap = roc_auc_score(y, scores), average_precision_score(y, scores)
            assert abs(auroc - float(reference['auroc'])) < 1e-12
            assert abs(ap - float(reference['average_precision'])) < 1e-12
            rows.append(dict(method=method, category=category, seed=0,
                             auroc=auroc, average_precision=ap, test_count=len(y)))

        for method, source_root, relative in native_methods(args.schedule, output):
            directory = source_root / category / relative
            result = json.loads((directory / 'result.json').read_text())
            assert result['complete'] and result['method'] == method
            assert sha256(directory / 'predictions.csv') == result['predictions_sha256']
            splits = json.loads((directory.parent / 'splits.json').read_text())
            assert splits == original_split
            measure(method, csv_rows(directory / 'predictions.csv'), result)
        pc_result = json.loads((original / 'result.json').read_text())
        assert sha256(original / 'predictions.csv') == pc_result['predictions_sha256']
        measure('PatchCore', pc_predictions, pc_result)
        nbd_metrics = {r['method']: r for r in csv_rows(nbd / 'metrics.csv')}
        for method in ('Bubble_A+D', 'Bubble_B+D'):
            predictions = [dict(sample_id=r['path'], anomaly_label=r['label'], score=r['score'])
                           for r in adapted_predictions if r['method'] == method]
            measure(method, predictions, nbd_metrics[method])
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['method']].append(row)
    assert len(rows) == 120 and len(grouped) == 8
    summary = []
    for method, values in grouped.items():
        assert len(values) == 15 and sum(r['test_count'] for r in values) == 1725
        summary.append(dict(method=method, seed=0, categories=15,
                            auroc=mean(r['auroc'] for r in values),
                            average_precision=mean(r['average_precision'] for r in values)))
    export_csv(output / 'comparison_per_category.csv', rows)
    export_csv(output / 'comparison_summary.csv', summary)
    write_json(output / 'comparison_verification.json', {
        'passed': True, 'methods': 8, 'categories': 15, 'seed': 0,
        'test_images_per_method': 1725, 'all_test_ids_labels_equal': True,
        'native_train_ids_equal': True,
        'schedule': args.schedule,
        'rbf_source': str(OUTPUT.relative_to(ROOT)),
        'drocc_primary_selection': SCHEDULES[args.schedule]['drocc_selection'],
        'protocol_sha256': sha256(output / 'protocol.json'),
        'summary_sha256': sha256(output / 'comparison_summary.csv'),
        'per_category_sha256': sha256(output / 'comparison_per_category.csv'),
    })
    for row in summary:
        print(row['method'], f"AUROC {100*row['auroc']:.2f}; AP {100*row['average_precision']:.2f}")


if __name__ == '__main__':
    main()
