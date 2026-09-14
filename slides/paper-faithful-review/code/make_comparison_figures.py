"""Two views of the same committed benchmark CSV; no model/metric rerun.

The early chart contains only the four already-introduced baseline methods.
The later chart adds the geometric-byte-budget control and the declared NBD.
No variant is renamed to another algorithm or selected by its performance.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets'
SUMMARY = pd.read_csv(ROOT / 'results/summary.csv').set_index('method')
BASELINES = ['PatchScore_same_centers', 'RBF_SVDD', 'DeepSVDD_head', 'DROCC_head']
COMPARISON = ['PatchScore_same_centers', 'PatchScore_byte_budget',
              'RBF_SVDD', 'DeepSVDD_head', 'DROCC_head', 'NBD']
NAMES = {
    'PatchScore_same_centers': 'PatchScore (128 centers)',
    'PatchScore_byte_budget': 'PatchScore (byte budget)',
    'RBF_SVDD': 'RBF SVDD',
    'DeepSVDD_head': 'Deep SVDD-head',
    'DROCC_head': 'DROCC-head',
    'NBD': 'NBD (B+A+D+F)',
}


def chart(methods: list[str], name: str) -> None:
    # Dot intervals do not imply a bar origin; nevertheless keep the full 0-100 scale.
    fig, ax = plt.subplots(figsize=(11.8, 3.5 if len(methods) == 4 else 4.0))
    x = SUMMARY.loc[methods, 'auroc_mean'].to_numpy() * 100
    sd = SUMMARY.loc[methods, 'auroc_sd'].to_numpy() * 100
    y = np.arange(len(methods))
    ax.errorbar(x, y, xerr=sd, fmt='o', capsize=4, markersize=8, linewidth=1.6)
    ax.set_yticks(y, [NAMES[m] for m in methods], fontsize=14)
    ax.set_ylim(len(methods) - .55, -.55)
    ax.set_xlim(0, 102)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.tick_params(axis='x', labelsize=13)
    ax.set_xlabel('Image AUROC (%)', fontsize=14)
    ax.grid(axis='x', alpha=.18)
    ax.spines[['top', 'right']].set_visible(False)
    ax.text(1.035, 1.04, 'Mean ± SD', transform=ax.transAxes, fontsize=13)
    for row, (m, mean, stdev) in enumerate(zip(methods, x, sd)):
        ax.text(1.035, row, f'{mean:.2f} ± {stdev:.2f}',
                transform=ax.get_yaxis_transform(), va='center', fontsize=14,
                fontweight='bold' if m == 'NBD' else 'normal')
        if m == 'NBD':
            ax.get_yticklabels()[row].set_fontweight('bold')
    fig.subplots_adjust(left=.28, right=.80, bottom=.19, top=.90)
    fig.savefig(ASSETS / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(ASSETS / (name + '.png'), dpi=230, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    chart(BASELINES, 'mvtec_baseline_auroc')
    chart(COMPARISON, 'mvtec_nbd_comparison')
    rows = []
    for m in BASELINES:
        r = SUMMARY.loc[m]
        rows.append(f"{NAMES[m]} & {r.fpr_mean*100:.2f} & {r.tpr_mean*100:.2f} \\\\")
    (ROOT / 'results/baseline_threshold_rows.tex').write_text('\n'.join(rows) + '\n')
    views = {
        'source': 'results/summary.csv',
        'commit': '91223d63adc3289786bbdeff16d5feeb59aeb3d7',
        'benchmark_rerun': False,
        'baseline_chart': BASELINES,
        'later_comparison_chart': COMPARISON,
        'display_names': NAMES,
        'note': 'PatchScore (128 centers) is the existing same-center CSV row, not a new run. All 10 variants remain in the appendix and category chart.',
    }
    (ROOT / 'results/figure_views.json').write_text(json.dumps(views, indent=2) + '\n')
    print('Saved baseline-only and later NBD comparisons from unchanged CSVs.')


if __name__ == '__main__':
    main()
