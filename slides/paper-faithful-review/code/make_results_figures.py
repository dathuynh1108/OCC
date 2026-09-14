"""Render the published benchmark; no training or score re-estimation.

Run from anywhere: python code/make_results_figures.py
All new plots read the committed CSV snapshots under results/.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets'
ASSETS.mkdir(exist_ok=True)
summary = pd.read_csv(ROOT / 'results/summary.csv').set_index('method')
category = pd.read_csv(ROOT / 'results/category_summary.csv')
paired = pd.read_csv(ROOT / 'results/paired_deltas.csv')
reported = json.loads((ROOT / 'results/reported_diagnostics.json').read_text())
methods = list(summary.index)
labels = ['PatchScore: same centers', 'PatchScore: byte budget', 'RBF SVDD',
          'Deep SVDD-head', 'DROCC-head', 'Bubble B', 'Bubble B+A',
          'Bubble B+A+D', 'Bubble B+A+F', 'NBD: B+A+D+F']

def save(fig, name):
    fig.savefig(ASSETS / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(ASSETS / (name + '.png'), dpi=230, bbox_inches='tight')
    plt.close(fig)

# Separate figure, all ten methods. Error bars are SD across seed macro means.
fig, ax = plt.subplots(figsize=(11.8, 4.85))
y = np.arange(len(methods))
x = 100 * summary.auroc_mean.to_numpy()
err = 100 * summary.auroc_sd.to_numpy()
ax.errorbar(x, y, xerr=err, fmt='o', capsize=3.5, markersize=7, linewidth=1.5)
ax.set_yticks(y, labels, fontsize=13.5)
ax.invert_yaxis()
ax.set_xlim(0, 103)
ax.set_xticks(np.arange(0,101,20))
ax.tick_params(axis='x', labelsize=12)
ax.set_xlabel('Image AUROC (%)', fontsize=13)
ax.grid(axis='x', alpha=.18)
ax.spines[['top','right']].set_visible(False)
ax.get_yticklabels()[-1].set_fontweight('bold')
ax.text(1.025, 1.035, 'Mean ± SD', transform=ax.transAxes, fontsize=13)
for k, (m, e) in enumerate(zip(x, err)):
    ax.text(1.025, k, f'{m:.2f} ± {e:.2f}', transform=ax.get_yaxis_transform(),
            va='center', fontsize=13, fontweight='bold' if k==9 else 'normal')
fig.subplots_adjust(left=.27, right=.80, top=.92, bottom=.14)
save(fig, 'mvtec_macro_auroc')

# Component ablations, kept distinct from the declared final NBD.
ablations=['Bubble_B','Bubble_BA','Bubble_BAD','Bubble_BAF','NBD']
fig, ax = plt.subplots(figsize=(6.4, 3.8))
for k, m in enumerate(ablations):
    ax.errorbar(summary.loc[m,'auroc_mean']*100,k,
                xerr=summary.loc[m,'auroc_sd']*100,fmt='o',capsize=4,markersize=7)
ax.set_yticks(range(5),['B','B+A','B+A+D','B+A+F','NBD\nB+A+D+F'],fontsize=13)
ax.invert_yaxis()
ax.set(xlim=(70,90),xticks=[70,75,80,85,90],xlabel='Image AUROC (%) — axis 70–90')
ax.tick_params(axis='x',labelsize=12)
ax.xaxis.label.set_size(12)
ax.grid(axis='x',alpha=.2)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
save(fig,'mvtec_ablations')

# The heatmap uses all categories, all variants and the entire AUROC scale.
cats=sorted(category.category.unique())
mat=category.pivot(index='method',columns='category',values='auroc_mean').loc[methods,cats].to_numpy()*100
fig, ax = plt.subplots(figsize=(12.0, 5.7))
im=ax.imshow(mat,vmin=0,vmax=100,aspect='auto',interpolation='nearest')
ax.set_xticks(range(15),[c.replace('_',' ') for c in cats],rotation=40,ha='right',fontsize=12)
ax.set_yticks(range(10),['PS: centers','PS: budget','RBF SVDD','Deep SVDD-head','DROCC-head','B','B+A','B+A+D','B+A+F','NBD'],fontsize=12)
ax.tick_params(length=0)
cb=fig.colorbar(im,ax=ax,pad=.016,fraction=.024)
cb.set_ticks([0,25,50,75,100]);cb.ax.tick_params(labelsize=11)
cb.set_label('Image AUROC (%)',fontsize=12)
# Text-free cells prevent unreadable labels on dark colors; exact values ship in CSV.
fig.tight_layout()
save(fig,'mvtec_categories')

# Paired deltas recomputed from 45 published category/seed rows, not from SD sums.
macro=paired.groupby('seed')[['NBD_minus_same_centers','NBD_minus_byte_budget']].mean()*100
fig,ax=plt.subplots(figsize=(6.4,2.8))
for k,col in enumerate(macro):
    ax.errorbar(macro[col].mean(),k,xerr=macro[col].std(ddof=1),fmt='o',capsize=4,markersize=8)
ax.axvline(0,linestyle='--',alpha=.5)
ax.set_yticks([0,1],['vs. same centers','vs. byte budget'],fontsize=12)
ax.set(xlim=(-26,3),xticks=[-25,-20,-15,-10,-5,0],xlabel='NBD minus PatchScore (AUROC pp)')
ax.tick_params(axis='x',labelsize=11)
ax.invert_yaxis();ax.grid(axis='x',alpha=.18)
ax.spines[['top','right']].set_visible(False)
fig.tight_layout()
save(fig,'mvtec_paired_deltas')

# LaTeX tables are generated directly from full-precision source values.
tex_names=['PatchScore (same centers)','PatchScore (byte budget)','RBF SVDD','Deep SVDD-head','DROCC-head',r'Bubble $B$',r'Bubble $B{+}A$',r'Bubble $B{+}A{+}D$',r'Bubble $B{+}A{+}F$',r'\textbf{NBD} $(B{+}A{+}D{+}F)$']
rows=[]
for m,label in zip(methods,tex_names):
    r=summary.loc[m]
    rows.append(f'{label} & ${r.auroc_mean*100:.2f}\\pm{r.auroc_sd*100:.2f}$ & ${r.ap_mean*100:.2f}\\pm{r.ap_sd*100:.2f}$ & {r.fpr_mean*100:.2f} & {r.tpr_mean*100:.2f} \\\\')
(ROOT/'results/full_table_rows.tex').write_text('\n'.join(rows)+'\n')
rows=[]
for m,label in zip(methods[:5]+['NBD'],tex_names[:5]+[r'\textbf{NBD}']):
    r=summary.loc[m]; rows.append(f'{label} & {r.fpr_mean*100:.2f} & {r.tpr_mean*100:.2f} \\\\')
(ROOT/'results/threshold_rows.tex').write_text('\n'.join(rows)+'\n')
print('Saved 4 result figures and 2 LaTeX tables from the published CSVs. No benchmark rerun.')
