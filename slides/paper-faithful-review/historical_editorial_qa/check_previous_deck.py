"""Check presentation order and measured-value retention (not a benchmark rerun)."""
from pathlib import Path
import json
import re
import fitz
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    tex = (ROOT / 'review.tex').read_text()
    body = tex.split('\\begin{document}', 1)[1]
    frames = re.findall(r'\\begin\{frame\}.*?\\end\{frame\}', body, re.S)
    doc = fitz.open(ROOT / 'review.pdf')
    assert len(frames) == len(doc) == 23
    baseline_text = '\n'.join(doc[i].get_text() for i in range(1, 10))
    assert 'NBD' not in baseline_text
    assert 'component scaling' not in baseline_text.lower()
    assert 'Bubble_B' not in baseline_text
    assert 'byte budget' not in doc[7].get_text().lower()
    assert 'Reserved normal' in doc[6].get_text()
    assert 'unused' in doc[6].get_text()
    assert 'NBD: local bubbles' in doc[10].get_text()
    assert 'component scaling' in doc[17].get_text().lower()
    assert '721' in doc[17].get_text() and '734' in doc[17].get_text()
    assert 'NBD versus the baseline methods' in doc[18].get_text()
    assert 'byte budget' in doc[18].get_text().lower()
    assert 'Ablation:' in doc[19].get_text()
    assert 'complete measured results' in doc[21].get_text()
    assert 'collapse and safeguards' in doc[4].get_text()

    summary = pd.read_csv(ROOT / 'results/summary.csv').set_index('method')
    baselines = ['PatchScore_same_centers', 'RBF_SVDD', 'DeepSVDD_head', 'DROCC_head']
    final_compare = [baselines[0], 'PatchScore_byte_budget', *baselines[1:], 'NBD']
    for page_index, methods in [(7, baselines), (18, final_compare), (21, list(summary.index))]:
        text = doc[page_index].get_text()
        for method in methods:
            for metric in ['auroc_mean', 'auroc_sd']:
                value = f'{100*summary.loc[method,metric]:.2f}'
                assert value in text, (page_index + 1, method, metric, value)
    threshold_text = doc[8].get_text()
    for method in baselines:
        for metric in ['fpr_mean', 'tpr_mean']:
            value = f'{100*summary.loc[method,metric]:.2f}'
            assert value in threshold_text, (method, metric, value)

    # Page-bound checks catch crop/overflow; visual review is still required.
    out_of_page = []
    for index, page in enumerate(doc):
        for block in page.get_text('dict')['blocks']:
            if block['type'] != 0:
                continue
            for line in block['lines']:
                for span in line['spans']:
                    if not span['text'].strip():
                        continue
                    x0,y0,x1,y1 = span['bbox']
                    if x0 < -.5 or y0 < -.5 or x1 > page.rect.width+.5 or y1 > page.rect.height+.5:
                        out_of_page.append({'page':index+1, 'text':span['text'], 'bbox':span['bbox']})
    assert not out_of_page, out_of_page
    result = {
        'status':'passed', 'pages':len(doc),
        'baseline_results_page':8, 'image_threshold_page':9,
        'nbd_model_starts_on_page':11, 'nbd_component_scaling_page':18,
        'nbd_comparison_page':19, 'nbd_ablation_page':20,
        'all_ten_variants_preserved_on_page':22,
        'no_nbd_in_baseline_section':True,
        'measured_values_match_source_csv':True,
        'no_text_spans_outside_page':True,
        'benchmark_rerun':False,
        'scope':'Editorial structure, displayed numeric values, PDF page bounds. Import identity checked by verify_import.py.',
    }
    (ROOT/'results/deck_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: 23 pages; baseline-only results before NBD; separate calibration roles; all displayed results unchanged.')


if __name__ == '__main__':
    main()
