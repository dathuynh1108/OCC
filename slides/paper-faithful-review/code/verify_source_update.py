"""Check this editorial update, not the pending training implementation."""
from pathlib import Path
import hashlib
import json
import re
import fitz

root=Path(__file__).resolve().parents[1]
doc=fitz.open(root/'review.pdf')
assert len(doc)==26, len(doc)
all_text='\n'.join(p.get_text() for p in doc)
urls=[link.get('uri','') for p in doc for link in p.get_links()]
expected=['amazon-science/patchcore-inspection','DMJTax/dd_tools',
          'lukasruff/Deep-SVDD','lukasruff/Deep-SVDD-PyTorch',
          'microsoft/EdgeML','dathuynh1108/OCC']
for path in expected:
    assert any(path in u for u in urls), path
assert 'collapse and safeguards' in all_text
assert 'Native-source reruns are pending' in re.sub(r'\s+',' ',all_text)
assert all_text.index('Baseline results') < all_text.index('NBD versus the baseline methods')
for number in ['95.86','98.56','78.79']:
    assert number in all_text, number
# At least one native link on overview/author pipeline/SVDD/collapse/DROCC/link-index pages.
for i in [2,3,4,5,6,7,25]:
    assert doc[i].get_links(), i+1
issues=[]
for i,p in enumerate(doc):
    for b in p.get_text('dict')['blocks']:
        if b['type']!=0:continue
        for line in b['lines']:
            for span in line['spans']:
                x0,y0,x1,y1=span['bbox']
                if x0 < -1 or y0 < -1 or x1 > p.rect.width+1 or y1 > p.rect.height+1:
                    issues.append({'page':i+1,'text':span['text'],'bbox':span['bbox']})
assert not issues, issues
out={'status':'passed','page_count':len(doc),'link_count':len(urls),
     'author_repositories':expected,'outside_page_text':issues,
     'benchmark_rerun':False,'validation_scope':'PDF source-link update only',
     'pending_work':'Native-source and new shared-encoder experiments in CODEX_REPRODUCE.md',
     'pdf_sha256':hashlib.sha256((root/'review.pdf').read_bytes()).hexdigest()}
(root/'results/source_update_verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
