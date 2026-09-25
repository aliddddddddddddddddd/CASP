"""Supplementary post-hoc analysis: recover entities from the prose (Markdown) answers of
the B0 condition, so that B0's entity recognition can be scored separately from its format.
Heuristic; reported in the paper as a secondary analysis only."""
import json, re, sys, statistics as st
sys.path.insert(0, '.')
from casp.scoring import _norm, _prf, KEYS
HEAD = {'people': 'persons', 'persons': 'persons', 'person': 'persons',
        'organizations': 'organizations', 'organisations': 'organizations', 'organization': 'organizations',
        'places': 'locations', 'locations': 'locations', 'location': 'locations', 'place': 'locations'}
NONE = re.compile(r'^(none|no |n/a|not )', re.I)

def clean(e):
    e = re.split(r'\s+[—–-]\s+', e)[0]
    e = re.sub(r'\([^)]*\)', '', e)
    return e.strip(' *_.;:')

def parse_prose(text):
    out = {k: [] for k in KEYS}; cur = None
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r'^\*\*\s*([A-Za-z]+)[^*:]*:?\s*\*\*\s*:?\s*(.*)$', s) or re.match(r'^#+\s*([A-Za-z]+)[^:]*:?\s*(.*)$', s)
        if m and m.group(1).lower() in HEAD:
            cur = HEAD[m.group(1).lower()]; rest = m.group(2).strip()
            rest = re.sub(r'\([^)]*\)', '', rest)
            if rest and not NONE.match(rest):
                out[cur] += [clean(x) for x in rest.split(',')]
            continue
        b = re.match(r'^[-*•]\s+(.*)$', s)
        if b and cur:
            if not NONE.match(b.group(1)): out[cur].append(clean(b.group(1)))
            continue
        if s and not b: cur = cur if s.startswith('**') else None
    return out

def score(obj, gold):
    gt = {(k, _norm(e)) for k in KEYS for e in gold[k]}; gu = {_norm(e) for k in KEYS for e in gold[k]}
    pt = {(k, _norm(e)) for k in KEYS for e in obj[k] if _norm(e)}; pu = {x for _, x in pt}
    return _prf(pt, gt)[2], _prf(pu, gu)[2]

rs = [json.loads(l) for l in open('results/casp_results.jsonl')][1:]
gold = {}
for f, t in [('data/extraction.jsonl', 'T1'), ('data/extraction_injected.jsonl', 'T1i')]:
    for l in open(f):
        d = json.loads(l); gold[(t, d['id'])] = d['gold']
for task in ('T1', 'T1i'):
    b = [r for r in rs if r['task'] == task and r['condition'] == 'B0']
    sc = [score(parse_prose(r['text']), gold[(task, r['item_id'])]) for r in b]
    print(task, 'n', len(sc), 'typed', round(st.mean(s[0] for s in sc), 3), 'untyped', round(st.mean(s[1] for s in sc), 3))
