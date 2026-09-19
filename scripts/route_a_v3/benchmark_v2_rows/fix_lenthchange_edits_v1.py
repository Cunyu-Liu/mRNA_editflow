#!/usr/bin/env python
"""Fix source_relative_edits for length-changing variant rows in M6/S1 eval rows.

Problem: naive enumerate(zip(src, cand)) positional diff on length-diff pairs produces
cascading shifted mismatches after the indel site (semantically misleading).
Fix: for len(source)!=len(candidate) rows, replace with a single canonical-style
LENGTH_CHANGE operation (mirrors GSE186455 canonical edit_operations semantics):
  {position: first-divergence-index, source_segment: src[p:], candidate_segment: cand[p:], type: LENGTH_CHANGE}
Equal-len rows are untouched (verified 0 mismatch). Files are this session's new
products (not frozen artifacts), so in-place rewrite is within additive-only discipline.
Also re-verifies all rows after the fix.
"""
import json

def fix(path):
    rows = [json.loads(l) for l in open(path)]
    fixed = 0
    for r in rows:
        s, c = r['source_sequence'], r['candidate_sequence']
        if len(s) == len(c):
            continue
        p = 0
        while p < min(len(s), len(c)) and s[p] == c[p]:
            p += 1
        r['source_relative_edits'] = [{
            'position': p,
            'source_segment': s[p:],
            'candidate_segment': c[p:],
            'type': 'LENGTH_CHANGE',
        }]
        fixed += 1
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    # verify
    bad = 0
    for r in rows:
        s, c = r['source_sequence'], r['candidate_sequence']
        if len(s) == len(c):
            diff = [i for i, (a, b) in enumerate(zip(s, c)) if a != b]
            claimed = {(e['position'], e['source_base'], e['candidate_base']) for e in r['source_relative_edits']}
            actual = {(i, s[i], c[i]) for i in diff}
            if claimed != actual:
                bad += 1
        else:
            e = r['source_relative_edits'][0]
            p = e['position']
            if s[p:] != e['source_segment'] or c[p:] != e['candidate_segment']:
                bad += 1
    print(path.split('/')[-2], 'fixed:', fixed, 'verify_bad:', bad, 'total:', len(rows))

fix('/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/benchmark_v2/m6_ndd5utr_eval_row/projection_rows.jsonl')
fix('/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/benchmark_v2/s1_stability_eval_row/projection_rows.jsonl')
