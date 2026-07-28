import json, numpy as np

with open('docs/p3_07_search_results.json') as f:
    d = json.load(f)

# Check grid_results structure
g = d['grid_results'][0]
print('grid_result keys:', list(g.keys()))
print('sample:', {k: g[k] for k in ['method','query_budget','edit_budget','best_score','source_id']})
print()

# Source scores and improvements
ss = {k: v['source_score'] for k,v in d['exact_one_edit'].items()}
opts = {k: v['optimum_score'] for k,v in d['exact_one_edit'].items()}
imps = {k: v['improvement'] for k,v in d['exact_one_edit'].items()}

print('=== EXACT ONE-EDIT ===')
print(f'n positive improvement: {sum(1 for v in imps.values() if v > 0)}/24')
print(f'mean improvement: {np.mean(list(imps.values())):.6f}')
print()

# Show edit_budget=1 results for key methods at each query budget
print('=== edit_budget=1: normalized reach (best_score - source_score) / improvement ===')
positive_sources = {k for k,v in imps.items() if v > 1e-9}
print(f'positive sources: {len(positive_sources)}/24')
print()

methods = ['random','best_single_edit','greedy','stage_b_ranker','beam_search',
           'simulated_annealing','mcts','oracle_guided_local_search',
           'dagger_ranker','dagger_ranker_plus_limited_search']

for m in methods:
    for qb in [32, 128, 512, 2048]:
        vals = []
        for g in d['grid_results']:
            if g['method'] != m or g['query_budget'] != qb or g['edit_budget'] != 1:
                continue
            sid = g['source_id']
            if sid not in positive_sources:
                continue
            imp = imps.get(sid, 1e-6)
            sc = max(imp, 1e-6)
            reach = (g['best_score'] - ss.get(sid, 0.0)) / sc
            vals.append(reach)
        if vals:
            mean_reach = np.mean(vals)
            if qb == 128:
                print(f'  {m} @ qb{qb}: mean_reach={mean_reach:.4f} ({len(vals)} sources)')
    print()

# Also show edit_budget=3,5,10 at qb128
print('=== ALL edit budgets at qb128 (greedy) ===')
for eb in [1, 3, 5, 10]:
    vals = []
    for g in d['grid_results']:
        if g['method'] != 'greedy' or g['query_budget'] != 128 or g['edit_budget'] != eb:
            continue
        sid = g['source_id']
        if sid not in positive_sources:
            continue
        imp = imps.get(sid, 1e-6)
        sc = max(imp, 1e-6)
        reach = (g['best_score'] - ss.get(sid, 0.0)) / sc
        vals.append(reach)
    if vals:
        print(f'  eb={eb}: mean_reach={np.mean(vals):.4f}')
