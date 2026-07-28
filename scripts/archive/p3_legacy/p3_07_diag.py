import sys, json, numpy as np
sys.path.insert(0, '.')
from scripts.run_p3_07 import build_ensemble_predict_fns, select_sources
from rl.p3_07_search import EnsembleDeltaOracle, score_candidate, legal_actions
from rl.p3_06_mdp import RewardV3Config, apply_edit_action

print('Building ensemble...')
predict_fns, ensemble = build_ensemble_predict_fns('data/p3/benchmark', max_proxy=10000, seed=42)
oracle = EnsembleDeltaOracle(predict_fns, max_seq_len=100)
cfg = RewardV3Config(context='protein_output_focused')

test_srcs, _, _ = select_sources('data/p3/benchmark', 24, 24, seed=0)
src = test_srcs[0]
print(f'Source: {src.transcript_id}, utr={src.five_utr[:20]}...')

sc0 = score_candidate(src, src, oracle, 0, cfg)
print(f'\nSource self-score (0 edits):')
print(f'  mean_delta={sc0["mean_delta"]:.6f}, unc={sc0["uncertainty"]:.6f}, scalar={sc0["scalar"]:.6f}')

deltas = []
scores = []
for a in legal_actions(src):
    if a.is_stop():
        continue
    child = apply_edit_action(src, a)
    sc = score_candidate(src, child, oracle, 1, cfg)
    deltas.append(sc['mean_delta'])
    scores.append(sc['scalar'])

deltas = np.array(deltas)
scores = np.array(scores)
print(f'\nSingle-edit predictions ({len(deltas)} actions):')
print(f'  mean_delta: min={deltas.min():.6f}, max={deltas.max():.6f}, mean={deltas.mean():.6f}, std={deltas.std():.6f}')
print(f'  n positive deltas: {(deltas > 0).sum()}/{len(deltas)}')
print(f'  scalar: min={scores.min():.6f}, max={scores.max():.6f}')
print(f'  n positive scalar: {(scores > 0).sum()}/{len(scores)}')
print(f'  best edit: delta={deltas.max():.6f}, scalar={scores.max():.6f}')
print(f'\n  w_edit_cost={cfg.w_edit_cost}, lambda_lcb={cfg.lambda_lcb}')
print(f'  source scalar={sc0["scalar"]:.6f}, best edit scalar={scores.max():.6f}')
print(f'  improvement over source: {scores.max() - sc0["scalar"]:.6f}')

# Check position sensitivity: raw predictions for edits at different positions
print('\n=== POSITION SENSITIVITY ===')
pos_deltas = {}
for a in legal_actions(src):
    if a.is_stop():
        continue
    child = apply_edit_action(src, a)
    sc = score_candidate(src, child, oracle, 1, cfg)
    pos = a.position if hasattr(a, 'position') else -1
    pos_deltas.setdefault(pos, []).append(sc['mean_delta'])

print(f'  unique positions: {sorted(pos_deltas.keys())[:10]}...')
print(f'  delta range across positions: {min(np.mean(v) for v in pos_deltas.values()):.6f} to {max(np.mean(v) for v in pos_deltas.values()):.6f}')
print(f'  delta std across position means: {np.std([np.mean(v) for v in pos_deltas.values()]):.6f}')
