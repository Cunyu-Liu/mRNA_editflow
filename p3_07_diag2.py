import sys, numpy as np
sys.path.insert(0, '.')
from scripts.run_p3_07 import build_ensemble_predict_fns, select_sources
from rl.p3_07_search import EnsembleDeltaOracle, score_candidate, legal_actions, exact_one_edit_optimum
from rl.p3_06_mdp import RewardV3Config, apply_edit_action

print('Building ensemble...')
predict_fns, ensemble = build_ensemble_predict_fns('data/p3/benchmark', max_proxy=10000, seed=42)
oracle = EnsembleDeltaOracle(predict_fns, max_seq_len=100)
cfg = RewardV3Config(context='protein_output_focused')

test_srcs, _, _ = select_sources('data/p3/benchmark', 24, 24, seed=0)

# Test 3 sources for quick check
improvements = []
for si in range(3):
    src = test_srcs[si]
    out = exact_one_edit_optimum(src, oracle)
    print(f'\nSource {si+1}: {src.transcript_id}')
    print(f'  source_score(0-edit)={out["source_score"]:.6f}')
    print(f'  optimum_score={out["optimum_score"]:.6f}')
    print(f'  improvement={out["improvement"]:.6f}')
    print(f'  n_edits_best={len(out["best_edits"])}')
    improvements.append(out["improvement"])

print(f'\n=== SUMMARY (3 sources) ===')
print(f'improvements: {improvements}')
print(f'n positive: {sum(1 for x in improvements if x > 0)}/3')
print(f'mean improvement: {np.mean(improvements):.6f}')
