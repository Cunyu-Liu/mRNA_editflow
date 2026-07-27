import sys, numpy as np
sys.path.insert(0, '.')
from scripts.run_p3_07 import build_ensemble_predict_fns, select_sources
from rl.p3_07_search import EnsembleDeltaOracle, exact_one_edit_optimum

print('Building ensemble...')
predict_fns, ensemble = build_ensemble_predict_fns('data/p3/benchmark', max_proxy=10000, seed=42)
oracle = EnsembleDeltaOracle(predict_fns, max_seq_len=100)

test_srcs, _, _ = select_sources('data/p3/benchmark', 24, 24, seed=0)
improvements = []
for si, src in enumerate(test_srcs):
    out = exact_one_edit_optimum(src, oracle)
    imp = out["improvement"]
    improvements.append(imp)
    print(f'  source {si+1}/24: improvement={imp:.6f} opt={out["optimum_score"]:.6f} src={out["source_score"]:.6f}')

improvements = np.array(improvements)
print(f'\n=== SUMMARY (24 sources) ===')
print(f'n positive: {(improvements > 0).sum()}/24 ({(improvements > 0).mean():.1%})')
print(f'mean improvement: {improvements.mean():.6f}')
print(f'max improvement: {improvements.max():.6f}')
print(f'frac_positive >= 0.5: {(improvements > 0).mean() >= 0.5}')
