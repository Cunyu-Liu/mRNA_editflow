#!/usr/bin/env python3
"""Extract key stats from p3_07_search_results.json for doc generation."""
import json, numpy as np

with open('docs/p3_07_search_results.json') as f:
    d = json.load(f)

dec = d['rl_necessity_decision']
print('=== DECISION ===')
print(f"route: {dec['route']}")
print(f"rationale: {dec['rationale']}")
print(f"degenerate: {dec['degenerate_reference']['flag']}")
print(f"frac_positive: {dec['degenerate_reference']['frac_sources_positive_improvement']:.4f}")
print(f"mean_improvement: {dec['degenerate_reference']['mean_improvement']:.6f}")
print(f"reach_128: {dec['normalized_reach']['best_search_qb128']}")
print(f"reach_2048: {dec['normalized_reach']['best_search_qb2048']}")
print(f"ranker_limited_128: {dec['normalized_reach']['dagger_plus_limited_qb128']}")
print(f"breakeven: {dec['lifecycle_cost']['breakeven_designed_cargos']}")
print(f"dagger_calls: {dec['lifecycle_cost']['dagger_training_oracle_calls']}")
print()

print('=== NORM SCORE BY METHOD (qb128, edit_budget=1) ===')
ns = dec['norm_score_by_method_budget']
for m in sorted(ns.keys()):
    print(f"  {m}: qb32={ns[m].get('32',0):.4f} qb128={ns[m].get('128',0):.4f} qb512={ns[m].get('512',0):.4f} qb2048={ns[m].get('2048',0):.4f}")
print()

print('=== EXACT ONE-EDIT SUMMARY ===')
imps = [v['improvement'] for v in d['exact_one_edit'].values()]
print(f"n_sources: {len(imps)}")
print(f"n_positive: {sum(1 for x in imps if x > 0)}")
print(f"mean: {np.mean(imps):.6f}")
print(f"max: {max(imps):.6f}")
print()

print('=== EXACT TWO-EDIT ===')
for k, v in d['exact_two_edit'].items():
    print(f"  {k}: opt={v['optimum_score']:.6f} n={v['n_evaluated']}")
print()

print('=== TINY-MDP DP ===')
for k, v in d['tiny_mdp_dp'].items():
    print(f"  {k}: opt={v['optimal_value']:.6f} states={v['n_states']} edits={v.get('optimal_edits','?')}")
print()

print('=== ALGORITHM SEMANTICS ===')
for k, v in d['algorithm_semantics'].items():
    print(f"  {k}: returns={v['expected_returns']} terminal_kl={v.get('terminal_kl','?')} argmax={v.get('argmax_agreement','?')} valid={v['constraint_validity']}")
print()

print('=== REGRET STATS (vs DP) ===')
for k, v in dec.get('regret_stats_vs_dp', {}).items():
    print(f"  {k}: mean={v['mean']:.6f} max={v['max']:.6f} frac_zero={v['frac_zero']:.4f}")
print()

print('=== CONFIG ===')
print(f"  total_wall: {d['total_wall_clock_sec']:.1f}s")
print(f"  oracle: {d['config']['oracle']}")
print(f"  baselines: {len(d['config']['baselines'])}")
print(f"  n_test: {d['config']['n_test_sources']}")
