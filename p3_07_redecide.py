#!/usr/bin/env python3
"""Re-run only the make_decision on existing p3_07_search_results.json."""
import json, sys, os
sys.path.insert(0, '.')

with open('docs/p3_07_search_results.json') as f:
    d = json.load(f)

from scripts.run_p3_07 import make_decision

# Reconstruct the inputs make_decision needs
grid_results = d['grid_results']
exact_one = d['exact_one_edit']
regret_table = d.get('regret_table', {})
dagger_out = {'training_oracle_calls': d['config'].get('dagger_training_oracle_calls', 3072)}

# make_decision expects test_srcs but only uses len(); pass a dummy list
test_srcs = [None] * d['config']['n_test_sources']

decision = make_decision(grid_results, exact_one, regret_table, dagger_out, test_srcs)

# Update the JSON with the new decision
d['rl_necessity_decision'] = decision

# Write back
with open('docs/p3_07_search_results.json', 'w') as f:
    json.dump(d, f, indent=2, default=str)

print(f"DECISION: {decision['route']}")
print(f"Rationale: {decision['rationale']}")
print(f"Degenerate: {decision['degenerate_reference']['flag']}")
print(f"reach_128: {decision['normalized_reach']['best_search_qb128']}")
print(f"reach_2048: {decision['normalized_reach']['best_search_qb2048']}")
print(f"ranker_limited_128: {decision['normalized_reach']['dagger_plus_limited_qb128']}")
if decision.get('lifecycle_cost', {}).get('breakeven_designed_cargos'):
    print(f"breakeven: {decision['lifecycle_cost']['breakeven_designed_cargos']}")
