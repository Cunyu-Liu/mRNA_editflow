# P3-09 External Benchmark Task Matrix

> Matched evaluation: same 24 test sources, edit_budget=1, same independent P3-02 oracle.
> All rewards are **predicted** by the independent oracle; they are not wet-lab measurements.

## Axes

### source_conditioned_minimal_edit
- **random_edit**: mean_reward=-0.0000, pos_rate=0.00%, constraint_validity=100.00%, mean_edits=0.00, wall_clock=1.92s, oracle_calls=1000
- **greedy_search**: mean_reward=-0.0000, pos_rate=0.00%, constraint_validity=100.00%, mean_edits=0.00, wall_clock=0.10s, oracle_calls=151
- **beam_search**: mean_reward=-0.0000, pos_rate=0.00%, constraint_validity=100.00%, mean_edits=0.00, wall_clock=0.10s, oracle_calls=151
- **simulated_annealing**: mean_reward=-0.0000, pos_rate=0.00%, constraint_validity=100.00%, mean_edits=0.00, wall_clock=1.92s, oracle_calls=1000
- **mcts_search**: mean_reward=-0.0000, pos_rate=0.00%, constraint_validity=100.00%, mean_edits=0.00, wall_clock=1.05s, oracle_calls=1000
- **local_search**: mean_reward=-0.0000, pos_rate=0.00%, constraint_validity=100.00%, mean_edits=0.00, wall_clock=0.02s, oracle_calls=17
- **ranker**: mean_reward=-0.0000, pos_rate=0.00%, constraint_validity=100.00%, mean_edits=0.00, wall_clock=0.05s, oracle_calls=0
- **ranker_plus_search**: mean_reward=-0.0000, pos_rate=0.00%, constraint_validity=100.00%, mean_edits=0.00, wall_clock=0.05s, oracle_calls=5
- **mef_policy**: mean_reward=-0.0495, pos_rate=100.00%, constraint_validity=100.00%, mean_edits=1.00, wall_clock=0.11s, oracle_calls=0
- **mef_policy_plus_search**: mean_reward=-0.0495, pos_rate=100.00%, constraint_validity=100.00%, mean_edits=1.00, wall_clock=0.21s, oracle_calls=0

### utr5_only
- **UTailoR**: literature-only — Adapter requires external executable / weights not present in this run.
- **UTRGAN**: literature-only — Adapter requires external executable / weights not present in this run.

### cds_protein_conditioned
- **LinearDesign**: literature-only — Adapter requires external executable / weights not present in this run.
- **EnsembleDesign**: literature-only — Adapter requires external executable / weights not present in this run.
- **codonGPT**: literature-only — Adapter requires external executable / weights not present in this run.

### de_novo_full_length
- **mRNA-GPT**: literature-only — Adapter not executable in this environment (no executable / weights / network).
- **ProMORNA**: literature-only — Adapter not executable in this environment (no executable / weights / network).
- **mRNAutilus**: literature-only — Adapter not executable in this environment (no executable / weights / network).
- **GEMORNA**: literature-only — Adapter not executable in this environment (no executable / weights / network).

## Evaluation Oracle

- Architecture: `difference`
- Description: Independent cross-fitted P3-02 oracle (difference architecture) — structurally independent from the training oracle (seq_diff + seq_linear).
- Edit budget: 1