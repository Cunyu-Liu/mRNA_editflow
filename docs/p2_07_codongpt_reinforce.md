# P2-07: CodonGPT REINFORCE Reproduction — Status & Gap

**Status**: GAP DOCUMENTED, full reproduction DEFERRED (low priority; substantial
implementation effort; independent of P2-02/P2-03 critical path).
**Date**: 2026-07-20

## Goal

Reproduce the CodonGPT paper's REINFORCE results: train a codon-level LM
policy with REINFORCE to optimize a codon-objective reward, using the public
pretrained `naniltx/codonGPT` checkpoint as the policy backbone.

## What exists (P1-09 baseline)

| Artifact | Path | Notes |
|----------|------|-------|
| Pretrained HF checkpoint | `external_tools/codonGPT_hf_ee7017c4/` | GPT-style causal LM; pretrained, NOT RL-trained. |
| Synonymous mask processor | `external_tools/codonGPT_hf_ee7017c4/synonymous_logit_processor.py` | `SynonymMaskingLogitsProcessor` masks non-synonymous codons per position. |
| Inference test | `scripts/test_codongpt_inference.py` | Loads checkpoint, generates synonymous variants for ACTB/HLA-A. |
| Head1024 run script | `scripts/run_external_codongpt_head1024.sh` | Runs pretrained checkpoint with synonymous-masked sampling on T4 head1024. |
| Head1024 results | `benchmark/external_sota/real_runs_t5_head1024/codonGPT/` | `cds_outputs.jsonl`, `summary.json` (2026-07-16). |
| Status JSON | `benchmark/external_sota/real_runs_t5_head1024/codonGPT.status.json` | `"paper_rl_protocol_reproduced": false`. |
| Isolated venv | `external_tools/envs/codongpt/` | Per constraint (AiZynthFinder/DFT/SOTA tools in isolated venv). |

The existing run used the **pretrained checkpoint with synonymous-masked
sampling** — this is a zero-shot / sampling baseline, NOT the paper's RL
protocol. The status JSON explicitly records `"paper_rl_protocol_reproduced":
false`.

## Gap: REINFORCE training loop

The codebase has **no REINFORCE training loop for CodonGPT**. The existing
`rl/tiny_mdp.py::REINFORCE` and the new `rl/grpo.py::GRPOREINFORCE` operate
on `TinyMDP` (a 4-char toy MDP), not on the CodonGPT LM. To reproduce the
paper's RL results, a new training loop is needed that:

1. **Wraps the CodonGPT LM as a policy** over the synonymous-codon action
   space (one action per codon position; action = synonymous codon choice).
2. **Computes log-probs** through the LM's next-token distribution with the
   `SynonymMaskingLogitsProcessor` applied.
3. **Defines a reward**: the CodonGPT paper uses CAI (Codon Adaptation
   Index) as the primary objective. We would use **CAI** for protocol
   fidelity and, as a secondary reward, **predicted TE (internal proxy)**
   from Oracle #3 — with the required qualifier.
4. **Runs REINFORCE** (or GRPO, using the new `rl/grpo.py`): sample N
   trajectories per protein, compute returns, update policy.
5. **Enforces split contract**: `--train-idx/--val-idx/--test-idx` on the
   protein-level split (per project constraint).
6. **10-seed paired significance test** vs the pretrained-only baseline.

### Minimal repro protocol (design)

```
# Pseudocode — NOT yet implemented
for protein in train_split:
    reference_codons = translate(protein.seq)
    for rollout in range(N):  # group for GRPO, or 1 for vanilla REINFORCE
        # 1. Sample synonymous variants via CodonGPT + mask processor
        variant = codongpt_sample(model, reference_codons, temperature=1.0)
        # 2. Compute reward
        cai = compute_cai(variant, host_freq_table)
        # 3. Compute log-prob (differentiable)
        logprob = codongpt_logprob(model, variant, mask_processor)
        # 4. Advantage
        advantage = (cai - baseline) / (std + eps)  # GRPO-style
        # 5. Accumulate loss
        loss -= advantage * logprob
    optimizer.step()
```

### Implementation estimate

- CodonGPT policy wrapper (~150 LOC): wraps HF model, exposes
  `sample()` and `action_logprobs()` compatible with `rl/policy.py` API.
- CAI reward (~50 LOC): standard CAI computation from a host frequency
  table (Homo sapiens, precomputed).
- Training loop (~200 LOC): REINFORCE or GRPO over protein split, with
  split-contract enforcement and 10-seed support.
- Unit tests (~200 LOC): CAI correctness, log-prob differentiability,
  mask processor behavior, split-contract enforcement.
- Total: ~600 LOC + tests.

## Decision: DEFER

P2-07 is **low priority** (per user goal priority order: P2-07 after
P2-05/06). The critical path is P2-02 (recovery run) → P2-05 (GRPO pilot,
blocked by P2-02). P2-07 is independent and can be done later without
affecting the critical path.

**Rationale**:
1. The pretrained-only CodonGPT baseline already exists (P1-09) and is
   recorded with `"paper_rl_protocol_reproduced": false`. This is honest
   and sufficient for the current paper draft.
2. Full REINFORCE repro requires ~600 LOC + tests, which is substantial
   for a low-priority task.
3. The paper's headline results do not depend on CodonGPT RL repro; they
   depend on the leakage-free headline (P2-03) and the GRPO pilot (P2-05).
4. If reviewer feedback requires CodonGPT RL repro, it can be added in
   a revision cycle.

## Constraint compliance

| Constraint | Status |
|------------|--------|
| AiZynthFinder/DFT/SOTA in isolated venv | OK — `external_tools/envs/codongpt/` is isolated. |
| 不擅自终止任何运行中进程 | OK — no processes touched. |
| 所有新代码配套单元测试 | N/A — no new code (deferred). |
| 10-seed paired significance test | N/A — deferred. |
| "improves TE" 加 predicted/internal proxy 限定词 | OK — this doc uses "predicted TE (internal proxy)" for the secondary reward. |

## Next steps (if/when prioritized)

1. Implement CodonGPT policy wrapper (HF model → `rl/policy.py`-compatible API).
2. Implement CAI reward (host frequency table from Kazusa, Homo sapiens).
3. Implement REINFORCE training loop with split contract.
4. Run 10-seed paired significance test vs pretrained-only baseline.
5. Update `benchmark/external_sota/real_runs_t5_head1024/codonGPT.status.json`
   to `"paper_rl_protocol_reproduced": true` when complete.
