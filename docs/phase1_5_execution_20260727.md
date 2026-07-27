# Phase 1--5 execution record

This package implements the five requested computational phases on top of the
P0 contract at commit `295ea86`.

## Phase 1

`data/nmi_benchmark_v2/` is a canonical JSONL store plus role indexes. The
builder preserves measured/proxy/unlabeled provenance and refuses to invent
family/context/assay holdouts. Final roles require an explicit loader flag.

## Phase 2

`models/paired_delta_former.py` receives source, candidate, explicit edits,
relative context and source value, and returns delta, variance, beneficial
probability and ranking outputs. The training entry point implements proxy
pretraining followed by measured-only fine-tuning and calibration, without
opening final-test manifests.

## Phase 3

`core/mixed_resolution_state.py` and
`models/mixed_resolution_editformer.py` implement nucleotide UTR states,
codon CDS states, atomic synonymous swaps, and a single normalized legal-action
distribution including STOP. Protein identity is checked by state transition.

## Phase 4

`models/legal_action_policy.py`, `rl/grpo_v2.py` and the search modules use the
same enumerated legal actions. The report records training, guidance and
verification query counts separately. The GRPO KL and entropy terms remain
autograd tensors.

## Phase 5

`scripts/run_phase5_experiments.py` generates trivial baselines, OOD status,
registered ablations and observational mechanism diagnostics. Empty
independent axes remain blockers; no SOTA or biological claim is promoted from
proxy-only evidence.
