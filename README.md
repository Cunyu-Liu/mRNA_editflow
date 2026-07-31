# mRNA-EditFlow

Source-conditioned, region-aware, grammar-constrained continuous-time Edit
Flow for 5′UTR and 3′UTR generation.

## Active scientific contract

The only active scientific and engineering contract is
`mrna_editflow_single_active_contract`:

- `docs/contracts/mrna_editflow_contract.md` — verbatim Goal document;
- `configs/utr_editflow_execution_policy.yaml` — hash-bound non-authoritative execution policy;
- `docs/utr_editflow_scientific_question.md` — frozen question and hypotheses;
- `docs/utr_editflow_claim_matrix.md` — claim/evidence boundaries;
- `docs/execution/task_registry.yaml` — phase and gate registry.

The Goal snapshot SHA256 is
`f2ae4ec8a153819f873706f652cfa5caedc6849356d68665df8154abeb40d829`.
If a lower-level file conflicts with the Goal, execution fails closed.

Previous P3/NMI and predictor-first V1 materials are retained
under archive directories as historical evidence. They are not active
constraint sources and must not be read by new training, selection, or
paper-generation code.

## Current question

Given a UTR source, region, assay/context, endpoint, target condition, edit
budget, and hard constraints, can a source-conditioned, region-aware,
grammar-constrained continuous-time Edit Flow learn transferable legal edit
trajectory distributions and generate diverse, sparse, variable-length,
controllable 5′UTR and 3′UTR candidates?

The project compares this method with matched-data and matched-budget
autoregressive generation, masked/discrete diffusion, generic Edit Flow,
direct scorers, and strong search. Candidate count, oracle queries,
wall-clock, GPU-hours, trainable parameters, action space, hard constraints,
and model-selection budget must be reported.

## Current scope

In scope:

- source-conditioned 5′UTR and 3′UTR generation;
- insertion, substitution, deletion, and STOP;
- variable-length and multi-step trajectories;
- edit budgets 1, 3, and 5, with larger budgets selected only on validation;
- region-, assay-, context-, endpoint-, and target-conditioned generation;
- foundation-model representations and priors;
- measured-support recovery and open-support legal generation;
- uncertainty, abstention, failure analysis, and matched-budget comparisons.

Out of scope for the current contract:

- CDS generation or optimization;
- protein-conditioned codon flow;
- joint full-transcript optimization;
- therapeutic or clinical efficacy claims;
- new wet-lab experiments;
- treating latent algorithmic trajectories as observed biological processes.

CDS and full-transcript work remain a future explicit decision record. Historical
implementations and results are preserved, but they are not Phase-1 tasks
under the active contract.

## Method boundary

mRNA-EditFlow is the primary method. A predictor, critic, benchmark, or
foundation model is supporting infrastructure and cannot replace the
generative edit process.

The model target is

```text
p(candidate, trajectory |
  source, region, assay/context, endpoint, target, constraints)
```

The rate field must condition on the dynamic current state and expose
non-negative rates for `INS`, `SUB`, `DEL`, and `STOP`. Legal masks apply
before normalization. A greedy reranker, post-hoc legality filter, or
candidate-only scorer is not a complete Edit Flow.

## Evidence boundary

Evidence grades are kept distinct:

```text
E0  engineering/synthetic
E1  internal computational
E2  held-out retrospective measured
E3  study- or context-disjoint measured
E4  historically exposed external
E5  genuinely untouched external
E6  prospective experiment (outside current scope)
```

GSE246381 is permanently recorded as:

```text
historically_exposed = true
role = historically_exposed_retrospective_external_stress_test
labels_allowed_for_new_training = false
labels_allowed_for_new_hyperparameter_selection = false
```

It cannot support an untouched-external claim. Open-support candidates may be
described only as predicted, computational, or proxy-supported unless they
have measured labels in a frozen evaluation track.

## Data discovery status

Phase D0 classifies every dataset by what it can test, not by size or
download convenience. Required artifacts are:

- `docs/data/hypothesis_data_requirement_matrix.md`;
- `data_registry/dataset_capability_matrix.csv`;
- `docs/data/systematic_search_protocol.md`;
- `docs/data/systematic_search_results.md`;
- `data_registry/foundation_exposure_ledger_v2.yaml`;
- `data_registry/encode_62_inventory_v2.csv`;
- `data_registry/encode_62_inventory_v2_summary.json`.

The ENCSR854RUF/ENA raw-read reconstruction is an observational/pretraining
candidate until its provenance, metadata, checksums, assay role, and overlap
are audited. Raw-download completion does not turn it into intervention
evidence.

No foundation checkpoint is selected or downloaded in C0/D0. The exposure
ledger records checkpoint, license, corpus, sequence overlap and downstream
label overlap as unknown until FM0 verifies them; unknown exposure must never
be described as unseen.

## GPU and run contract

All formal neural training is CUDA-only. A formal run must prove:

- CUDA is available;
- model parameters and real inputs are on CUDA;
- real forward, backward, and optimizer steps completed on CUDA;
- allocated GPU memory is non-zero;
- CPU fallback count is zero.

Failure produces `FAILED_WITH_EVIDENCE`; it never silently switches device.
C0 and D0 do not launch formal training.

Run infrastructure:

- `configs/execution_contract.yaml`;
- `schemas/run_manifest.schema.json`;
- `docs/execution/state_machine.md`;
- `scripts/execution/preflight.py`;
- `scripts/execution/launch_gpu_run.py`;
- `scripts/execution/monitor_run.py`.

Every paper-eligible run stores local manifests, resolved configuration,
commands, JSONL metrics/system events, checkpoints, checksums, Git state, and
failure evidence. W&B may supplement these files but cannot replace them.

## Contract checks

```bash
pytest -q tests/test_single_contract.py
python scripts/contracts/audit_single_contract.py --strict
python scripts/execution/validate_registry.py \
  --registry docs/execution/task_registry.yaml
```

These checks validate contract alignment only. Passing them does not establish
model performance, biological improvement, or paper readiness.

## Installation

```bash
python -m pip install -e .
```

The repository contains historical development code and artifacts from older
contracts. New work must start by reading the active contract, checking Git
state, processes, GPU/RAM/disk, and registering its task and dependent gate.

## Forward-only execution

Forward-only means preserving the scientific question, hard constraints,
anti-leakage rules, strong baselines, preregistered gates, failed seeds, and
negative results. Safe pause, diagnosis, repair, and a new run ID are valid
progress. Deleting evidence, lowering a gate, retuning on final labels, or
relabelling proxy scores as measurements is not.
