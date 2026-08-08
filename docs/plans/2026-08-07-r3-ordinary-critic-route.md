# R3 Ordinary Benchmark + Critic Route Implementation Plan

> **Execution owner:** Codex `/root` in the current authorized session.
>
> **Scope:** Build a publication-grade ordinary-only benchmark and critic comparison, then execute Route A before Route B. Draft, but do not activate, the C3 contract downgrade. Never access sealed data and never lower or substitute the `macro sign accuracy >= 0.60` gate.

**Goal:** Produce a fail-closed, hash-bound ordinary-only implementation that compares strong information-set-matched baselines, an exact potential-difference critic (A), and a sparse edit-set critic (B), with preregistered statistics, negative controls, matched budgets, and an inactive C3 redline.

**Architecture:** Add an isolated `scripts/r3x/` package rather than changing the active M4/G1 path. The package must bind route authorization and frozen input bytes before reading labels, construct leakage-safe source components, keep information sets on separate leaderboards, expose A/B through a common prediction contract, and emit machine-readable reports. Real-data training remains blocked until all authority artifacts are frozen and hash-bound.

**Technology:** Python 3, NumPy, PyTorch, scikit-learn when available, pytest, JSON/JSONL, SHA-256.

---

## Non-negotiable boundaries

- Ordinary-only development. Any path, nested field, accession, or symlink indicating sealed/restricted material must fail before input read or output creation.
- Active contract and claim matrix are immutable in this task. C3 is delivered as `PROPOSED_REDLINE_NOT_ACTIVE` with parent path and SHA-256.
- The effect gate is conjunctive: macro Spearman `>=0.25`, macro sign accuracy `>=0.60`, a preregistered stable selection metric, and a win over the strongest same-information-set executable baseline.
- The legacy signed mean-ratio enrichment is prohibited because it reverses on negative-valued endpoints. It may appear only as an explicitly invalid historical diagnostic.
- Six information sets have separate leaderboards: zero/prior, edit-only, candidate-only, paired sequence without measured anchor, measured-source-anchor-assisted, and foundation-exposure-stratified.
- No study ID in primary inference. Endpoint/context/region inputs require frozen authority; unseen levels use explicit UNK.
- Current two-study, endpoint-nested reciprocal S4 is `DEVELOPMENT_ONLY_TWO_STUDY_NESTED_ENDPOINTS`, never publication cross-study evidence.
- O0/G1/E0 remain locked. A checkpoint or dev metric cannot unlock generation, guidance, or sealed evaluation.
- Seeds are frozen as `[11, 23, 42, 67, 101]`; no best-seed reporting.

## Activation prerequisites for real ordinary training

All of the following exact bytes must be `FROZEN`, semantically valid, mutually consistent, and listed with SHA-256 in one activation manifest:

1. route authorization;
2. ready dataset and inclusion manifest;
3. context/source crosswalk and source-relation evidence;
4. endpoint estimand registry;
5. source-component split manifest;
6. metric and zero-policy registry;
7. model/optimizer/seed configuration;
8. analysis-code tree manifest verified against runtime bytes;
9. the exact method protocol;
10. the paired strongest-baseline/primary-model policy;
11. a row-level inclusion registry binding every ready row's canonical content hash.

If any item is missing, proposed, mutable, mismatched, or development-only, the runner returns `HOLD_INPUT_AUTHORITY` before labels are read and before output directories are created.

---

### Task 1: Freeze route authorization and fail-closed protocol

**Files:**
- Create: `scripts/r3x/__init__.py`
- Create: `scripts/r3x/protocol.py`
- Create: `tests/migration/test_r3x_protocol.py`
- Verify: `configs/r3x/r3_route_authorization_v1.json`

**Step 1: Write failing tests**

Cover exact schema/enum validation; inactive or null authorization; wrong sign threshold; sealed permission; missing/mismatched input hashes; proposed crosswalk/estimand; two-study nested endpoint publication attempt; path/symlink/nested-field protected-data detection; and the rule that failure occurs before output creation.

**Step 2: Run the focused tests and confirm failure**

Run: `python -m pytest -q tests/migration/test_r3x_protocol.py`

**Step 3: Implement minimal protocol primitives**

Implement canonical JSON hashing, exact-byte hash verification, route authorization parsing, ordinary path/record guards, activation-manifest validation, and structured `HOLD_*` errors. Do not create directories inside validation.

**Step 4: Re-run focused tests**

Expected: all protocol tests pass.

### Task 2: Build source-component-disjoint splits

**Files:**
- Create: `scripts/r3x/splits.py`
- Create: `tests/migration/test_r3x_splits.py`

**Step 1: Write failing split tests**

Require a global undirected relation graph over normalized source and candidate sequence identities. Add every accepted source-candidate relation edge and every frozen alias/grouping-atom edge, then use the full transitive closure as the split unit; no component may cross train/dev/test. `source_group_id_v2` is only a within-context ranking-pool key and must never be accepted as the split unit. Cover row-order invariance, the A(id X, sequence a)-B(id X, sequence b)-C(id Y, sequence b) transitive bridge, candidate-to-source chains, missing relation evidence, endpoint/study nesting labels, and deterministic seed behavior.

**Step 2: Implement deterministic component construction and assignment**

Produce a versioned split manifest with relation-graph component IDs, node/edge and membership hashes, per-cell counts, dominance, and overlap audit. Reject random row-level splitting, source-group-only splitting, unresolved relation nodes, and empty/degenerate publication cells.

**Step 3: Verify**

Run: `python -m pytest -q tests/migration/test_r3x_splits.py`.

### Task 3: Enforce information-set separation

**Files:**
- Create: `scripts/r3x/information_sets.py`
- Create: `scripts/r3x/dataset.py`
- Create: `tests/migration/test_r3x_information_sets.py`

**Step 1: Write failing tests**

Assert exact allowlists for all six information sets; measured source value is absent from direct-delta feature sets; candidate-only cannot inspect source sequence; foundation features require exposure/license qualification; study ID is excluded; and leaderboard comparisons reject unequal information sets.

**Step 2: Implement immutable row views**

Build feature-view objects from already loaded ordinary rows, with explicit field provenance and an information-set fingerprint. Keep target values outside the feature mapping.

**Step 3: Verify**

Run: `python -m pytest -q tests/migration/test_r3x_information_sets.py`.

### Task 4: Implement publication-grade metrics and clustered inference

**Files:**
- Create: `scripts/r3x/metrics.py`
- Create: `scripts/r3x/statistics.py`
- Create: `tests/migration/test_r3x_statistics.py`

**Step 1: Write failing metric tests**

Cover cell-macro sign, Spearman, RMSE, MAE, Gaussian NLL, source-component paired bootstrap, paired permutation, Holm correction, seed aggregation, all-negative outcomes, ties, zeros, single-cell HOLD, and deterministic resampling.

**Step 2: Define stable selection diagnostics**

Implement normalized selection gain `(selected - all)/(oracle - all)` only when the denominator is positive and sufficiently separated; otherwise return HOLD. Make any request for legacy signed-ratio enrichment raise a protocol error.

**Step 3: Implement cluster-aware comparisons**

Report every held-out study-estimand-publication-context-edit-count cell, seed, source component, measured ranking pool, paired method difference, and interval. Point estimates give each relation component equal mass; rank/selection metrics are computed within `source_group_id_v2` pools before cell macro aggregation. Undefined registered cells produce HOLD rather than disappearing. Seeds are optimization repeats, not biological replicates. The five fixed seed models form one prediction-level ensemble for the primary critic; single-seed results are descriptive only.

**Step 4: Verify**

Run: `python -m pytest -q tests/migration/test_r3x_statistics.py`.

### Task 5: Implement strong information-set-matched baselines

**Files:**
- Create: `scripts/r3x/features.py`
- Create: `scripts/r3x/baselines.py`
- Create: `tests/migration/test_r3x_baselines.py`

**Step 1: Write failing baseline tests**

Cover zero predictor, group-balanced train prior, edit-only Ridge/logistic diagnostic, candidate k-mer Ridge, paired local/global k-mer difference Ridge, measured-anchor-only, absolute-candidate-plus-anchor, nearest-neighbor, and deterministic fit/predict. Test that no baseline reads held-out labels or disallowed fields.

**Step 2: Implement frozen feature maps**

Use train-only standardization/vocabulary. Edit-only features use only the declared edit count, fixed-scale positions, and ref-to-alt transitions; source length or sequence-derived values are prohibited in that information set. Candidate/paired tracks may use length, GC, k-mers and local/global pair differences. All tracks receive the same allowed train-only endpoint/context/region coding. Separate direct-delta and anchor-assisted leaderboards.

**Step 3: Add optional strong executables**

Provide LightGBM/HistGradientBoosting adapters with exact availability labels. If GPU LightGBM or a foundation model is not executable under the frozen exposure/license policy, emit `NOT_RUN_WITH_REASON`; never replace it silently.

**Step 4: Verify**

Run: `python -m pytest -q tests/migration/test_r3x_baselines.py`.

### Task 6: Implement Route A exact potential-difference critic

**Files:**
- Create: `scripts/r3x/models.py`
- Create: `scripts/r3x/checkpoint.py`
- Create: `tests/migration/test_r3x_models.py`

**Step 1: Write failing algebraic-invariant tests**

Require `mu(x,x,c)=0` exactly within numerical tolerance, `mu(s,c,e)=-mu(c,s,e)`, symmetric log-variance, shared encoder weights, explicit UNK context behavior, and no study embedding. Checkpoint loading must bind architecture, schema, input hashes, split hash, information set, seed, and exact code version.

**Step 2: Implement A**

Use a shared sequence encoder and endpoint/context-conditioned scalar potential `phi`; define `mu=phi(candidate)-phi(source)`. Keep edit summary out of the mean path; it may enter a symmetric uncertainty head. Derive hard sign only from raw `mu`. Train one independent model per primary estimand so unlike raw units never share one loss; use deterministic training-only UNK masking for unseen context handling.

**Step 3: Implement parameter-matched unrestricted control**

Use the same shared encoder and allowed inputs with a concat/difference head, constrained to within 5% of A parameter count. This control distinguishes architectural constraint from regularization/capacity.

**Step 4: Verify**

Run: `python -m pytest -q tests/migration/test_r3x_models.py`.

### Task 7: Implement Route B sparse edit-set critic

**Files:**
- Modify: `scripts/r3x/models.py`
- Create: `scripts/r3x/action_support.py`
- Create: `tests/migration/test_r3x_action_support.py`

**Step 1: Write failing edit-token tests**

Require exact per-edit `(position, source_nt, target_nt, source_hidden)` tokens, order invariance, padding invariance, exact replay validation, budget `{1,3,5}`, visible edit positions, and rejection of non-SUB/unequal/wide-span inputs.

**Step 2: Implement B-additive**

Encode the source once, gather source hidden states at edit positions, encode ref-to-alt and normalized position, and sum per-edit contributions. Candidate sequence validates replay but is not a free dense branch.

**Step 3: Gate B-interaction**

Implement pairwise/set interactions only behind a separate frozen action-support policy. The policy must specify minimum independent source components per action stratum, study-context coverage, dominance, and manifest hashes. Without qualification, return `NOT_RUN_ACTION_SUPPORT` rather than training a random interaction branch.

**Step 4: Verify**

Run: `python -m pytest -q tests/migration/test_r3x_action_support.py tests/migration/test_r3x_models.py`.

### Task 8: Build deterministic training and checkpoint selection

**Files:**
- Create: `scripts/r3x/train.py`
- Create: `tests/migration/test_r3x_training.py`

**Step 1: Write failing training tests**

Cover seed determinism, source-component-balanced batches, cell-balanced loss, no test-based selection, coherent probit sign probability, no variance gaming of hard sign, rank disabled unless publication-qualified, and lexicographic checkpoint selection.

**Step 2: Implement the shared objective**

Use Huber mean loss, coherent Gaussian NLL, and sign BCE derived from `Phi(mu/sigma)` while keeping hard sign equal to `mu>=0`. Apply rank loss directly to `mu` only when a separate frozen publication rank policy passes.

**Step 3: Freeze selection rule**

Select by cell-macro dev criteria in this order: pass sign `>=0.60`, then maximize cell-macro Spearman, then minimize NLL, then deterministic epoch tie-break. Never select on ordinary test or seed winner.

**Step 4: Verify**

Run: `python -m pytest -q tests/migration/test_r3x_training.py`.

### Task 9: Add preregistered negative controls

**Files:**
- Create: `scripts/r3x/controls.py`
- Create: `tests/migration/test_r3x_controls.py`

**Step 1: Write failing control tests**

Cover source-candidate pairing shuffle within cell, endpoint/context permutation, A swap/identity checks, B source-context shuffle within action, position shuffle, target-token shuffle, B additive versus interaction, and deterministic control manifests.

**Step 2: Implement leakage-safe controls**

Permute only within frozen training partitions and preserve component boundaries. Test partitions must remain untouched until final evaluation. Report each control independently; do not combine controls into a new tuned training distribution.

**Step 3: Verify**

Run: `python -m pytest -q tests/migration/test_r3x_controls.py`.

### Task 10: Implement ordinary-only runner and report

**Files:**
- Create: `scripts/r3x/run_ordinary.py`
- Create: `scripts/r3x/report.py`
- Create: `tests/migration/test_r3x_report.py`

**Step 1: Write failing CLI/report tests**

Require preflight before any label read/output creation, exclusive output creation, canonical command/config/hash recording, per-method information set, parameter/compute accounting, every seed/cell, cluster intervals, negative controls, and explicit `NOT_RUN/HOLD/DEVELOPMENT_ONLY` states.

**Step 2: Implement dry-run and synthetic modes**

`--preflight-only` validates bytes and exits without creating paths. `--synthetic-fixture` exercises the complete A/B/baseline/report pipeline without real ordinary data. Real mode refuses to run until activation prerequisites pass.

**Step 3: Implement manuscript-safe report logic**

The report must never emit `GO_EFFECT` unless every conjunctive gate passes for one preregistered model under one information set. It must never unlock O0/G1/E0. Two-study nested-endpoint output must carry the development-only label in every headline table.

**Step 4: Verify**

Run: `python -m pytest -q tests/migration/test_r3x_report.py`.

### Task 11: Draft the inactive C3 downgrade redline

**Files:**
- Create: `docs/contracts/mrna_xeditflow_v2_1_C3_redline_PROPOSAL.md`
- Create: `docs/execution/xeditflow_claim_matrix_v2_1_PROPOSAL.yaml`
- Create: `tests/migration/test_r3x_contract_redline.py`

**Step 1: Write governance tests**

Require source-proposal path/SHA plus explicit current-active-v1.1 lineage, `PROPOSED_REDLINE_NOT_ACTIVE`, unchanged sign gate, sealed one-shot rule, no active-contract overwrite, and the three orthogonal guidance fields (`computational_exactness`, `target_epistemic_status`, `method_provenance`).

**Step 2: Draft C3 language**

Set general exact-guidance originality to rejected as a default claim; name the proposed implementation `DGM/Doob-derived learned rate guidance`. Require all three orthogonal fields on every guidance artifact: `computational_exactness` (declared-target computation only), `target_epistemic_status` (what is or is not known about the biological target), and `method_provenance` (including `DGM_DOOB_DERIVED_ADAPTATION` or `GENERATE_THEN_RERANK`).

**Step 3: Split claim gates**

Separate benchmark/critic, measured optimization, empirical guidance adaptation, and new-theory claims. New theory stays prohibited unless prior-art difference, formal theorem, independent checker, full complexity, and approximation-error propagation all pass.

**Step 4: Verify**

Run: `python -m pytest -q tests/migration/test_r3x_contract_redline.py`.

### Task 12: Integration, bundle, and real-training decision

**Files:**
- Create: `artifacts/r3x/R3X_LOCAL_VERIFICATION.json`
- Create: `docs/experiments/R3X_ORDINARY_TRAINING_RUNBOOK.md`

**Step 1: Run focused R3X suite**

Run: `python -m pytest -q tests/migration/test_r3x_*.py`.

**Step 2: Run the complete migration suite**

Run: `python -m pytest -q tests/migration`.

**Step 3: Run static and whitespace checks**

Run: `python -m compileall -q scripts/r3x` and `git diff --check` when a Git worktree is available; otherwise record `NOT_RUN_WITH_REASON` for Git-only checks.

**Step 4: Produce a verification manifest**

Record exact file hashes, test commands/results, dependency versions, excluded actions, and activation state. Preserve the golden R3-0 snapshot unchanged.

**Step 5: Decide whether real ordinary training is eligible**

If every activation prerequisite is frozen and verified, execute the fixed five-model prediction ensemble per primary estimand for scalar baselines, full-pair, the development-only R2-style no-explicit-study adaptation, A, the low-rank A diagnostic, B-additive, and conditionally B-interaction. Otherwise stop at `HOLD_INPUT_AUTHORITY` with the missing artifact list; synthetic passing is not scientific success.

**Step 6: Handoff**

Deliver the implementation/verification bundle and inactive C3 redline. Explicitly state that sealed, O0, G1, E0, remote training, contract activation, and scientific GO were not performed unless later independently authorized and evidenced.

---

## Frozen comparison matrix

| Order | Family | Information set | Required control | Publication status before evidence |
|---|---|---|---|---|
| 0 | zero / balanced prior | no anchor | none | baseline |
| 1 | edit Ridge / sign logistic | edit-only | position/action nuisance | baseline |
| 2 | candidate k-mer/tree | candidate-only | train-only vocabulary | baseline |
| 3 | paired local/global k-mer Ridge/tree/NN | paired no measured anchor | pairing shuffle | baseline |
| 4 | measured-anchor-only and candidate-absolute+anchor | anchor-assisted | separate leaderboard | baseline |
| 5 | foundation embedding linear head | exposure-stratified | license/exposure audit | conditional NOT_RUN |
| A0 | current R2 with study proxy | study-proxy information set | historical artifact only | diagnostic, never strongest-baseline gate |
| A0b | R2-style no-explicit-study adaptation | paired no measured anchor | development-only pending historical architecture crosswalk + semantic proxy audit | diagnostic before A, not strongest-baseline eligible |
| A0c | full-pair unrestricted Siamese | paired no measured anchor | matched development selection | required baseline before A |
| A1 | exact potential difference | paired no measured anchor | parameter-matched unrestricted Siamese | candidate |
| B1 | sparse edit-set additive | paired no measured anchor | edit-only and source-context shuffle | candidate |
| B2 | sparse edit-set interaction | paired no measured anchor | additive B1 + action-support gate | conditional NOT_RUN |

## Stop conditions

- Any protected/sealed path, record, or state is encountered.
- Any source component overlaps evaluation partitions.
- Endpoint estimand, context, source relation, or source value authority is incomplete or inconsistent.
- A exact identity/swap invariants fail.
- A does not beat its parameter-matched unrestricted control and strongest paired baseline consistently across eligible cells.
- B retains most performance after source-context/action/position shuffles.
- B interaction support is not publication-qualified or does not beat additive B across eligible cells.
- Wrong-pair/global/context controls preserve the claimed gain.
- A/B only improve one endpoint-nested study cell.
- Full primary sign gate remains below `0.60`.
- Metric, seed, threshold, zero handling, information set, or model family would need post-result adjustment.
- No stable preregistered selection metric exists; optimization/search claims remain HOLD.

## Execution decision

The user explicitly authorized execution in the current session on 2026-08-07. Therefore this plan proceeds in-session, with test-first checkpoints and without a further route-choice prompt. Real training is still conditional on the activation prerequisites above.
