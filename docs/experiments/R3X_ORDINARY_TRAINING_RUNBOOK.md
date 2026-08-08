# R3X Ordinary Benchmark + Critic Training Runbook

## Current state

```text
route: USER_AUTHORIZED
legacy monolithic runner: UNCONDITIONAL HOLD BEFORE READY_DATASET ACCESS
development activation trust v3: IMPLEMENTED / EXACT REPLAY / TRAINING_READY=false
trusted code + HGB execution qualification: IMPLEMENTED / SYNTHETIC EXECUTABILITY ONLY
OS capability separation: REQUIRED / NOT YET ATTESTED
typed semantic pre-test freeze: IMPLEMENTED / STRICT DEVELOPMENT REPLAY / CAPABILITY_READY=false
pre-test control registry: IMPLEMENTED / BROKEN-PAIR INACTIVE / CAN_SATISFY_FREEZE=false
test-feature stage manifest: IMPLEMENTED / PREPARATION ONLY / PRODUCTION RUNTIME DISABLED
ordinary outcome stage manifest: IMPLEMENTED / PREPARATION ONLY / PRODUCTION RUNTIME DISABLED
outcome aggregate evaluator: REQUIRED / NOT CONNECTED
activation v1 / monolithic ready dataset: OUTCOME-BEARING DEVELOPMENT AUDIT ONLY; NOT BLIND PRE-TEST
real ordinary training: NOT_RUN / HOLD_CAPABILITY_AND_SEMANTIC_FREEZE_INCOMPLETE
development-only strongest-baseline selection: STRICT SCHEMA IMPLEMENTED / NO REAL EVIDENCE MATERIALIZED
predictor-provenance-bound selector schema: IMPLEMENTED / NOT YET FROZEN IN A REAL RUN
preregistered negative controls: REQUIRED FOR GO / CURRENT RUNNER EMITS EXPLICIT NOT_RUN HOLD
commit-last run closure: IMPLEMENTED / SCIENTIFIC HOLD UNTIL ALL EVIDENCE REPLAYS
sealed_access: false
sealed: NOT_ACCESSED / PROHIBITED
O0/G1/E0: LOCKED
sign gate: unchanged at macro >= 0.60
C3 downgrade redline: DRAFT ONLY / INACTIVE
C3 exact-guidance claim: PROPOSED DGM/Doob-derived adaptation wording only
active contract mutation: NONE
```

This runbook is operational guidance, not scientific GO evidence. The local synthetic fixture and tests establish only engineering behavior. No real ordinary benchmark or A→B training has been executed under a publication-qualified capability-separated protocol. The current guard intentionally stops before the outcome-bearing v1 activation validator, the monolithic `ready_dataset` row loader, output-directory creation, GPU allocation, or any internal-test artifact. A manifest cannot self-declare its way past this guard.

The v3 development activation artifact now strips the old full activation, firewall path, future-stage paths, test data hashes, projection hashes, row counts and support from the development runtime view. Seven typed authorities and three rich development projections are exact-replayed. The trusted-execution kernel independently replays approved code bytes, Python/package fingerprints and a deterministic CPU HGB smoke. These two properties are necessary but not sufficient: current artifacts deliberately retain `training_ready=false` because the development process has not yet proved, at the operating-system or externally held decryption-key level, that it lacks test-feature, test-outcome and sealed capabilities.

Feature- and outcome-stage manifests are also present, with strict cross-stage field firewalls and fixed sign/sealed boundaries. Their public runtime functions are unconditionally disabled: the feature stage lacks a connected, capability-qualified typed semantic freeze and OS mount verifier; the outcome stage additionally lacks a descriptor-pinned claim namespace and aggregate-only evaluator. A hash-only staged commit cannot unlock either stage. The old raw disclosure APIs therefore remain non-authoritative even when their synthetic fixtures pass.

The typed pre-test freeze now strictly replays the exact eight-baseline policy, development selector, predictor provenance, checkpoint-selection envelopes and underlying development checkpoint states. Its direct control verifier is deliberately marked `SYNTHETIC_SCAFFOLD_ONLY_NOT_PRETEST_CONTROL_QUALIFICATION`. The separate pre-test control registry remains `can_satisfy_pretest_freeze=false` because the broken-pair control is inactive and the parent namespace is not descriptor-pinned. Therefore `training_ready=false`, `capability_ready=false` and `feature_stage_acceptable=false` are mandatory, not provisional defaults.

The public `validate_activation_manifest` name routes only to the unconditional legacy guard. The old v1 graph audit is private, returns `LEGACY_OUTCOME_BEARING_DEVELOPMENT_AUDIT_COMPLETE`, and can never report an activated or publication-eligible scientific scope. The old real runner is likewise retained only as a disabled reference scaffold. Each public entrypoint has a second fail-closed check after the guard, so a future edit that merely makes the current guard return cannot fall through into the old monolithic loader. New execution must use new stage-specific entrypoints; it may not reuse the dead continuation. Direct row materializers reject protected records, missing or renamed partitions, and all `partition=internal_test` rows outside a private synthetic-engineering fixture scope; ordinary rows cannot self-declare a capability through row fields.

## Required exact inputs

Do not allocate a GPU or create a run directory until every item below is independently reviewed, declares `FROZEN`, and is bound by exact SHA-256 in the capability-separated v3 preparation DAG:

1. route authorization;
2. a **train + development outcome authority**, physically separate from both internal-test authorities;
3. an **internal-test feature authority** containing no outcome, target-derived field, target-derived statistic, or reversible target proxy;
4. an **internal-test outcome authority**, physically separate from the feature authority and inaccessible until the registered release condition is replayed;
5. one row-level inclusion/rejection registry containing every admitted row ID, canonical content hash, role and exclusion reason where applicable, without leaking internal-test outcomes into earlier stages;
6. publication context/source crosswalk and source-relation evidence;
7. endpoint estimand registry;
8. global source-candidate-alias relation-component split manifest;
9. metric/zero-policy registry with a stable non-pathological development-selection metric;
10. paired strongest-same-information-set baseline comparison policy and frozen eligible-method order;
11. model/optimizer/seeds/compute config;
12. analysis-code tree manifest verified against the runtime files;
13. exact method protocol;
14. a predictor-provenance selector schema and release policy that bind the development selector to the exact predictors it compares.

The three data authorities must be different physical files/objects with different authority roles, independent hashes and a validator that rejects path aliasing, embedded outcome fields, reversible joins, sidecars and unregistered target-derived summaries. A split label inside one outcome-bearing file is not physical separation. The existing v1 monolithic `ready_dataset` therefore remains usable only through the explicitly private legacy development-audit helper; it cannot support a fresh confirmatory internal-test claim. The current public APIs expose no internal-test feature/outcome release token. Those capabilities must be newly implemented and independently replayed rather than emulated with flags on the existing materializers.

In addition, the baseline policy cannot be `PUBLICATION_QUALIFIED` merely because it names a comparator. A true full-pair unrestricted Siamese comparator must be executable under the same paired information set, and the strongest eligible comparator must be selected from complete, hash-bound **development-only** predictions under a frozen method order. Internal-test features or outcomes may not select, rename, retrain or tune the comparator. The selector must bind, for every eligible predictor, its method identity, information-set fingerprint, runtime code tree, training/config authority, train/development row authority, exact checkpoint, exact development-prediction artifact and replay result. The selector and predictor-provenance schemas now support strict development-only replay, but no real ordinary predictor evidence has been materialized under a capability-qualified activation. Therefore no selector artifact is qualified for confirmatory use, and FullPair is only a preregistered strong comparator, not a proven strongest baseline. The historical current R2 contains an explicit study embedding and is diagnostic only. The new comparator is only an `R2-style no-explicit-study adaptation`: it is not a historical-R2 single-variable ablation and endpoint/context may remain study proxies, so it is development-only pending an architecture crosswalk and hash-bound proxy audit. Each primary estimand must have an independent model/loss path; the current multi-estimand adapter is intentionally HOLD.

The current proposal activation intentionally lacks these bindings and must return HOLD.

## Phase 0 — fresh read-only recovery

On the server, first record:

```text
RUN_RECOVERY_ID=<UTC timestamp plus short random suffix>
repository absolute path
current HEAD
git status
active worktree list
running GPU jobs and owners
available disk and GPU memory
authoritative contract path and SHA-256
candidate R3X patch/bundle SHA-256
```

Do not kill, pause, or reuse unrelated jobs. Do not copy over an active dirty worktree. Materialize a new isolated worktree/run root and verify every copied byte against the local bundle manifest.

## Phase 1 — mandatory pre-test HOLD check

From the isolated R3X repository root:

```bash
python -m scripts.r3x.run_ordinary \
  --preflight-only \
  --activation-manifest configs/r3x/r3_activation_manifest.PROPOSAL.json \
  --activation-manifest-sha256 <EXACT_PROPOSAL_SHA256> \
  --output-dir <NONEXISTENT_ORDINARY_OUTPUT_DIR>
```

Expected result:

```text
status = HOLD_STAGED_DISCLOSURE_AND_SELECTOR_PROVENANCE_SCHEMA_REQUIRED_BEFORE_READY_OUTCOME_VALIDATION
output directory does not exist
legacy outcome-bearing activation validator not called
monolithic ready_dataset row loader not called
no internal-test feature or outcome artifact created
no GPU allocation
sealed_access = false
```

This same hold is currently required even for a hash-valid non-proposal v1 activation. A PASS, output directory, dataset-row read, prediction artifact, metric artifact, or GPU allocation here is a protocol bug: stop and investigate. Do not bypass the guard by editing manifest metadata.

## Phase 2 — build and independently review the capability-separated v3 authorities

- Rebuild the three separately serialized data authorities only from qualified ordinary inputs. Do not materialize a combined outcome-bearing `ready_dataset` for the confirmatory runner.
- Reject, rather than impute, missing/ambiguous context, estimand, source relation or conflicting source anchor.
- Build the split graph with an undirected edge for every accepted source-candidate relation and every frozen alias/grouping atom. Use its global transitive component as the split unit.
- Keep `source_group_id_v2` only as a same-context ranking-pool key.
- Confirm exact zero overlap of relation components across train/development/internal test.
- Label the current two-study endpoint-nested reciprocal track `DEVELOPMENT_ONLY_TWO_STUDY_NESTED_ENDPOINTS`.
- Freeze the six information-set leaderboards independently.
- Freeze the stable within-source-pool selection metric, primary model family, eligible-baseline order, strongest-baseline development selector, paired comparison and Holm family before looking at internal-test outcomes.
- Bind each development selector entry to exact predictor provenance and recursively replay every checkpoint and development-prediction reference. A selector JSON that records only method names, scores or prediction references is not sufficient.
- Freeze seeds exactly as `[11, 23, 42, 67, 101]`; seeds are optimization repeats, not biological replicates.
- Define the five seed models as one fixed prediction-level ensemble. Never average metrics across seeds to manufacture a conjunctive PASS.
- Freeze the seven negative-control definitions, control-specific tensorization/evaluation paths, checkpoint lineage, row bindings, failure semantics and independent verifiers. The current control transforms and `NOT_RUN_WITH_REASON` summaries are scaffolding only; controls remain `NOT_RUN` and impose `SCIENTIFIC_HOLD`, not a biological negative result.

Two independent reviewers should sign the authority manifest before its status becomes `FROZEN`. This phase does not authorize reading either internal-test authority.

## Phase 3 — required three-stage physical disclosure

The future capability-separated runner must enforce the following irreversible order. These are protocol stages, not labels inside a shared table:

1. **Development commit.** Open only the train + development outcome authority. Fit the publication-grade baselines, construct the complete development-only leaderboard, select the strongest eligible comparator, and commit/replay the predictor-provenance-bound selector. Neither internal-test authority may be opened.
2. **Pre-test freeze commit.** Freeze every A/B model, ensemble member, eligible baseline, control checkpoint, inference configuration, metric policy, paired-comparison policy and analysis-code tree. Write and independently replay `PRE_TEST_FREEZE_COMMIT` before reading internal-test features. No training, model selection, threshold change or control redesign is permitted after this commit.
3. **Test prediction then outcome release.** Open only the feature-only internal-test authority, generate all frozen predictions and control predictions, and write/replay an immutable test-prediction commit. Only after the qualified development selector, `PRE_TEST_FREEZE_COMMIT`, feature-authority audit and prediction commit all replay may a separate release gate open the internal-test outcome authority once for registered evaluation. The outcome shard must never flow back into fitting, selector choice, method naming, checkpoint choice or control construction.

Any missing artifact, hash mismatch, authority alias, unqualified selector, incomplete predictor provenance, incomplete control checkpoint set, or premature test-outcome access is a HOLD. A prior run that touched internal-test outcomes before these stages is permanently `DEVELOPMENT_ONLY` and cannot be upgraded retroactively.

## Phase 4 — future exact preflight through new stage-specific entrypoints

There is intentionally no currently executable PASS command. The legacy `scripts.r3x.run_ordinary` entrypoint must continue to return the Phase 1 unconditional HOLD forever; it must not acquire a flag or manifest value that falls through into its old monolithic continuation.

The future implementation must expose new, separately deployed entrypoints for:

1. privileged preparation and signed development-release production;
2. development-only baseline/A/B training and typed semantic freeze;
3. feature-only prediction and semantic prediction commit;
4. claim-first aggregate-only outcome evaluation.

Each entrypoint must receive only stage-specific file descriptors, mounts or externally released decryption capabilities. A shared run root, a full activation object, caller-provided verifier registry or hash-only commit is prohibited. Preflight must create no output directory, initialize no GPU and open neither internal-test authority.

## Phase 5 — future five-seed execution

Only after the new stage-specific implementation, independent trust root, method-scoped execution qualification, OS mount/UID/container evidence and control registry all pass exact replay may a real ordinary run proceed. No such real ordinary execution has occurred in the current work; no command should be copied from this runbook until that new interface is implemented and separately reviewed.

The fixed order is:

1. publication-grade strong baselines;
2. full-pair unrestricted Siamese, followed by the development-only R2-style no-explicit-study adaptation;
3. A exact potential-difference critic;
4. A low-rank parameter-matched algebraic diagnostic (not strongest-baseline eligible);
5. B sparse edit-set additive critic;
6. B interaction only if a separate action-support policy is publication-qualified.

The future runner writes immutable run-local snapshots of the activation manifest and all authorities, then writes `R3X_RUN_START.json` only after authority/data/split/config validation, reconstruction of the global source-relation graph, runtime code-tree verification, row-level inclusion replay, baseline-suite completeness checks, and the applicable disclosure-stage gate. It runs all five seeds with bounded cell/component-balanced epochs and mini-batches, while enforcing the three-stage commits above. A crash after start leaves the start record and any completed seed commits for audit; do not overwrite the run directory and do not describe the current implementation as automatically resumable.

The artifact order is fixed:

```text
development-authority snapshot -> RUN_PLAN -> RUN_START
  -> development checkpoints/predictions -> provenance-bound selector commit
  -> all frozen checkpoints/control plans -> PRE_TEST_FREEZE_COMMIT
  -> test-feature-authority snapshot -> frozen test predictions
  -> TEST_PREDICTION_COMMIT -> registered outcome-release gate
  -> test-outcome-authority snapshot -> metrics
  -> paired comparison -> negative-control summary -> report -> RUN_COMPLETE
```

`RUN_COMPLETE.json` is written last and only after exact replay of all references. It is an integrity closure, not sealed authority. A run with missing controls, `NOT_RUN` controls, incomplete predictor provenance, missing strongest-baseline selection evidence, or disclosure-stage failure must close as `SCIENTIFIC_HOLD`.

## Monitoring

Record absolute paths for:

```text
run root
stdout/stderr log
R3X_RUN_START.json
R3X_RUN_REPORT.json
all per-seed checkpoints and exact SHA-256 values
row-level per-seed and fixed-ensemble prediction artifacts
paired comparison, Holm and negative-control artifacts
development-only baseline evidence and selector
PRE_TEST_FREEZE_COMMIT and exact replay result
TEST_PREDICTION_COMMIT and exact replay result
internal-test outcome-release record
run-local activation/authority snapshots
RUN_COMPLETE.json and its exact replay result
environment/dependency manifest
GPU utilization and peak memory samples
```

Monitor at low frequency. Stop gracefully on NaN/Inf, edit replay failure, source-component overlap, unexpected UNK behavior, silent truncation attempt, OOM, authority hash drift, or a protected-data guard. Preserve the last valid checkpoint and full error log.

## Scientific interpretation

- Report every seed and every study-estimand-publication-context-edit-count cell.
- Compute sign/error point estimates with equal component mass; compute Spearman/selection only within qualified measured-source pools before cell macro aggregation.
- Any undefined registered primary cell or required pool is a HOLD; it may not be dropped from the macro denominator.
- Use source-relation components for paired bootstrap/permutation; apply Holm to the frozen family.
- Never select a best seed or splice one model's sign with another model's Spearman.
- Direct-delta and measured-anchor-assisted methods stay on separate leaderboards.
- Legacy signed mean-ratio enrichment is prohibited.
- A must satisfy identity-zero and swap antisymmetry algebraically and beat its <=5% parameter-gap control.
- B must lose the claimed signal under source-context/action/position negative controls; interaction must beat additive under qualified action support.
- All seven preregistered controls must have hash-bound evidence and independent replay. Any `NOT_RUN`, `UNDEFINED`, missing manifest, missing verifier, changed test-row binding or failed control is a scientific HOLD.
- `macro sign accuracy < 0.60` means effect GO is not achieved, regardless of other metrics.
- The sign gate remains exactly `macro sign accuracy >= 0.60`; no implementation, selector, C3 wording or ordinary result may lower or replace it.
- Any development-only scope remains development-only even when numeric thresholds pass.
- An effect GO never automatically unlocks O0/G1/E0 or sealed.

## C3 draft boundary

The C3 downgrade is an inactive redline only. Its proposed claim boundary describes exact guidance as a **DGM/Doob-derived adaptation**; it does not claim an exact Doob transform, does not modify the active contract, does not change any metric or threshold, and cannot be activated from ordinary results in this runbook. Activation requires a separate pre-unblinding authorization and authority-preserving contract revision. Until then, C3 has no effect on training, evaluation, GO logic or downstream unlocks.

## Sealed boundary

This runbook contains no sealed command. Do not add one. `sealed_access` remains `false`; sealed has not been accessed or reserved by this work. Sealed remains a one-time E0 action only after all ordinary gates, code, checkpoint, thresholds and analysis are frozen and independently verified.
