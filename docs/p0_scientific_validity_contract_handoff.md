# P0 Scientific Validity Contract v1 — implementation handoff

Date: 2026-07-18

Project: `/home/cunyuliu/mrna_editflow_goal/mrna_editflow`

## Final outcome

P0 Scientific Validity Contract v1 is implemented and verified. Development
workflows remain available but are always labelled `claim_tier="development_only"`
and `paper_eligible=false`. Paper training, teacher, ranker, evaluation,
statistics, and builder paths now fail closed unless their immutable split,
role, artifact, oracle, namespace, checkpoint, code, configuration, and seed
contracts verify.

The implementation does not create or claim a paper-eligible biological
result. The current GENCODE manifest remains `paper_eligible=false` for the
recorded scientific blockers.

## Safety and recovery baseline

- Repository baseline: no Git commit exists and all repository files were
  untracked. Git was not initialized and no commit was created.
- Baseline CPU test result: 208/208 passed in 171.793 seconds.
- Four Stage A jobs were running before implementation. No stop, restart,
  signal, priority change, log edit, checkpoint edit, or other process-control
  operation was performed.
- Project-external source snapshot:
  `/home/cunyuliu/mrna_editflow_goal/backups/p0-scientific-validity-contract-20260718T1400/source-before.tar.gz`
- Snapshot SHA-256:
  `ca8ffaa31bb47b4b6646c2091804c26dac883673814b47f835f7ced0d2371f86`
- The backup directory also contains pre-change process commands,
  configuration hashes, and the baseline test transcript. The snapshot excludes
  data, benchmark outputs, logs, checkpoints, and caches.

## Contract implementation

### Flow loss and auxiliary supervision

- The one-bridge-draw path is regression tested:
  `make_hybrid_batch -> sample_cond_pt -> rm_gap_tokens_with_aux -> forward -> edit_flow_loss`.
- Fixed inputs, weights, RNG state, and CPU execution reproduce loss components
  and gradients within `rtol=1e-6`, `atol=1e-7`.
- The default and all three shipped Stage A production configs disable the
  experimental structure head and set its weight to zero.
- Disabled auxiliary loss is an exact scalar zero and is not added to total
  loss. Enabled supervision requires an explicit shape-correct target plus
  source, target kind, and artifact SHA provenance; no implicit zero biological
  target remains.
- Checkpoints record whether auxiliary supervision was enabled and its target
  provenance when applicable.

### Immutable split contract

`data/split_contract.py` exposes:

- `load_and_verify_split_manifest(path, records_path=None)`;
- `select_role_records(records, contract, role)`;
- `build_split_provenance(contract, role)`;
- frozen `VerifiedSplitContract` and `VerifiedRole` objects;
- typed schema, SHA, index, overlap, and role errors.

Verification covers records and role-file SHA-256, counts, unique stable
transcript identifiers, full and selected identifier digests, in-memory record
content, integer/duplicate/range checks, complete universe coverage or a
reasoned excluded set, exact cross-role sequence overlap, and immutable cluster
assignments for paper-eligible manifests. Relative paths resolve only against
the manifest directory and cannot escape it.

Paper-eligible manifests must include cluster assignments that independently
recompute family/cluster disjointness, plus leakage-report evidence for exact
match and near-neighbour gates. A boolean declaration alone is insufficient.

### Artifact, oracle, entry-point, and namespace contracts

- Stage A/B, multiobjective and UTR teachers, hybrid teacher, cascade teacher,
  proposal ranker, evaluation, and multi-seed benchmark support explicit
  run-mode/split-manifest/split-role contracts.
- Programmatic paper APIs require a real `VerifiedSplitContract`; arbitrary
  provenance mappings and mixed-case run-mode bypasses are rejected.
- Paper training/teacher/ranker paths permit `train`; paper headline evaluation
  and multi-seed paths permit `test`. Model-selection artifacts must use `val`
  when introduced.
- Paper Stage B/evaluation checkpoints must carry compatible, paper-eligible
  train provenance and matching embedded configuration digests.
- Paper ranker training verifies both teacher JSONL and teacher summary,
  including bound artifact SHA, split/role, oracle, and upstream provenance,
  before loading the model.
- Provenance sidecars bind the exact artifact path and SHA-256. Changing or
  moving an artifact invalidates it.
- Paper artifact verification re-verifies the referenced split manifest,
  records and role identities, current source-file hashes, block reasons, and,
  for functional claims, the independent oracle manifest and artifact.
- Oracle manifests require type, source, artifact path/SHA, independence flag,
  and a non-empty independence statement. `LocalTranslationOracle` and aliases
  are always heuristic development oracles and are forbidden in paper mode.
- Development outputs cannot claim paper eligibility or write into
  `benchmark/paper/`. Eligible paper outputs must live in `benchmark/paper/`.
  Blocked diagnostic reports may remain under `docs/`.
- All five table builders, all three figure builders, SOTA readiness, and SOTA
  gap reporting use the shared paper gate. Eligible builder outputs are
  namespace-checked and receive their own bound provenance sidecars. With no
  eligible inputs they emit truthful blocked/empty reports and never backfill
  legacy development values.

### Seed and statistical semantics

- Benchmark artifacts distinguish `training_seed` from `decoder_seed` and
  record checkpoint identity plus transcript and family/cluster units.
- Resume fingerprints cover code, checkpoint, configuration, split, oracle,
  and decoding settings and reject stale results.
- Decoder-only bootstrap is explicitly conditional development uncertainty,
  not an independent-training confidence interval.
- Paper comparisons require at least three matched independent training seeds
  in both arms. Decoder observations remain nested within each training seed;
  training-seed means, not decoder draws, are the inference units.

## Existing GENCODE manifest

- Path:
  `benchmark/dev/gencode_family_leakage_protocol/split_manifest.json`
- Verified SHA-256:
  `9d35e176e8347d6a64ac9d9afc824a0a97c6c7cac8c80d99688db4cee1cc357e`
- Records: 54,680.
- Roles: train 43,744; val 5,468; test 5,468.
- Status: `paper_eligible=false`.
- Block reasons: `foundation_corpus_not_audited`,
  `external_reference_missing`, `near_neighbor_candidates_present`, and
  `records_pretruncated`.
- The current legacy report has no persisted cluster-assignment artifact and is
  therefore also incapable of satisfying the strengthened paper gate.

## Verification evidence

### Focused and full tests

Focused commands were run throughout implementation. The final full command was:

```text
cd /home/cunyuliu/mrna_editflow_goal
PYTHONPATH=/home/cunyuliu/mrna_editflow_goal \
/home/cunyuliu/miniconda3/envs/editflow/bin/python -m unittest discover -v \
  -s mrna_editflow/tests -t .
```

Actual final output:

```text
Ran 230 tests in 206.233s
OK
```

There were zero failures, errors, or skips. Tests are CPU/offline and do not
depend on the large GENCODE records, GPU execution, network access, or existing
benchmark artifacts.

The positive paper path is covered with a tiny synthetic paper-eligible split,
train-provenance checkpoint, independently manifested oracle adapter, test-role
evaluation, verified output sidecar, and verified paper builder output.

### Positive development smoke

A two-record offline evaluation was run under
`/tmp/p0_scientific_validity_dev_smoke/benchmark/dev/eval`.

Actual result:

```text
n_candidates=2
claim_tier=development_only
paper_eligible=False
oracle_type=heuristic_development_oracle
```

### Required negative paper paths

All six returned nonzero before prohibited training/evaluation work:

| Path | Return code |
|---|---:|
| paper training without split manifest | 1 |
| paper training with test role | 1 |
| paper evaluation with train role | 1 |
| paper evaluation with heuristic/non-independent oracle | 1 |
| paper builder receiving a development artifact | 2 |
| resume after input/config fingerprint change | 1 |

Machine-readable report:
`docs/p0_negative_path_verification.json`

SHA-256:
`24bc75873809a23cf8a0d519985017529e722498ac5cbdd2d6bb4b74dffa7f2c`

## Read-only Stage A health audit

- Reports: `docs/stage_a_health_audit.json` and
  `docs/stage_a_health_audit.md`.
- JSON SHA-256:
  `d9cdfb83ebcff53f8a7e6253bc34d3236ee30b010fcb2cebe96c815ed761381e`.
- Generated at: `2026-07-18T08:24:03.161155+00:00`.
- Overall verdict: `manual_review_required`.
- `process_control_actions=[]`; the audit is advisory only.

All four profiles report frequent retries, large pre-clip gradients, and no
held-out curve. Seeds 0, 2, and 5 also show frequent AMP fallback. No
non-finite-loss or non-finite-gradient condition triggered `stop_recommended`.
The report does not infer biological progress from training loss.

Scientific warnings remain: unrestricted full corpus, pre-truncated records,
the legacy running snapshot's auxiliary zero-target configuration, missing
split provenance, and missing held-out evaluation.

At the final read-only process check, all original PIDs remained in running
state and their profiles continued to advance:

| seed | PID | latest observed step |
|---:|---:|---:|
| 0 | 1495551 | 7,195 |
| 1 | 1495455 | 9,425 |
| 2 | 1495549 | 8,921 |
| 5 | 1499316 | 7,760 |

## Files created

- `data/split_contract.py`
- `eval/artifact_contract.py`
- `eval/audit_stage_a_health.py`
- `docs/flow_loss_correctness_note.md`
- `docs/stage_a_health_audit.json`
- `docs/stage_a_health_audit.md`
- `docs/p0_negative_path_verification.json`
- `docs/p0_scientific_validity_contract_handoff.md`
- `benchmark/dev/gencode_family_leakage_protocol/split_manifest.json`

## Files modified

- `core/config.py`
- `configs/stage_a_full_a100_max.json`
- `configs/stage_a_full_bs8_gradaccum4.json`
- `configs/stage_a_mig_tiny_gencode.json`
- `train_backbone.py`
- `train_adapter.py`
- `train_proposal_ranker.py`
- `baselines/multiobjective_teacher_export.py`
- `baselines/utr_teacher_export.py`
- `baselines/hybrid_teacher_export.py`
- `baselines/cascade_hard_negative_teacher.py`
- `eval/family_leakage_protocol.py`
- `eval/run_eval.py`
- `eval/run_multiseed_benchmark.py`
- `eval/compare_benchmarks.py`
- `eval/build_paper_table1_sota_landscape.py`
- `eval/build_paper_table2_t1_t7.py`
- `eval/build_paper_table3_external_baselines.py`
- `eval/build_paper_table4_architecture_ablation.py`
- `eval/build_paper_table5_scale_law_readiness.py`
- `eval/build_paper_figure1_full_length_edit_flow.py`
- `eval/build_paper_figure2_cascade_recall_precision.py`
- `eval/build_paper_figure3_oracle_gap_closure.py`
- `eval/audit_sota_readiness.py`
- `eval/sota_gap_report.py`
- `tests/test_training_sampling.py`
- `tests/test_flow_shapes.py`
- `tests/test_data_pipeline.py`
- `tests/test_eval.py`
- `scripts/eval_multiobjective_ranker_ablation_head256.sh`
- `scripts/eval_region_adapter_ablation.sh`
- `scripts/rerun_stage_a10k_ranker.sh`
- `scripts/run_ablation.sh`
- `scripts/run_after_stage_a_a100_max.sh`
- `scripts/run_after_stage_a_full1k.sh`
- `scripts/run_cascade_10krecall_hardneg_head256.sh`
- `scripts/run_head256_ranker_fair_eval.sh`
- `scripts/run_mef_utr5only_head1024.sh`
- `scripts/run_multiobjective_ranker_ablation_head256.sh`
- `scripts/run_region_adapter_ablation.sh`
- `scripts/run_stage_a10k_ranker_head256.sh`
- `scripts/run_stage_a_a100_max_train.sh`
- `scripts/run_stage_a_downstream_eval_queue.sh`
- `scripts/run_stage_a_scalelaw_sweep.sh`
- `scripts/run_stage_a_scaleup_10k.sh`

No existing benchmark, checkpoint, profile, or historical result was moved,
rewritten, or promoted.

## Remaining blockers and next goal

Paper claims remain blocked until the project has:

- a reconstructed, untruncated, fully audited foundation corpus;
- persisted family/cluster assignments and defensible cross-source,
  family-disjoint, near-neighbour-gated frozen manifests;
- a genuinely independent biological oracle with executable adapter and full
  manifest provenance;
- newly trained train-role checkpoints, validation-only selection, test-only
  evaluation, and at least three matched independent training seeds.

Recommended next goal: **P0 Data Reconstruction v1** — rebuild untruncated
GENCODE/RefSeq canonical records and derived views, then issue frozen
cross-source/family split manifests using this contract before any paper-mode
training.
