# P0 Data Reconstruction v1 — Implementation Handoff

Date: 2026-07-19
Project: `/home/cunyuliu/mrna_editflow_goal/mrna_editflow`
Prior contract handoff: `docs/p0_scientific_validity_contract_handoff.md`
(SHA-256 `7aa650e73a9e8c66972f38fb20096fad9696aada9663d1370ccd33274a8ccab2`)

## Final outcome

P0 Data Reconstruction v1 is implemented, independently integrity-audited, and
contract-tested. The untruncated GENCODE v45 and RefSeq human RNA canonical
records, the capped `model_capped_v1` derived views with full lineage, and the
four frozen cross-source/family split manifests are all present, SHA-256
verified, and pass the contract test suite. The bundle remains
`paper_eligible=false` with two explicit, recorded block reasons.

No biological claim is made. No Stage A process was started, stopped,
signalled, reprioritised, or had its checkpoints or logs edited. No existing
benchmark, checkpoint, profile, log, canonical record, derived view, or split
manifest was moved, rewritten, or promoted.

## Safety and recovery baseline

- Repository baseline: still no Git commit; project files remain untracked.
- Baseline full test result before this session (per prior handoff): 230/230
  passed in 206.233s. After this session: 241/241 passed in 175.538s. The 11
  additional tests are the new `tests/test_data_reconstruction.py` module.
- Stage A processes: all four prior PIDs (1495455, 1495549, 1495551, 1499316)
  were alive at the audit and had advanced from 7,195/9,425/8,921/7,760 steps
  to 8,103/10,365/9,929/8,781 steps. `process_control_actions=[]`.
- The P0 Data Reconstruction v1 frozen tree at
  `data/reconstructed/p0_data_reconstruction_v1/` and the frozen split
  namespace at `benchmark/dev/p0_data_reconstruction_v1/` were already present
  from the prior session's rebuild; this session's work was an independent
  read-only integrity audit plus contract test verification, and the issuance
  of the audit/handoff documents. No artifact under either frozen tree was
  modified.

## What was verified

### Untruncated canonical records

| source | canonical count | truncated_5utr | truncated_3utr |
|---|---:|---:|---:|
| gencode_v45 | 80,290 | 0 | 0 |
| refseq_human_rna | 197,627 | 0 | 0 |

The `truncated_5utr=0` / `truncated_3utr=0` invariants are enforced by
`data/reconstruction.py` and regression-tested in
`tests/test_data_reconstruction.py::CanonicalAndDerivedTest::test_canonical_keeps_full_regions_and_derived_view_is_traceable`.

### Capped derived views with lineage

`model_capped_v1` (caps: `5UTR<=128`, `CDS<=1536`, `3UTR<=256`):

| source | canonical | model_view kept | dropped_cds_too_long | truncated_5utr | truncated_3utr |
|---|---:|---:|---:|---:|---:|
| gencode_v45 | 80,290 | 54,680 | 25,610 | 27,521 | 40,742 |
| refseq_human_rna | 197,627 | 94,163 | 103,464 | 62,402 | 73,984 |

Every model-view row has a `lineage` row binding it to its canonical record.
Canonical records remain the untruncated source of truth; the capped view is
model-facing and traceable.

### Frozen cross-source / family split manifests

Four split manifests under `benchmark/dev/p0_data_reconstruction_v1/`. Each
manifest carries its own SHA-256, `train/val/test.idx`,
`cluster_assignments.json`, and `leakage_report.json`. The
`gencode_to_refseq` cross-source split additionally persists an `excluded.idx`
for its reasoned excluded universe (89,492 records).

| split | records | train | val | test | excluded | manifest_sha256 |
|---|---:|---:|---:|---:|---:|---|
| combined_family | 148,843 | 119,075 | 14,884 | 14,884 | 0 | `8add072a57c53746ba663942797050b068e5487f235a46ddfce85606dd3bae2c` |
| gencode_family | 54,680 | 43,744 | 5,468 | 5,468 | 0 | `5a86f3bec24dca96e338a326cf3b53418e5fa3a27a7315c9a6eb5f48a42e69d7` |
| refseq_family | 94,163 | 75,331 | 9,416 | 9,416 | 0 | `42535fdef0cdb076f759414001860ef1691aeaff4edfc1e088e5161aae8be4ab` |
| gencode_to_refseq | 148,843 | 54,680 | 2,374 | 2,297 | 89,492 | `1965f86779bcf1492ea6efe68f6485f5006d90fe7cc6014639a3c04b01b06367` |

Combined bundle: 148,843 records = 54,680 GENCODE + 94,163 RefSeq; 17,057
families of which 13,475 are cross-source.

### Independent integrity verification

- Script: `/tmp/verify_p0_recon.py` (uploaded for the audit; not part of the repo).
- Method: independent SHA-256 recompute of every artifact referenced in the
  frozen manifests; presence checks for every `train/val/test.idx`,
  `excluded.idx`, `cluster_assignments.json`, and `leakage_report.json`.
- Result: **59/59 checks passed, 0 failed.**
- Coverage: 1 GENCODE raw + 15 RefSeq raw partitions + GENCODE canonical
  records/metadata + GENCODE model_view records/lineage + RefSeq canonical
  records/metadata + RefSeq model_view records/lineage + combined
  records/metadata + family assignments/evidence + 2 source manifest refs + 4
  split manifests + 12 role idx files + 1 excluded.idx + 4 cluster_assignments
  + 4 leakage_reports + 4 canonical untruncated invariants.

### Tests

Focused:

```text
PYTHONPATH=/home/cunyuliu/mrna_editflow_goal \
/home/cunyuliu/miniconda3/envs/editflow/bin/python -m unittest \
  mrna_editflow.tests.test_data_reconstruction -v
Ran 11 tests in 0.147s
OK
```

Full suite:

```text
PYTHONPATH=/home/cunyuliu/mrna_editflow_goal \
/home/cunyuliu/miniconda3/envs/editflow/bin/python -m unittest discover \
  -s mrna_editflow/tests -t .
Ran 241 tests in 175.538s
OK
```

Zero failures, errors, or skips. Tests are CPU/offline and do not depend on
GPU execution, network access, or existing benchmark artifacts.

## Read-only Stage A health snapshot

All four Stage A profiles were observed, not modified. `process_control_actions=[]`.

| seed | PID | latest observed step | prior handoff step |
|---:|---:|---:|---:|
| 0 | 1495551 | 8,103 | 7,195 |
| 1 | 1495455 | 10,365 | 9,425 |
| 2 | 1495549 | 9,929 | 8,921 |
| 5 | 1499316 | 8,781 | 7,760 |

The four PIDs remained in running state and their profiles continued to
advance between the prior handoff and this audit. No stop, restart, signal,
priority change, log edit, or checkpoint edit was performed.

## Manifest, code, and document hashes

| artifact | SHA-256 |
|---|---|
| gencode source manifest | `e8b306bc2ca9b17c117363654937cfc4fd5861cdd7e7e4f09ad46ddd9ceb2bf3` |
| refseq source manifest | `c552907d3db80e9bf4099a303d69bdfc98e699abf78f831ef13d96486f997d42` |
| combined manifest | `32e803305fb6ba41c24fd996fbcae37df23c725a9ea7b3eeed82a8c8d8554fa2` |
| GENCODE v45 raw | `2b30d353f3fe36b45fa9d7ae0aab7755700f55067d1bff26dd9fe0f7c3e05cd5` |
| `data/reconstruction.py` | `dbf0c981378498054866054e9641b9c0159209884c9ee6b09c40380b8be521a0` |
| `data/split_contract.py` | `2713a56d20e2d2f9f17d3f22140078932e3c8ea25d27e8e9cf5f026c5b292a13` |
| `scripts/run_p0_data_reconstruction.sh` | `1f755f736c9c5306daa28d7a865605243f6c8242c517565bae9533cc01f2f272` |
| `scripts/resume_p0_data_reconstruction.py` | `e6da67c90b3e1484e381457e9f014261ac7005ee6e775b3a8e24349155e5d84d` |
| `tests/test_data_reconstruction.py` | `12a6e8626f0f1170d02118067f653c22a57612bb0877b06cd770b18b86f87c01` |
| prior contract handoff doc | `7aa650e73a9e8c66972f38fb20096fad9696aada9663d1370ccd33274a8ccab2` |

## Remaining blockers for paper eligibility

The P0 Data Reconstruction v1 bundle and all four split manifests remain
`paper_eligible=false` with two recorded block reasons:

1. `exhaustive_cross_role_near_neighbor_audit_pending` — the recorded leakage
   reports cover exact-match and the implemented near-neighbour gate, but the
   exhaustive cross-role near-neighbour sweep still needs independent sign-off.
2. `gene_symbol_alias_mapping_not_independently_audited` — the cross-source
   family union uses both `gene_id` and exact-protein hash per the contract,
   but the GENCODE HGNC vs RefSeq gene-name alias mapping is not yet
   independently audited.

These are honest, recorded blockers. The v1 reconstruction is the
untruncated, fully SHA-verified foundation corpus; the v2 work that lifts
these two blockers can build directly on top of this frozen namespace without
rebuilding canonical records or derived views.

## Files created in this session

- `docs/p0_data_reconstruction_v1_audit.json`
- `docs/p0_data_reconstruction_v1_audit.md`
- `docs/p0_data_reconstruction_v1_handoff.md`

## Files modified in this session

None. No existing benchmark, checkpoint, profile, log, canonical record,
derived view, or split manifest was moved, rewritten, or promoted. No Stage A
process was started, stopped, signalled, reprioritised, or had its
checkpoints or logs edited.

## Recommended next goal

**P0 Data Reconstruction v2 — lift the two recorded paper-eligibility
blockers** without rebuilding canonical records or derived views:

1. Run and persist the exhaustive cross-role near-neighbour audit for every
   frozen split manifest; sign off and flip
   `exhaustive_cross_role_near_neighbor_audit_pending`.
2. Independently audit the GENCODE HGNC vs RefSeq gene-name alias mapping
   used by the cross-source family union; sign off and flip
   `gene_symbol_alias_mapping_not_independently_audited`.

Once both blockers are lifted, the combined bundle and the four split
manifests can be promoted to `paper_eligible=true` under the contract
defined in `docs/p0_scientific_validity_contract_handoff.md` and the
`data/split_contract.py` module, before any paper-mode training.
