# P0 Data Reconstruction v1 — Audit Report

Date: 2026-07-19
Project: `/home/cunyuliu/mrna_editflow_goal/mrna_editflow`
Frozen root: `data/reconstructed/p0_data_reconstruction_v1`
Split root: `benchmark/dev/p0_data_reconstruction_v1`

## Final verdict

`reconstruction_complete_and_integrity_verified_paper_eligible_false_with_recorded_blockers`

P0 Data Reconstruction v1 has been rebuilt and independently integrity-audited.
The untruncated canonical records, the capped derived views, and the four frozen
cross-source/family split manifests are all present, SHA-256 verified, and
covered by the contract tests. The bundle remains `paper_eligible=false` with
two explicit, recorded block reasons; no biological claim is made.

## Stage A processes — read-only snapshot

`process_control_actions=[]`. No Stage A process was started, stopped,
signalled, reprioritised, or had its checkpoints or logs edited. All four
prior PIDs were alive at the audit and had continued to advance:

| seed | PID | latest observed step | prior handoff step |
|---:|---:|---:|---:|
| 0 | 1495551 | 8,103 | 7,195 |
| 1 | 1495455 | 10,365 | 9,425 |
| 2 | 1495549 | 9,929 | 8,921 |
| 5 | 1499316 | 8,781 | 7,760 |

## Raw sources — integrity

### GENCODE v45

- Path: `data/reconstructed/p0_data_reconstruction_v1/raw/gencode.v45.pc_transcripts.fa.gz`
- SHA-256: `2b30d353f3fe36b45fa9d7ae0aab7755700f55067d1bff26dd9fe0f7c3e05cd5`
- Size: 47,890,586 B (compressed); 285,299,152 B (uncompressed).
- `gzip_complete=true`; `complete_release_partition_set=true`.

### RefSeq human RNA (15 partitions)

- Dir: `data/reconstructed/p0_data_reconstruction_v1/raw/refseq`
- Partitions: `human.1.rna.gbff.gz` … `human.15.rna.gbff.gz`.
- All 15 partitions: `gzip_complete=true`; SHA-256 matches manifest.
- Acquisition: `resume_from_same_process_verified_frozen_raw`.
- `complete_release_partition_set=true`.

## Canonical records — untruncated

| source | parsed | emitted | canonical | truncated_5utr | truncated_3utr |
|---|---:|---:|---:|---:|---:|
| gencode_v45 | 111,048 | 111,048 | 80,290 | 0 | 0 |
| refseq_human_rna | 273,517 | 197,843 | 197,627 | 0 | 0 |

Both canonical corpora are full-length and untruncated. The
`truncated_5utr=0` / `truncated_3utr=0` invariants are enforced and tested
in `tests/test_data_reconstruction.py`.

## Derived views — model_capped_v1

Caps: `max_5utr=128`, `max_cds=1536`, `max_3utr=256` (intentional, model-facing).

| source | canonical | model_view kept | dropped_cds_too_long | truncated_5utr | truncated_3utr |
|---|---:|---:|---:|---:|---:|
| gencode_v45 | 80,290 | 54,680 | 25,610 | 27,521 | 40,742 |
| refseq_human_rna | 197,627 | 94,163 | 103,464 | 62,402 | 73,984 |

Every model-view row carries a `lineage` row binding it back to its canonical
record; lineage SHA-256 is independently verified.

## Combined bundle

- Combined records: 148,843 = 54,680 GENCODE + 94,163 RefSeq.
- Families: 17,057 total, of which 13,475 are cross-source.
- `paper_eligible=false`.
- Block reasons: `exhaustive_cross_role_near_neighbor_audit_pending`,
  `gene_symbol_alias_mapping_not_independently_audited`.

## Frozen split manifests

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

All four split manifests are `paper_eligible=false` and propagate the same two
block reasons from the combined bundle.

## Independent integrity verification (SHA-256)

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

## Contract-level verification (load_and_verify_split_manifest)

- Script: `/tmp/handoff_verify.py` (uploaded for the audit; not part of the repo).
- Method: calls `mrna_editflow.data.split_contract.load_and_verify_split_manifest`
  on each frozen split manifest. This re-runs the project's own schema, count,
  identifier, role-file SHA, transcript-id-digest, cross-role overlap, and
  cluster-disjointness verifiers against the **actual frozen records** (not
  the synthetic fixtures used by the unit tests).
- Result: **4/4 manifests verified, 0 failures.**

| split | train | val | test | excluded | family_disjoint | near_neighbor_passed | paper_eligible |
|---|---:|---:|---:|---:|---|---|---|
| combined_family | 119,075 | 14,884 | 14,884 | 0 | true | false | false |
| gencode_family | 43,744 | 5,468 | 5,468 | 0 | true | false | false |
| refseq_family | 75,331 | 9,416 | 9,416 | 0 | true | false | false |
| gencode_to_refseq | 54,680 | 2,374 | 2,297 | 89,492 | true | false | false |

All four manifests verify `family_disjoint=true` (cluster disjointness holds)
and `near_neighbor_threshold_passed=false` (consistent with the recorded
`exhaustive_cross_role_near_neighbor_audit_pending` blocker).

## Tests

### Focused

```text
PYTHONPATH=/home/cunyuliu/mrna_editflow_goal \
/home/cunyuliu/miniconda3/envs/editflow/bin/python -m unittest \
  mrna_editflow.tests.test_data_reconstruction -v

Ran 11 tests in 0.147s
OK
```

### Full suite

```text
PYTHONPATH=/home/cunyuliu/mrna_editflow_goal \
/home/cunyuliu/miniconda3/envs/editflow/bin/python -m unittest discover \
  -s mrna_editflow/tests -t .

Ran 241 tests in 175.538s
OK
```

Zero failures, errors, or skips. Up from 230 in the prior contract handoff;
the additional 11 tests are the new `test_data_reconstruction` module covering
raw integrity, canonical/derived correctness, family assignment, and
four-split contract verification.

## Manifest and code hashes

| artifact | SHA-256 |
|---|---|
| gencode source manifest | `e8b306bc2ca9b17c117363654937cfc4fd5861cdd7e7e4f09ad46ddd9ceb2bf3` |
| refseq source manifest | `c552907d3db80e9bf4099a303d69bdfc98e699abf78f831ef13d96486f997d42` |
| combined manifest | `32e803305fb6ba41c24fd996fbcae37df23c725a9ea7b3eeed82a8c8d8554fa2` |
| `data/reconstruction.py` | `dbf0c981378498054866054e9641b9c0159209884c9ee6b09c40380b8be521a0` |
| `data/split_contract.py` | `2713a56d20e2d2f9f17d3f22140078932e3c8ea25d27e8e9cf5f026c5b292a13` |
| `scripts/run_p0_data_reconstruction.sh` | `1f755f736c9c5306daa28d7a865605243f6c8242c517565bae9533cc01f2f272` |
| `scripts/resume_p0_data_reconstruction.py` | `e6da67c90b3e1484e381457e9f014261ac7005ee6e775b3a8e24349155e5d84d` |
| `tests/test_data_reconstruction.py` | `12a6e8626f0f1170d02118067f653c22a57612bb0877b06cd770b18b86f87c01` |

## Remaining blockers for paper eligibility

1. `exhaustive_cross_role_near_neighbor_audit_pending` — the recorded leakage
   reports cover exact-match and the implemented near-neighbour gate; the
   exhaustive cross-role sweep still needs independent sign-off.
2. `gene_symbol_alias_mapping_not_independently_audited` — the cross-source
   family union uses both `gene_id` and exact-protein hash per the contract,
   but the GENCODE HGNC vs RefSeq gene-name alias mapping is not yet
   independently audited.

## Scientific warnings

- No biological claim is made in this audit.
- No Stage A process control action was performed; all four prior PIDs were
  alive and advancing at the audit timestamp.
- The derived `model_capped_v1` view intentionally applies caps
  (5UTR≤128, CDS≤1536, 3UTR≤256) and drops long-CDS records. Canonical
  records remain the untruncated source of truth; every derived row carries a
  traceable lineage entry.
- The combined bundle and all four split manifests are
  `paper_eligible=false` by design. The block reasons are recorded in the
  manifest and propagated to every split manifest.

## Files

Created in this session:

- `docs/p0_data_reconstruction_v1_audit.json`
- `docs/p0_data_reconstruction_v1_audit.md`
- `docs/p0_data_reconstruction_v1_handoff.md`

No existing benchmark, checkpoint, profile, log, canonical record, derived
view, or split manifest was moved, rewritten, or promoted.
