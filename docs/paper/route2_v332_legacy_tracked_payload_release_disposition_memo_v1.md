# Route 2 V3.3.2 legacy payload current-HEAD migration disposition

## Decision

The five legacy payloads have been preserved under
`/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/legacy_repository_payloads/`
and removed from current-HEAD tracking under explicit user authorization. Five
exact ignore rules prevent the same generated paths from being re-added. The
current-HEAD formal-release payload boundary is therefore compliant.

This resolution does not authorize a formal tag or GitHub Release. Minimum-
package, accountable-human rights and license review, clean-environment
reproduction, immutable archive and manuscript metadata/disclosure blockers
remain. Shared Git history was not rewritten.

## Preserved files

The migration covers exactly one 46,498-byte discovery Parquet and four
superseded B0 JSONL files totalling 34,739,577 bytes. The five-file total is
34,786,075 bytes. Each source/destination pair was checked by exact path and
byte size after a no-overwrite copy; payload content was not opened and no
project-generated checksum file was created.

| former current-HEAD path | preserved `/mnt` file | bytes |
|---|---|---:|
| `data_registry/excel_inventory.parquet` | `excel_inventory.parquet` | 46,498 |
| `data/b0_splits/split_study_disjoint.jsonl` | `split_study_disjoint.jsonl` | 11,462,850 |
| `data/b0_splits/split_cross_region_transfer.jsonl` | `split_cross_region_transfer.jsonl` | 11,299,013 |
| `data/b0_splits/split_5utr_source_disjoint.jsonl` | `split_5utr_source_disjoint.jsonl` | 7,638,905 |
| `data/b0_splits/split_3utr_source_disjoint.jsonl` | `split_3utr_source_disjoint.jsonl` | 4,338,809 |

The destination includes `PROVENANCE.md`, recording the source branch and
pre-migration commit, exact file mapping, sizes, authorization boundary and the
absence of shared-history rewrite or formal release.

## Repository behavior after migration

The four callable legacy B0 entrypoints remain fail closed on the repository-
root `data/b0_splits` path through
`d1_staging/scripts/b0/legacy_split_guard.py`. They raise
`SUPERSEDED_NOT_LOADABLE` before canonical-record or split-manifest reads. The
guard remains effective even though the four JSONL files are absent from a
fresh checkout.

The discovery Parquet producer now defaults future output to
`/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/data_registry/excel_inventory.parquet`
while retaining the small historical audit Markdown in Git. Its renderer
records the actual selected Parquet path.

## Current boundary

Current HEAD tracks none of the five payloads. The payload-policy component of
the internal release candidate is resolved, but public payload redistribution
is not authorized: the 14-study accountable-human rights register still has
zero completed/signoff rows. No Development TEST, new final Evaluation,
E-MTAB-10902 outcome, sealed GSE246381 outcome or guided XEditFlow output was
opened during this migration.
