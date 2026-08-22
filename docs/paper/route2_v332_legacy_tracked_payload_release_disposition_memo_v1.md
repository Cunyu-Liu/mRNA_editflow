# Route 2 V3.3.2 legacy tracked-payload release disposition

## Decision

Do not create a formal tag or GitHub Release from the current HEAD. The four
legacy B0 readers now fail closed on the superseded repository-root split
directory. After explicit user authorization, preserve the five payloads under
the Route 2 `/mnt` root and stop tracking them in the current Git tree. Do not
rewrite shared Git history as part of this task; any repository-wide history
rewrite requires separate authorization and coordination.

## Evidence and conflict

The current tree tracks one 46,498-byte discovery Parquet and four superseded B0
JSONL files totalling 34,739,577 bytes. The five-file total is 34,786,075 bytes.
The V3.3.2 contract excludes raw/canonical JSONL and Parquet payloads from GitHub.
The legacy v3.1 contract simultaneously requires the four old B0 JSONL files to
remain unmodified as historical evidence and requires active loaders to reject
them.

The content of the five payloads was not opened for this audit. Path, tracking,
size and text-reference checks found no current Route 2 V3.3.2 runtime consumer.
Four callable legacy entrypoints previously directly read the old
`data/b0_splits` directory:

- `d1_staging/scripts/b0/audit_split_manifests.py`
- `d1_staging/scripts/b0/eval_tracks.py`
- `d1_staging/scripts/b0/leakage_audit.py`
- `d1_staging/scripts/fm0/fm0_exposure_audit.py`

They now share `d1_staging/scripts/b0/legacy_split_guard.py`, which raises
`SUPERSEDED_NOT_LOADABLE` before a canonical-record or split-manifest read when
the requested split root is the preserved repository directory. Seven focused
tests cover the shared guard, all four command entrypoints, unchanged JSONL
sizes and guard-before-load ordering. The discovery Parquet has one producer,
`scripts/data/import_excel_inventory.py`, but no current Route 2 consumer.
Retaining the five files in a formal release still violates the current Git
payload boundary.

## Ordered migration after authorization

1. Preserve the five payloads under
   `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/legacy_repository_payloads/`
   with a small provenance note. Do not create project-generated checksum files.
2. Change the Excel inventory producer default away from the Git tree, retaining
   only its small audit summary in Git.
3. After explicit deletion/migration authorization, stop tracking the five
   payloads in the current HEAD and add narrow ignore rules for the same paths.
4. Run focused and full V3.3.2 tests and re-adjudicate the internal release
   candidate. A formal tag/Release remains unauthorized until that adjudication.

## Current boundary

This memo authorizes no deletion, move, copy, history rewrite, release or tag.
It records a narrow behavior change to the four legacy readers only; no tracked
payload was opened or modified.
It corrects the release description: large Route 2 runtime artifacts remain
outside Git, but five legacy data payloads are still tracked in the working
repository and therefore keep the formal-release payload boundary non-compliant.
