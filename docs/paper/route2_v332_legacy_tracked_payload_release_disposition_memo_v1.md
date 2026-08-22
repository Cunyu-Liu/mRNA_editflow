# Route 2 V3.3.2 legacy tracked-payload release disposition

## Decision

Do not create a formal tag or GitHub Release from the current HEAD. After
explicit user authorization, first make the four legacy B0 readers fail closed,
then preserve the five payloads under the Route 2 `/mnt` root and stop tracking
them in the current Git tree. Do not rewrite shared Git history as part of this
task; any repository-wide history rewrite requires separate authorization and
coordination.

## Evidence and conflict

The current tree tracks one 46,498-byte discovery Parquet and four superseded B0
JSONL files totalling 34,739,577 bytes. The five-file total is 34,786,075 bytes.
The V3.3.2 contract excludes raw/canonical JSONL and Parquet payloads from GitHub.
The legacy v3.1 contract simultaneously requires the four old B0 JSONL files to
remain unmodified as historical evidence and requires active loaders to reject
them.

The content of the five payloads was not opened for this audit. Path, tracking,
size and text-reference checks found no current Route 2 V3.3.2 runtime consumer.
However, four callable legacy entrypoints still directly read the old
`data/b0_splits` directory:

- `d1_staging/scripts/b0/audit_split_manifests.py`
- `d1_staging/scripts/b0/eval_tracks.py`
- `d1_staging/scripts/b0/leakage_audit.py`
- `d1_staging/scripts/fm0/fm0_exposure_audit.py`

No corresponding negative-loader test was found. The discovery Parquet has one
producer, `scripts/data/import_excel_inventory.py`, but no current Route 2
consumer. Removing files before changing the four readers would leave callable
legacy commands with missing default inputs; retaining the files in a formal
release would violate the current Git payload boundary.

## Ordered migration after authorization

1. Add fail-closed guards and negative tests for the four legacy B0 readers,
   without modifying the four JSONL files.
2. Preserve the five payloads under
   `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/legacy_repository_payloads/`
   with a small provenance note. Do not create project-generated checksum files.
3. Change the Excel inventory producer default away from the Git tree, retaining
   only its small audit summary in Git.
4. After explicit deletion/migration authorization, stop tracking the five
   payloads in the current HEAD and add narrow ignore rules for the same paths.
5. Run focused and full V3.3.2 tests and re-adjudicate the internal release
   candidate. A formal tag/Release remains unauthorized until that adjudication.

## Current boundary

This memo authorizes no deletion, move, copy, history rewrite, release or tag.
It corrects the release description: large Route 2 runtime artifacts remain
outside Git, but five legacy data payloads are still tracked in the working
repository and therefore keep the formal-release payload boundary non-compliant.
