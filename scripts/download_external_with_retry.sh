#!/bin/bash
# Retry-with-backoff driver for the external catalog download (2026-07-27).
# Zenodo/GitHub are intermittently unreachable from this network; NCBI is
# throttled to ~10KB/s and handled by the parallel-range driver instead.
# Usage: bash scripts/download_external_with_retry.sh DATASET [DATASET...]
set -u
TARGET_ROOT="${TARGET_ROOT:-/mnt/cunyuliu/mrna_editflow_extdata/raw}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
FAILED=()
for ds in "$@"; do
  ok=0
  for attempt in 1 2 3 4 5; do
    echo "[$(date -u +%FT%TZ)] $ds attempt $attempt -> $TARGET_ROOT"
    if python -m data.download_external_catalog --datasets "$ds" --target-root "$TARGET_ROOT"; then
      ok=1
      break
    fi
    sleep $((attempt * 30))
  done
  if [ "$ok" -ne 1 ]; then
    FAILED+=("$ds")
    echo "[$(date -u +%FT%TZ)] $ds FAILED after 5 attempts"
  fi
done
if [ "${#FAILED[@]}" -gt 0 ]; then
  printf 'FAILED\t%s\n' "${FAILED[@]}"
  exit 1
fi
echo "ALL_OK"
