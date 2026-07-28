#!/bin/bash
# Parallel byte-range downloader for NCBI-hosted catalog datasets (2026-07-27).
#
# Rationale: ftp.ncbi.nlm.nih.gov is throttled to ~10 KB/s per connection from
# this network, but throttling is per-connection (verified: 4 parallel range
# requests aggregate linearly). This script downloads each file with NPAR
# parallel byte-range parts, per-part resume, stall abort and retries, then
# concatenates parts and verifies total byte size. After all files of a
# dataset are present it invokes the catalog downloader once so the standard
# manifest.json + SHA-256 records are written by the audited code path.
#
# Usage: bash scripts/ncbi_parallel_download.sh DATASET [NPAR]
set -u
DS="$1"
NPAR="${2:-24}"
TARGET_ROOT="${TARGET_ROOT:-/mnt/cunyuliu/mrna_editflow_extdata/raw}"
DEST_DIR="$TARGET_ROOT/$DS"
mkdir -p "$DEST_DIR"

# Pull the registered file list (url<TAB>filename) from the static registry.
mapfile -t LINES < <(python - "$DS" <<'PY'
import sys
from data.download_external_catalog import EXTERNAL_CATALOG
for f in EXTERNAL_CATALOG[sys.argv[1]]["files"]:
    print(f["url"], f["filename"], sep="\t")
PY
)

fetch_part() {  # url part_path start end -> 0 on complete
  local url="$1" part="$2" start="$3" end="$4" want=$((end - start + 1))
  for attempt in 1 2 3 4 5 6 7 8; do
    local have=0
    [ -f "$part" ] && have=$(stat -c%s "$part")
    if [ "$have" -eq "$want" ]; then return 0; fi
    if [ "$have" -gt "$want" ]; then rm -f "$part"; have=0; fi
    # stall abort: <1KB/s for 30s kills curl so we retry
    if [ "$have" -gt 0 ]; then
      curl -sS --speed-limit 1024 --speed-time 30 -m 7200 \
           -r $((start + have))-"$end" "$url" >> "$part" && :
    else
      curl -sS --speed-limit 1024 --speed-time 30 -m 7200 \
           -r "${start}"-"$end" -o "$part" "$url" && :
    fi
    [ -f "$part" ] && [ "$(stat -c%s "$part")" -eq "$want" ] && return 0
    sleep $((attempt * 15))
  done
  return 1
}

download_file() {  # url dest
  local url="$1" dest="$2"
  if [ -f "$dest" ]; then echo "SKIP $dest (exists)"; return 0; fi
  local size
  size=$(curl -sSI -m 60 "$url" | grep -i '^content-length' | tail -1 | tr -dc '0-9')
  if [ -z "${size:-}" ]; then echo "ERR no content-length for $url"; return 1; fi
  local chunk=$(( (size + NPAR - 1) / NPAR ))
  local pids=() i start end
  for ((i = 0; i < NPAR; i++)); do
    start=$((i * chunk)); end=$(( (i + 1) * chunk - 1 ))
    [ "$start" -ge "$size" ] && break
    [ "$end" -ge "$size" ] && end=$((size - 1))
    fetch_part "$url" "$dest.rpart$(printf %03d "$i")" "$start" "$end" &
    pids+=($!)
  done
  local rc=0 p
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  [ "$rc" -eq 0 ] || { echo "ERR parts failed for $dest"; return 1; }
  : > "$dest.cat"
  for f in "$dest".rpart*; do cat "$f" >> "$dest.cat"; done
  if [ "$(stat -c%s "$dest.cat")" -ne "$size" ]; then
    echo "ERR size mismatch for $dest"; rm -f "$dest.cat"; return 1
  fi
  mv "$dest.cat" "$dest"
  rm -f "$dest".rpart*
  echo "OK $dest ($size bytes)"
}

rc_all=0
for line in "${LINES[@]}"; do
  url="${line%%$'\t'*}"; fn="${line##*$'\t'}"
  download_file "$url" "$DEST_DIR/$fn" || rc_all=1
done
[ "$rc_all" -eq 0 ] || { echo "DATASET_INCOMPLETE $DS"; exit 1; }

# All files present: audited manifest + SHA-256 path (existing files skipped).
python -m data.download_external_catalog --datasets "$DS" --target-root "$TARGET_ROOT"
