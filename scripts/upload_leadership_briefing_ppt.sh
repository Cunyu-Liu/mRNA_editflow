#!/usr/bin/env bash
# Retry upload for the generated leadership briefing deck.
set -euo pipefail

ROOT="${ROOT:-/Users/bytedance/Documents/research/mrna_editflow}"
REMOTE="${REMOTE:-cunyuliu@36.137.135.49:/home/cunyuliu/mrna_editflow_goal/mrna_editflow/}"
TRIES="${TRIES:-20}"
SLEEP_SECONDS="${SLEEP_SECONDS:-60}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-45}"

cd "${ROOT}"

for attempt in $(seq 1 "${TRIES}"); do
  echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] upload attempt ${attempt}/${TRIES}"
  if rsync -avR -e "ssh -o BatchMode=yes -o ConnectTimeout=${CONNECT_TIMEOUT}" \
    docs/presentations/mrna_editflow_leadership_briefing_20260715.pptx \
    docs/presentations/mrna_editflow_leadership_briefing_20260715_outline.md \
    scripts/build_leadership_briefing_ppt.py \
    "${REMOTE}"; then
    echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] upload succeeded"
    ssh -o BatchMode=yes -o ConnectTimeout="${CONNECT_TIMEOUT}" "${REMOTE%%:*}" \
      'cd /home/cunyuliu/mrna_editflow_goal/mrna_editflow && sha256sum docs/presentations/mrna_editflow_leadership_briefing_20260715.pptx docs/presentations/mrna_editflow_leadership_briefing_20260715_outline.md scripts/build_leadership_briefing_ppt.py'
    exit 0
  fi
  echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] upload failed; retrying after ${SLEEP_SECONDS}s"
  sleep "${SLEEP_SECONDS}"
done

echo "[$(date '+%Y-%m-%dT%H:%M:%S%z')] upload failed after ${TRIES} attempts" >&2
exit 1
