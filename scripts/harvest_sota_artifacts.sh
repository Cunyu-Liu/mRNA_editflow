#!/usr/bin/env bash
# Harvest completed SOTA artifacts from the remote server, then audit readiness.
#
# This script is intentionally read-only with respect to the remote. It copies
# only small result/report artifacts that are needed for local claim auditing,
# writes a SHA manifest, and refreshes the unified readiness report locally.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ROOT="${LOCAL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
REMOTE_HOST="${REMOTE_HOST:-cunyuliu@36.137.135.49}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SLICE="${SLICE:-head256}"
TOP_K="${TOP_K:-64}"
MANIFEST_JSON="${MANIFEST_JSON:-${LOCAL_ROOT}/benchmark/sota_harvest_manifest_${SLICE}.json}"
MANIFEST_MD="${MANIFEST_MD:-${LOCAL_ROOT}/benchmark/sota_harvest_manifest_${SLICE}.md}"
READINESS_JSON="${READINESS_JSON:-${LOCAL_ROOT}/docs/sota_readiness_audit_${SLICE}.json}"
READINESS_MD="${READINESS_MD:-${LOCAL_ROOT}/docs/sota_readiness_audit_${SLICE}.md}"
MO_CLAIM_JSON="${MO_CLAIM_JSON:-${LOCAL_ROOT}/docs/multiobjective_scaleup_claim_audit_head256_head1024.json}"
MO_CLAIM_MD="${MO_CLAIM_MD:-${LOCAL_ROOT}/docs/multiobjective_scaleup_claim_audit_head256_head1024.md}"

usage() {
  cat <<'EOF'
Usage:
  harvest_sota_artifacts.sh [--dry-run]

Copies completed remote SOTA artifacts into the local worktree, writes a SHA
manifest, and refreshes docs/sota_readiness_audit_<slice>.{json,md}. Missing
remote artifacts are recorded in the manifest; they do not cause the harvest to
fail unless the SSH/rsync transport itself fails.

Environment overrides:
  LOCAL_ROOT, REMOTE_HOST, REMOTE_ROOT, PYTHON_BIN, SLICE, TOP_K,
  MANIFEST_JSON, MANIFEST_MD, READINESS_JSON, READINESS_MD,
  MO_CLAIM_JSON, MO_CLAIM_MD
EOF
}

target_list() {
  cat <<EOF
benchmark/region_adapter_vs_hardneg_v2_top${TOP_K}_${SLICE}.json
benchmark/region_adapter_vs_hardneg_v2_top${TOP_K}_${SLICE}.md
benchmark/region_adapter_vs_mo_grpo_top${TOP_K}_${SLICE}.json
benchmark/region_adapter_vs_mo_grpo_top${TOP_K}_${SLICE}.md
benchmark/region_adapter_vs_mo_scalar_top${TOP_K}_${SLICE}.json
benchmark/region_adapter_vs_mo_scalar_top${TOP_K}_${SLICE}.md
benchmark/region_adapter_vs_mo_pareto_top${TOP_K}_${SLICE}.json
benchmark/region_adapter_vs_mo_pareto_top${TOP_K}_${SLICE}.md
benchmark/region_adapter_vs_mo_te_only_top${TOP_K}_${SLICE}.json
benchmark/region_adapter_vs_mo_te_only_top${TOP_K}_${SLICE}.md
benchmark/region_adapter_decision_report_${SLICE}.json
benchmark/region_adapter_decision_report_${SLICE}.md
benchmark/region_adapter_result_audit_${SLICE}.json
benchmark/region_adapter_result_audit_${SLICE}.md
benchmark/protein_conditioned_cds_gc_sweep_${SLICE}.jsonl
benchmark/protein_conditioned_cds_gc_sweep_${SLICE}.summary.json
benchmark/protein_conditioned_cds_gc_sweep_${SLICE}.md
benchmark/protein_conditioned_cds_gc_sweep_${SLICE}.audit.json
benchmark/protein_conditioned_cds_gc_sweep_${SLICE}.audit.md
benchmark/multiseed_t5_public_head256_mo_te_only_top64/multiseed_summary.json
benchmark/multiseed_t5_public_head256_mo_scalar_top64/multiseed_summary.json
benchmark/multiseed_t5_public_head256_mo_pareto_top64/multiseed_summary.json
benchmark/multiseed_t5_public_head256_mo_grpo_top64/multiseed_summary.json
benchmark/multiseed_t5_public_head256_hardneg_v2_top64/multiseed_summary.json
benchmark/multiseed_t5_public_head1024_mo_te_only_top64/multiseed_summary.json
benchmark/multiseed_t5_public_head1024_mo_scalar_top64/multiseed_summary.json
benchmark/multiseed_t5_public_head1024_mo_pareto_top64/multiseed_summary.json
benchmark/multiseed_t5_public_head1024_mo_grpo_top64/multiseed_summary.json
benchmark/multiseed_t5_public_head1024_hardneg_v2_top64/multiseed_summary.json
benchmark/frozen_backbone_protocol_${SLICE}/query.jsonl
benchmark/frozen_backbone_protocol_${SLICE}/reference.jsonl
benchmark/frozen_backbone_protocol_${SLICE}/summary.json
benchmark/frozen_backbone_protocol_${SLICE}/table.md
benchmark/frozen_backbone_protocol_${SLICE}/leakage.json
docs/sota_readiness_audit_${SLICE}.json
docs/sota_readiness_audit_${SLICE}.md
EOF
}

print_plan() {
  echo "SOTA ARTIFACT HARVEST"
  echo "LOCAL_ROOT=${LOCAL_ROOT}"
  echo "REMOTE=${REMOTE_HOST}:${REMOTE_ROOT}"
  echo "SLICE=${SLICE}  TOP_K=${TOP_K}"
  echo "MANIFEST_JSON=${MANIFEST_JSON}"
  echo "MANIFEST_MD=${MANIFEST_MD}"
  echo "READINESS_JSON=${READINESS_JSON}"
  echo "READINESS_MD=${READINESS_MD}"
  echo "MO_CLAIM_JSON=${MO_CLAIM_JSON}"
  echo "MO_CLAIM_MD=${MO_CLAIM_MD}"
  echo "targets:"
  target_list | sed 's/^/  - /'
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--dry-run" ]]; then print_plan; exit 0; fi

present_file="$(mktemp)"
missing_file="$(mktemp)"
targets_file="$(mktemp)"
remote_report_file="$(mktemp)"
trap 'rm -f "${present_file}" "${missing_file}" "${targets_file}" "${remote_report_file}"' EXIT

target_list > "${targets_file}"
ssh "${REMOTE_HOST}" "cd '${REMOTE_ROOT}' && while IFS= read -r path; do if [ -e \"\${path}\" ]; then printf 'P\t%s\n' \"\${path}\"; else printf 'M\t%s\n' \"\${path}\"; fi; done" \
  < "${targets_file}" > "${remote_report_file}"
awk -F'\t' '$1 == "P" {print $2}' "${remote_report_file}" > "${present_file}"
awk -F'\t' '$1 == "M" {print $2}' "${remote_report_file}" > "${missing_file}"

if [[ -s "${present_file}" ]]; then
  rsync -av --files-from="${present_file}" "${REMOTE_HOST}:${REMOTE_ROOT}/" "${LOCAL_ROOT}/"
else
  echo "No remote target artifacts found yet."
fi

export PYTHONPATH="$(dirname "${LOCAL_ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" -m mrna_editflow.eval.audit_multiobjective_scaleup_claims \
  --project-root "${LOCAL_ROOT}" \
  --out-json "${MO_CLAIM_JSON}" \
  --out-md "${MO_CLAIM_MD}"
"${PYTHON_BIN}" -m mrna_editflow.eval.audit_sota_readiness \
  --project-root "${LOCAL_ROOT}" \
  --slice "${SLICE}" \
  --top-k "${TOP_K}" \
  --out-json "${READINESS_JSON}" \
  --out-md "${READINESS_MD}"

mkdir -p "$(dirname "${MANIFEST_JSON}")" "$(dirname "${MANIFEST_MD}")"
"${PYTHON_BIN}" - "${LOCAL_ROOT}" "${MANIFEST_JSON}" "${MANIFEST_MD}" "${present_file}" "${missing_file}" "${REMOTE_HOST}" "${REMOTE_ROOT}" "${SLICE}" "${TOP_K}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

local_root, out_json, out_md, present_path, missing_path, remote_host, remote_root, slice_name, top_k = sys.argv[1:10]

def read_lines(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]

present = read_lines(present_path)
missing = read_lines(missing_path)
rows = []
for rel in present:
    abs_path = os.path.join(local_root, rel)
    sha = None
    size = None
    if os.path.exists(abs_path):
        size = os.path.getsize(abs_path)
        h = hashlib.sha256()
        with open(abs_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        sha = h.hexdigest()
    rows.append({"path": rel, "exists_local": os.path.exists(abs_path), "size_bytes": size, "sha256": sha})

payload = {
    "artifact_kind": "sota_harvest_manifest",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "local_root": local_root,
    "remote_host": remote_host,
    "remote_root": remote_root,
    "slice": slice_name,
    "top_k": int(top_k),
    "n_present_remote": len(present),
    "n_missing_remote": len(missing),
    "present": rows,
    "missing_remote": missing,
}
with open(out_json, "w", encoding="utf-8") as fh:
    json.dump(payload, fh, indent=2, sort_keys=True)

lines = [
    "# SOTA Harvest Manifest",
    "",
    f"- Remote: `{remote_host}:{remote_root}`",
    f"- Slice: {slice_name}",
    f"- Top-k: {top_k}",
    f"- Present remote artifacts: {len(present)}",
    f"- Missing remote artifacts: {len(missing)}",
    "",
    "| path | size bytes | sha256 |",
    "|---|---:|---|",
]
for row in rows:
    lines.append(f"| `{row['path']}` | {row['size_bytes']} | `{row['sha256']}` |")
lines.extend(["", "## Missing Remote Artifacts", "", "| path |", "|---|"])
for rel in missing:
    lines.append(f"| `{rel}` |")
with open(out_md, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
PY

echo "Harvest manifest -> ${MANIFEST_JSON}"
echo "Readiness audit  -> ${READINESS_JSON}"
