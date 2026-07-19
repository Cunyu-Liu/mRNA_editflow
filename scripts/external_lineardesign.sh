#!/usr/bin/env bash
# Stable wrapper for the official LinearDesign binary bundled by EnsembleDesign.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
TOOL_ROOT="${LINEARDESIGN_ROOT:-${ROOT}/external_tools/EnsembleDesign/tools/LinearDesign}"
LAMBDA="${LINEARDESIGN_LAMBDA:-1.0}"
VERBOSE="${LINEARDESIGN_VERBOSE:-0}"
CODON_TABLE="${LINEARDESIGN_CODON_TABLE:-codon_usage_freq_table_human.csv}"
BINARY="${LINEARDESIGN_CORE_BIN:-${TOOL_ROOT}/bin/LinearDesign_2D}"

if [[ "${1:-}" == "--version" ]]; then
  commit="$(git -C "${ROOT}/external_tools/EnsembleDesign" rev-parse HEAD 2>/dev/null || echo unknown)"
  sha="$(sha256sum "${BINARY}" | awk '{print $1}')"
  printf 'LinearDesign official_source=LinearFold/EnsembleDesign commit=%s binary_sha256=%s\n' "${commit}" "${sha}"
  exit 0
fi

if [[ ! -x "${BINARY}" ]]; then
  echo "LinearDesign binary is not executable: ${BINARY}" >&2
  exit 2
fi

cd "${TOOL_ROOT}"
exec "${BINARY}" "${LAMBDA}" "${VERBOSE}" "${CODON_TABLE}"
