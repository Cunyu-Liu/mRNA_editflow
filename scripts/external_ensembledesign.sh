#!/usr/bin/env bash
# Stable wrapper for the official EnsembleDesign Python entry point.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
TOOL_ROOT="${ENSEMBLEDESIGN_ROOT:-${ROOT}/external_tools/EnsembleDesign}"
PYTHON_BIN="${ENSEMBLEDESIGN_PYTHON:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"

if [[ "${1:-}" == "--version" ]]; then
  commit="$(git -C "${TOOL_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
  sha="$(sha256sum "${TOOL_ROOT}/bin/EnsembleDesign" | awk '{print $1}')"
  printf 'EnsembleDesign official_source=LinearFold/EnsembleDesign commit=%s binary_sha256=%s\n' "${commit}" "${sha}"
  exit 0
fi

if [[ ! -x "${TOOL_ROOT}/bin/EnsembleDesign" ]]; then
  echo "EnsembleDesign binary is not executable: ${TOOL_ROOT}/bin/EnsembleDesign" >&2
  exit 2
fi

export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
cd "${TOOL_ROOT}"
exec "${PYTHON_BIN}" EnsembleDesign.py "$@"
