#!/usr/bin/env bash
# Stable wrapper for the official UTailoR web-tool workflow and weights.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
OFFICIAL_ROOT="${UTAILOR_ROOT:-${ROOT}/external_tools/UTailoR_official}"
TOOL_ROOT="${UTAILOR_WEB_ROOT:-${OFFICIAL_ROOT}/web_tool/UtailR web tool}"
ENV_PREFIX="${UTAILOR_ENV_PREFIX:-${ROOT}/external_tools/envs/utrgan_cf}"
PYTHON_BIN="${UTAILOR_PYTHON:-${ENV_PREFIX}/bin/python}"
GPU="${UTAILOR_CUDA_VISIBLE_DEVICES:-2}"

if [[ "${1:-}" == "--version" ]]; then
  archive_sha="$(sha256sum "${OFFICIAL_ROOT}/UtailR_web_tool.rar" | awk '{print $1}')"
  predictor_sha="$(sha256sum "${TOOL_ROOT}/utailor_app/utailor_utils/models/CGRU_25_100.hd5/saved_model.pb" | awk '{print $1}')"
  generator_sha="$(sha256sum "${TOOL_ROOT}/utailor_app/utailor_utils/models/UTR_genAE_CGRU_25_100_compiled.hd5/saved_model.pb" | awk '{print $1}')"
  printf 'UTailoR official_source=cuilab.cn/utailor archive_sha256=%s predictor_sha256=%s generator_sha256=%s license=not_present_in_archive\n' \
    "${archive_sha}" "${predictor_sha}" "${generator_sha}"
  exit 0
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "UTailoR Python environment is unavailable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${TOOL_ROOT}/utailor_app/utailor_utils/workflow.py" ]]; then
  echo "UTailoR official workflow is unavailable: ${TOOL_ROOT}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m mrna_editflow.baselines.external_utailor_runner \
  --tool-root "${TOOL_ROOT}" "$@"
