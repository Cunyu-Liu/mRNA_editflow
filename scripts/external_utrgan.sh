#!/usr/bin/env bash
# Stable wrapper for the official UTRGAN MRL/TE optimization entry point.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
TOOL_ROOT="${UTRGAN_ROOT:-${ROOT}/external_tools/UTRGAN}"
ENV_PREFIX="${UTRGAN_ENV_PREFIX:-${ROOT}/external_tools/envs/utrgan_cf}"
PYTHON_BIN="${UTRGAN_PYTHON:-${ENV_PREFIX}/bin/python}"
GPU="${UTRGAN_CUDA_VISIBLE_DEVICES:-6}"

if [[ "${1:-}" == "--version" ]]; then
  commit="$(git -C "${TOOL_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
  wgan_sha="$(sha256sum "${TOOL_ROOT}/models/checkpoint_3000.h5" | awk '{print $1}')"
  mrl_sha="$(sha256sum "${TOOL_ROOT}/models/utr_model_combined_residual_new.h5" | awk '{print $1}')"
  printf 'UTRGAN official_source=ciceklab/UTRGAN commit=%s wgan_sha256=%s mrl_sha256=%s\n' \
    "${commit}" "${wgan_sha}" "${mrl_sha}"
  exit 0
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "UTRGAN Python environment is unavailable: ${PYTHON_BIN}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
cd "${TOOL_ROOT}/src/mrl_te_optimization"
exec "${PYTHON_BIN}" optimize_te_mrl.py "$@"
