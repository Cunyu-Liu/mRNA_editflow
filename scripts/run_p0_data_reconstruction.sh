#!/usr/bin/env bash
# P0 Data Reconstruction v1: immutable full canonical corpora + derived views.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
PYTHON_BIN="${PYTHON_BIN:-/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10}"
FROZEN_ROOT="${FROZEN_ROOT:-${ROOT}/data/reconstructed/p0_data_reconstruction_v1}"
SPLIT_ROOT="${SPLIT_ROOT:-${ROOT}/benchmark/dev/p0_data_reconstruction_v1}"
GENCODE_SOURCE="${GENCODE_SOURCE:-${ROOT}/data/raw/gencode.v45.pc_transcripts.fa.gz}"
GENCODE_SHA256="${GENCODE_SHA256:-2b30d353f3fe36b45fa9d7ae0aab7755700f55067d1bff26dd9fe0f7c3e05cd5}"
REFSEQ_URL="${REFSEQ_URL:-https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/mRNA_Prot/human.1.rna.gbff.gz}"
REFSEQ_SOURCE="${REFSEQ_SOURCE:-}"
REFSEQ_SOURCE_DIR="${REFSEQ_SOURCE_DIR:-}"
REFSEQ_SIZE_BYTES="${REFSEQ_SIZE_BYTES:-235266946}"
REFSEQ_SHA256="${REFSEQ_SHA256:-}"
SEED="${SEED:-20260714}"

if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' \
    "ROOT=${ROOT}" \
    "FROZEN_ROOT=${FROZEN_ROOT}" \
    "SPLIT_ROOT=${SPLIT_ROOT}" \
    "GENCODE_SOURCE=${GENCODE_SOURCE}" \
    "GENCODE_SHA256=${GENCODE_SHA256}" \
    "REFSEQ_URL=${REFSEQ_URL}" \
    "REFSEQ_SOURCE=${REFSEQ_SOURCE}" \
    "REFSEQ_SOURCE_DIR=${REFSEQ_SOURCE_DIR}" \
    "REFSEQ_SIZE_BYTES=${REFSEQ_SIZE_BYTES}" \
    "REFSEQ_SHA256=${REFSEQ_SHA256}" \
    "SEED=${SEED}"
  exit 0
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python interpreter: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${GENCODE_SOURCE}" ]]; then
  echo "Missing GENCODE source: ${GENCODE_SOURCE}" >&2
  exit 2
fi
if [[ -e "${FROZEN_ROOT}" || -e "${SPLIT_ROOT}" ]]; then
  echo "Refusing to overwrite frozen reconstruction or split namespace." >&2
  exit 2
fi

cd "$(dirname "${ROOT}")"
refseq_args=(--refseq-url "${REFSEQ_URL}" --refseq-size-bytes "${REFSEQ_SIZE_BYTES}")
if [[ -n "${REFSEQ_SOURCE_DIR}" ]]; then
  if [[ ! -d "${REFSEQ_SOURCE_DIR}" ]]; then
    echo "Missing frozen RefSeq source directory: ${REFSEQ_SOURCE_DIR}" >&2
    exit 2
  fi
  refseq_args=(--refseq-source-dir "${REFSEQ_SOURCE_DIR}")
elif [[ -n "${REFSEQ_SOURCE}" ]]; then
  if [[ ! -f "${REFSEQ_SOURCE}" ]]; then
    echo "Missing frozen RefSeq source: ${REFSEQ_SOURCE}" >&2
    exit 2
  fi
  refseq_args=(--refseq-source "${REFSEQ_SOURCE}" --refseq-size-bytes "${REFSEQ_SIZE_BYTES}")
fi
if [[ -n "${REFSEQ_SHA256}" ]]; then
  refseq_args+=(--refseq-sha256 "${REFSEQ_SHA256}")
fi
PYTHONPATH="$(dirname "${ROOT}")${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" -m mrna_editflow.data.reconstruction \
    --gencode-source "${GENCODE_SOURCE}" \
    --gencode-sha256 "${GENCODE_SHA256}" \
    "${refseq_args[@]}" \
    --frozen-root "${FROZEN_ROOT}" \
    --split-root "${SPLIT_ROOT}" \
    --seed "${SEED}"
