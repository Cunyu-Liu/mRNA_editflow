#!/usr/bin/env bash
# Install the pinned codonGPT runtime and public Hugging Face snapshot.
set -euo pipefail

ROOT="${ROOT:-/home/cunyuliu/mrna_editflow_goal/mrna_editflow}"
BASE_PYTHON="${BASE_PYTHON:-/home/cunyuliu/miniconda3/envs/pc_cng_gpu/bin/python3.10}"
ENV_DIR="${CODONGPT_ENV_DIR:-${ROOT}/external_tools/envs/codongpt}"
PYTHON_BIN="${ENV_DIR}/bin/python"
MODEL_DIR="${CODONGPT_MODEL_DIR:-${ROOT}/external_tools/codonGPT_hf_ee7017c4}"
HF_REPO="${CODONGPT_HF_REPO:-naniltx/codonGPT}"
HF_REVISION="${CODONGPT_HF_REVISION:-ee7017c4bdd285206b87be2e65a28272ff4ac88e}"
EXPECTED_WEIGHTS_SHA256="${CODONGPT_WEIGHTS_SHA256:-df41546883e31ba13598d5ae74044666502a89ba34630d6f6c32943836e6f454}"

print_plan() {
  echo "SETUP EXTERNAL CODONGPT"
  echo "ROOT=${ROOT}"
  echo "ENV_DIR=${ENV_DIR}"
  echo "MODEL_DIR=${MODEL_DIR}"
  echo "HF_REPO=${HF_REPO}"
  echo "HF_REVISION=${HF_REVISION}"
  echo "EXPECTED_WEIGHTS_SHA256=${EXPECTED_WEIGHTS_SHA256}"
  echo "BASE_PYTHON=${BASE_PYTHON}"
  echo "RUNTIME_REQUIREMENT=torch>=2.6 with transformers and huggingface_hub"
}

if [[ "${1:-}" == "--dry-run" ]]; then print_plan; exit 0; fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "创建 codonGPT 隔离环境：${ENV_DIR}"
  "${BASE_PYTHON}" -m venv --system-site-packages "${ENV_DIR}"
fi

mkdir -p "${MODEL_DIR}"
MODEL_DIR="${MODEL_DIR}" HF_REPO="${HF_REPO}" BASE_PYTHON="${BASE_PYTHON}" \
EXPECTED_WEIGHTS_SHA256="${EXPECTED_WEIGHTS_SHA256}" \
HF_REVISION="${HF_REVISION}" "${PYTHON_BIN}" - <<'PY'
import hashlib
import json
import os
import platform

from huggingface_hub import snapshot_download

model_dir = os.environ["MODEL_DIR"]
repo = os.environ["HF_REPO"]
revision = os.environ["HF_REVISION"]
required = [
    "README.md",
    "config.json",
    "generation_config.json",
    "pytorch_model.bin",
    "synonymous_logit_processor.py",
    "tokenizer.py",
]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

weights_path = os.path.join(model_dir, "pytorch_model.bin")
offline_snapshot_ready = bool(
    all(os.path.isfile(os.path.join(model_dir, name)) for name in required)
    and sha256(weights_path) == os.environ["EXPECTED_WEIGHTS_SHA256"]
)
if not offline_snapshot_ready:
    snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
        allow_patterns=required,
    )

files = []
for name in required:
    path = os.path.join(model_dir, name)
    if not os.path.isfile(path):
        raise SystemExit(f"missing required model file: {path}")
    files.append(
        {
            "path": name,
            "size_bytes": os.path.getsize(path),
            "sha256": sha256(path),
        }
    )
manifest = {
    "artifact_kind": "codongpt_official_hf_snapshot",
    "hf_repo": repo,
    "hf_revision": revision,
    "files": files,
    "license": "free_for_research_use_model_card",
    "license_source": "README.md model card; no SPDX license file",
    "redistribution_rights_assumed": False,
    "python": platform.python_version(),
    "runtime_base_python": os.environ.get("BASE_PYTHON"),
}
path = os.path.join(model_dir, "model_manifest.json")
with open(path, "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
print(
    json.dumps(
        {
            "model_dir": model_dir,
            "hf_revision": revision,
            "weights_sha256": next(
                row["sha256"]
                for row in files
                if row["path"] == "pytorch_model.bin"
            ),
        },
        sort_keys=True,
    )
)
PY

"${PYTHON_BIN}" - <<'PY'
import torch
import transformers
import huggingface_hub
print(
    "codonGPT 环境就绪："
    f"torch={torch.__version__}, "
    f"transformers={transformers.__version__}, "
    f"huggingface_hub={huggingface_hub.__version__}"
)
if tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2]) < (2, 6):
    raise SystemExit("torch must be >=2.6 for safe checkpoint loading")
PY
