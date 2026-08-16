#!/usr/bin/env bash
set -euo pipefail

asset_root="${1:?asset output directory required}"
mkdir -p "$asset_root/optimus5prime" "$asset_root/framepool" "$asset_root/rnafm"

optimus_revision="d53df410c7fb3fcd4bc4541bd7e8c6dc52b66fbe"
framepool_revision="c575f9cdca0cac1ffa88eb18e4435fdfbc674b08"
multimolecule_rnafm_revision="7d6e73ad3b48e042b378f9a788a56ccb4d573a27"

test -s "$asset_root/optimus5prime/main_MRL_model.hdf5" || curl -fL --retry 3 --retry-delay 2 \
  "https://raw.githubusercontent.com/pjsample/human_5utr_modeling/$optimus_revision/modeling/saved_models/main_MRL_model.hdf5" \
  -o "$asset_root/optimus5prime/main_MRL_model.hdf5"
test -s "$asset_root/optimus5prime/training_MRL_CNN.ipynb" || curl -fL --retry 3 --retry-delay 2 \
  "https://raw.githubusercontent.com/pjsample/human_5utr_modeling/$optimus_revision/modeling/training_MRL_CNN.ipynb" \
  -o "$asset_root/optimus5prime/training_MRL_CNN.ipynb"
test -s "$asset_root/optimus5prime/LICENSE" || curl -fL --retry 3 --retry-delay 2 \
  "https://raw.githubusercontent.com/pjsample/human_5utr_modeling/$optimus_revision/LICENSE" \
  -o "$asset_root/optimus5prime/LICENSE"

test -s "$asset_root/framepool/Framepool_combined_residual.h5" || curl -fL --retry 3 --retry-delay 2 \
  "https://raw.githubusercontent.com/Karollus/5UTR/$framepool_revision/kipoi/5UTR_Model/model/Framepool_combined_residual.h5" \
  -o "$asset_root/framepool/Framepool_combined_residual.h5"
test -s "$asset_root/framepool/model.py" || curl -fL --retry 3 --retry-delay 2 \
  "https://raw.githubusercontent.com/Karollus/5UTR/$framepool_revision/Modelling/model.py" \
  -o "$asset_root/framepool/model.py"
test -s "$asset_root/framepool/kipoi_model.py" || curl -fL --retry 3 --retry-delay 2 \
  "https://raw.githubusercontent.com/Karollus/5UTR/$framepool_revision/kipoi/5UTR_Model/model.py" \
  -o "$asset_root/framepool/kipoi_model.py"
test -s "$asset_root/framepool/LICENSE" || curl -fL --retry 3 --retry-delay 2 \
  "https://raw.githubusercontent.com/Karollus/5UTR/$framepool_revision/LICENSE" \
  -o "$asset_root/framepool/LICENSE"

for filename in config.json model.safetensors tokenizer_config.json vocab.txt license.md license-faq.md README.md; do
  test -s "$asset_root/rnafm/$filename" || curl -fL --retry 3 --retry-delay 2 \
    "https://huggingface.co/multimolecule/rnafm/resolve/$multimolecule_rnafm_revision/$filename" \
    -o "$asset_root/rnafm/$filename"
done
