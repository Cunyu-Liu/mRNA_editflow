# mRNA-EditFlow 从 0 到 1 训练与测评命令文档

本文档记录在远端服务器上复现当前 mRNA-EditFlow 训练、评测、审计与报告刷新的命令顺序。

声明边界：以下命令只能复现 proxy/offline 训练和审计流程。它们本身不构成 full-length
de novo SOTA、wet-lab TE/stability 或真实外部 baseline 复现证据。

远端根目录：

```bash
export ROOT=/home/cunyuliu/mrna_editflow_goal/mrna_editflow
export PYTHON_BIN=/home/cunyuliu/miniconda3/envs/editflow/bin/python3.10
export PYTHONPATH=/home/cunyuliu/mrna_editflow_goal
cd "${ROOT}"
```

## 0. 环境与硬件审计

```bash
date -Is
hostname
uptime
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
"${PYTHON_BIN}" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
PY
```

## 1. 公共数据下载、清洗与 manifest 生成

当前完整可用的最大训练语料是 GENCODE human cleaned full-transcript records。
RefSeq 仍在排队构建；只有 raw、records、manifest、family/leakage audit 全部完成后，
才能把 RefSeq 纳入正式数据规模 claim。

### 1.1 GENCODE 数据下载与清洗

```bash
PYTHONPATH="$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.data.public_pipeline \
  --download \
  --dataset gencode_human_transcripts \
  --data-dir "${ROOT}/data/raw" \
  --out-dir "${ROOT}/data/processed" \
  --seed 20260714
```

该命令会完成：

- 下载 GENCODE public FASTA；
- 解析 full transcript 与 CDS 坐标；
- 标准化 `T -> U`；
- 清洗非法字符、非法 CDS、非 AUG 起始、非终止 stop、内部 stop、frame 错误；
- 按配置截断 UTR 上下文、丢弃过长 CDS；
- 输出 `data/processed/gencode_human_transcripts.records.jsonl`；
- 输出 `data/processed/gencode_human_transcripts.data_manifest.json`。

### 1.2 RefSeq 官方数据构建队列

```bash
nohup bash "${ROOT}/scripts/run_refseq_public_build.sh" \
  > "${ROOT}/logs/refseq_public_build.queue.log" 2>&1 &
echo $!
```

该脚本会等待 `MAX_LOADAVG=80`，然后下载和解析 NCBI RefSeq
`human.1.rna.gbff.gz`，生成 cleaned records 与 manifest。构建未完成前，不能声明
RefSeq-scale training。

### 1.3 family/leakage split 侧边审计队列

GENCODE family/leakage audit：

```bash
nohup bash "${ROOT}/scripts/run_gencode_family_leakage_audit.sh" \
  > "${ROOT}/logs/gencode_family_leakage_protocol.queue.log" 2>&1 &
echo $!
```

RefSeq family/leakage audit，等待 RefSeq records/manifest 出现后自动继续：

```bash
nohup bash "${ROOT}/scripts/run_refseq_family_leakage_audit.sh" \
  > "${ROOT}/logs/refseq_family_leakage_protocol.queue.log" 2>&1 &
echo $!
```

GENCODE sidecar readiness watcher：

```bash
nohup bash "${ROOT}/scripts/watch_gencode_family_readiness.sh" \
  > "${ROOT}/logs/gencode_family_readiness.watch.log" 2>&1 &
echo $!
```

## 2. 数据治理与 readiness 审计

```bash
PYTHONPATH="$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.eval.dataset_manifest_audit \
  --project-root "${ROOT}" \
  --out-json "${ROOT}/docs/dataset_manifest_audit.json" \
  --out-md "${ROOT}/docs/dataset_manifest_audit.md"

PYTHONPATH="$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.eval.downstream_data_acquisition_audit \
  --project-root "${ROOT}" \
  --out-json "${ROOT}/docs/downstream_data_acquisition_audit.json" \
  --out-md "${ROOT}/docs/downstream_data_acquisition_audit.md"

PYTHONPATH="$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.eval.build_data_scaleup_readiness \
  --project-root "${ROOT}" \
  --out-json "${ROOT}/docs/data_scaleup_readiness.json" \
  --out-md "${ROOT}/docs/data_scaleup_readiness.md"
```

下游真实 MPRA/TE 与 stability 数据获取队列：

```bash
nohup bash "${ROOT}/scripts/run_downstream_predictor_protocol.sh" \
  > "${ROOT}/logs/downstream_predictor_protocol.queue.log" 2>&1 &
echo $!
```

P3 readiness 通用 watcher：

```bash
nohup bash "${ROOT}/scripts/watch_p3_readiness.sh" \
  > "${ROOT}/logs/p3_readiness_watcher.log" 2>&1 &
echo $!
```

## 3. 启动当前最大 A100 Stage A 训练

这是当前“已验证最好架构开关 + 最大可用完整数据 + 最大参数量”的 Stage A 训练：

- 完整 GENCODE cleaned records；
- `region_film=true`；
- `codon_constraint=true`；
- `rope=true`；
- `aux_struct=true`；
- 参考 mRNA-LM full-length context：5'UTR `512`，3'UTR `1024`；
- A100 规模 head：`model_dim=768`、`num_layers=12`、`num_heads=12`；
- `100000` optimizer steps；
- `batch_size=1`、`grad_accum=32`；
- 保留 `MAX_LOADAVG=80` 与 A100 free-memory gate。

```bash
bash "${ROOT}/scripts/run_stage_a_a100_max_train.sh" --dry-run

nohup bash "${ROOT}/scripts/run_stage_a_a100_max_train.sh" \
  > "${ROOT}/logs/stage_a_full_a100_max_gencode_100k_seed0.queue.log" 2>&1 &
echo $!
```

监控：

```bash
tail -f "${ROOT}/benchmark/stage_a_full_a100_max_gencode_100k_seed0/progress.jsonl"
tail -f "${ROOT}/logs/stage_a_full_a100_max_gencode_100k_seed0.train.log"
PYTHONPATH="$(dirname "${ROOT}")" "${PYTHON_BIN}" - <<'PY'
import json
p="/home/cunyuliu/mrna_editflow_goal/mrna_editflow/benchmark/stage_a_full_a100_max_gencode_100k_seed0/status.json"
d=json.load(open(p))
print(json.dumps(d, indent=2)[:4000])
PY
```

## 4. Stage A 完成后的 proposal-ranking 审计

在 `stage_a_best.pt` 出现后执行：

```bash
export MAX_RUN=stage_a_full_a100_max_gencode_100k_seed0
export MAX_DIR="${ROOT}/ckpts/${MAX_RUN}"
export MAX_CKPT="${MAX_DIR}/stage_a_best.pt"

PYTHONPATH="$(dirname "${ROOT}")" CUDA_VISIBLE_DEVICES=auto "${PYTHON_BIN}" -m mrna_editflow.eval.proposal_ranking \
  --records-jsonl "${ROOT}/data/processed/gencode_human_transcripts.records.jsonl" \
  --checkpoint "${MAX_CKPT}" \
  --task-id T5 \
  --limit 1024 \
  --candidate-cap 0 \
  --top-k 64 \
  --device cuda \
  --out-json "${ROOT}/benchmark/proposal_ranking_t5_${MAX_RUN}_head1024.json" \
  --out-jsonl "${ROOT}/benchmark/proposal_ranking_t5_${MAX_RUN}_head1024.candidates.jsonl"
```

## 5. 训练 TE proposal ranker

```bash
PYTHONPATH="$(dirname "${ROOT}")" CUDA_VISIBLE_DEVICES=auto "${PYTHON_BIN}" -m mrna_editflow.train_proposal_ranker \
  --records-jsonl "${ROOT}/data/processed/gencode_human_transcripts.records.jsonl" \
  --teacher-jsonl "${ROOT}/benchmark/proposal_ranking_t5_${MAX_RUN}_head1024.candidates.jsonl" \
  --base-checkpoint "${MAX_CKPT}" \
  --save-dir "${ROOT}/ckpts/proposal_ranker_t5_${MAX_RUN}_head1024_teacher" \
  --profile-path "${ROOT}/logs/proposal_ranker_t5_${MAX_RUN}_head1024_teacher.profile.jsonl" \
  --steps 500 \
  --batch-records 4 \
  --max-pairs-per-record 32 \
  --lr 2e-5 \
  --pair-source-mode global \
  --device cuda \
  --seed 0
```

## 6. 下游 T1-T7 多 seed 测评命令

T5 是主优化 benchmark。T1-T4/T6/T7 是必要下游审计，分别覆盖合法性、分布保持、
novelty、多样性、CDS/protein identity、长度控制、motif/frame 控制。尽量保持同一
records、checkpoint、seeds、bootstrap 和硬约束指标。

通用变量：

```bash
export EVAL_RECORDS="${ROOT}/data/processed/gencode_human_transcripts.records.jsonl"
export EVAL_CKPT="${ROOT}/ckpts/proposal_ranker_t5_${MAX_RUN}_head1024_teacher/proposal_ranker_best.pt"
export EVAL_SEEDS="0 1 2 3 4 5 6 7 8 9"
export EVAL_LIMIT=1024
export EVAL_TOP_K=64
export EVAL_BOOTSTRAP=1000
```

### 6.1 T1 合法性 / Oracle TE

```bash
PYTHONPATH="$(dirname "${ROOT}")" CUDA_VISIBLE_DEVICES=auto "${PYTHON_BIN}" -m mrna_editflow.eval.run_multiseed_benchmark \
  --records-jsonl "${EVAL_RECORDS}" \
  --checkpoint "${EVAL_CKPT}" \
  --task-id T1 \
  --edit-budget 3 \
  --proposal-top-k "${EVAL_TOP_K}" \
  --proposal-temperature 1.0 \
  --limit "${EVAL_LIMIT}" \
  --device cuda \
  --seeds ${EVAL_SEEDS} \
  --n-bootstrap "${EVAL_BOOTSTRAP}" \
  --max-novelty-sources 0 \
  --resume \
  --out-dir "${ROOT}/benchmark/multiseed_t1_public_head1024_${MAX_RUN}_ranker_top64"
```

### 6.2 T2 分布保持

```bash
PYTHONPATH="$(dirname "${ROOT}")" CUDA_VISIBLE_DEVICES=auto "${PYTHON_BIN}" -m mrna_editflow.eval.run_multiseed_benchmark \
  --records-jsonl "${EVAL_RECORDS}" \
  --checkpoint "${EVAL_CKPT}" \
  --task-id T2 \
  --edit-budget 3 \
  --proposal-top-k "${EVAL_TOP_K}" \
  --proposal-temperature 1.0 \
  --limit "${EVAL_LIMIT}" \
  --device cuda \
  --seeds ${EVAL_SEEDS} \
  --n-bootstrap "${EVAL_BOOTSTRAP}" \
  --max-novelty-sources 0 \
  --resume \
  --out-dir "${ROOT}/benchmark/multiseed_t2_public_head1024_${MAX_RUN}_ranker_top64"
```

### 6.3 T3 多样性 / Novelty

```bash
PYTHONPATH="$(dirname "${ROOT}")" CUDA_VISIBLE_DEVICES=auto "${PYTHON_BIN}" -m mrna_editflow.eval.run_multiseed_benchmark \
  --records-jsonl "${EVAL_RECORDS}" \
  --checkpoint "${EVAL_CKPT}" \
  --task-id T3 \
  --edit-budget 3 \
  --proposal-top-k "${EVAL_TOP_K}" \
  --proposal-temperature 1.0 \
  --limit "${EVAL_LIMIT}" \
  --device cuda \
  --seeds ${EVAL_SEEDS} \
  --n-bootstrap "${EVAL_BOOTSTRAP}" \
  --max-novelty-sources 0 \
  --resume \
  --out-dir "${ROOT}/benchmark/multiseed_t3_public_head1024_${MAX_RUN}_ranker_top64"
```

### 6.4 T4 Protein identity / CDS 同义编辑

```bash
PYTHONPATH="$(dirname "${ROOT}")" CUDA_VISIBLE_DEVICES=auto "${PYTHON_BIN}" -m mrna_editflow.eval.run_multiseed_benchmark \
  --records-jsonl "${EVAL_RECORDS}" \
  --checkpoint "${EVAL_CKPT}" \
  --task-id T4 \
  --edit-budget 3 \
  --proposal-top-k "${EVAL_TOP_K}" \
  --proposal-temperature 1.0 \
  --limit "${EVAL_LIMIT}" \
  --device cuda \
  --seeds ${EVAL_SEEDS} \
  --n-bootstrap "${EVAL_BOOTSTRAP}" \
  --max-novelty-sources 0 \
  --resume \
  --out-dir "${ROOT}/benchmark/multiseed_t4_public_head1024_${MAX_RUN}_ranker_top64"
```

补充 protein-conditioned CDS constructive benchmark。该命令从同一批 source CDS
翻译出目标 protein，再用同义 codon lattice 重新设计 CDS，并记录 native CDS 对照。
这是对齐 codonGPT/Prot2RNA 口径的 CDS-only/protein-conditioned proxy evidence；
不能当作外部模型复现。

```bash
PYTHONPATH="$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.baselines.protein_conditioned_cds \
  --records-jsonl "${ROOT}/benchmark/multiseed_t5_public_head256_hardneg_v2_top64/sources.jsonl" \
  --reference-records-jsonl "${ROOT}/benchmark/multiseed_t5_public_head1024_sources.jsonl" \
  --use-native-baseline \
  --limit 256 \
  --out-jsonl "${ROOT}/benchmark/protein_conditioned_cds_head256.jsonl" \
  --out-json "${ROOT}/benchmark/protein_conditioned_cds_head256.summary.json"

PYTHONPATH="$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.eval.audit_protein_conditioned_codon_metrics \
  --project-root "${ROOT}" \
  --jsonl "${ROOT}/benchmark/protein_conditioned_cds_head256.jsonl" \
  --top-n 20 \
  --out-json "${ROOT}/benchmark/protein_conditioned_codon_metrics_head256.json" \
  --out-md "${ROOT}/benchmark/protein_conditioned_codon_metrics_head256.md"
```

上述 codon-level audit 会输出 native codon recovery、seed/native codon edit
fraction、同义/非同义替换比例、GC3、codon usage KL、codon-pair KL，并把
protein/start/stop/frame exact-1 作为 ready gate。当前报告仍是 proxy/offline，
不能替代 codonGPT/Prot2RNA official executable baseline。

如果要对某个 slice 一次性跑完整 protein-conditioned T4/CDS 审计链路（CDS
codon-lattice DP、protein→CDS、codon-level metrics、CAI-GC sweep、T4 汇总），
使用下面的队列脚本。`SLICE=head1024` 会输出
`codon_lattice_dp_head1024`、`protein_conditioned_cds_head1024`、
`protein_conditioned_codon_metrics_head1024`、
`protein_conditioned_cds_gc_sweep_head1024` 和
`t4_protein_identity_cai_gc_report_head1024`。该脚本默认不等待 load average。

```bash
nohup env SLICE=head1024 LIMIT=1024 TOP_K=64 \
  CODON_DP_MAX_CODON_CHANGES=3 PROTEIN_MAX_CODON_CHANGES= MAX_LOADAVG=0 \
  bash "${ROOT}/scripts/run_protein_conditioned_t4_slice.sh" \
  > "${ROOT}/logs/protein_conditioned_t4_head1024.queue.log" 2>&1 &
```

### 6.5 T5 Edit budget / TE 优化

```bash
PYTHONPATH="$(dirname "${ROOT}")" CUDA_VISIBLE_DEVICES=auto "${PYTHON_BIN}" -m mrna_editflow.eval.run_multiseed_benchmark \
  --records-jsonl "${EVAL_RECORDS}" \
  --checkpoint "${EVAL_CKPT}" \
  --task-id T5 \
  --edit-budget 3 \
  --proposal-top-k "${EVAL_TOP_K}" \
  --proposal-temperature 1.0 \
  --limit "${EVAL_LIMIT}" \
  --device cuda \
  --seeds ${EVAL_SEEDS} \
  --n-bootstrap "${EVAL_BOOTSTRAP}" \
  --max-novelty-sources 0 \
  --resume \
  --out-dir "${ROOT}/benchmark/multiseed_t5_public_head1024_${MAX_RUN}_ranker_top64"
```

### 6.6 T6 长度控制曲线

运行全部 target length delta。正向 lengthening 可能降低 TE，只能写成长度控制证据，
不能写成 TE 提升证据。

```bash
for DELTA in -30 -15 0 15 30; do
  if [[ "${DELTA}" == -* ]]; then LABEL="neg${DELTA#-}"; else LABEL="pos${DELTA}"; fi
  PYTHONPATH="$(dirname "${ROOT}")" CUDA_VISIBLE_DEVICES=auto "${PYTHON_BIN}" -m mrna_editflow.eval.run_multiseed_benchmark \
    --records-jsonl "${EVAL_RECORDS}" \
    --checkpoint "${EVAL_CKPT}" \
    --task-id T6 \
    --edit-budget 30 \
    --target-length-delta "${DELTA}" \
    --proposal-top-k "${EVAL_TOP_K}" \
    --proposal-temperature 1.0 \
    --limit "${EVAL_LIMIT}" \
    --device cuda \
    --seeds ${EVAL_SEEDS} \
    --n-bootstrap "${EVAL_BOOTSTRAP}" \
    --max-novelty-sources 0 \
    --resume \
    --out-dir "${ROOT}/benchmark/multiseed_t6_public_head1024_${MAX_RUN}_len_${LABEL}_top64"
done
```

### 6.7 T7 Motif / Frame 控制

先使用保守 UTR motif。CDS motif 插入/切除不允许，因为会破坏 frame/protein-safe
约束。T7 constructive motif evaluation 不是通用 `run_multiseed_benchmark` 模式；
应先用 `sample.py` 生成 motif-edit candidates，再用离线 T7 evaluator 统计。

```bash
for ACTION in insert excise; do
  PYTHONPATH="$(dirname "${ROOT}")" CUDA_VISIBLE_DEVICES=auto "${PYTHON_BIN}" -m mrna_editflow.sample \
    --records-jsonl "${EVAL_RECORDS}" \
    --checkpoint "${EVAL_CKPT}" \
    --task-id T7 \
    --limit "${EVAL_LIMIT}" \
    --device cuda \
    --steps 8 \
    --edit-budget 8 \
    --motif GCCACC \
    --motif-action "${ACTION}" \
    --motif-region 5utr \
    --output-jsonl "${ROOT}/benchmark/t7_motif_edit_${MAX_RUN}_${ACTION}.candidates.jsonl"

  PYTHONPATH="$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.eval.run_eval \
    --candidates "${ROOT}/benchmark/t7_motif_edit_${MAX_RUN}_${ACTION}.candidates.jsonl" \
    --sources "${EVAL_RECORDS}" \
    --task-id T7 \
    --out-dir "${ROOT}/benchmark/t7_motif_edit_${MAX_RUN}_${ACTION}_eval" \
    --seeds ${EVAL_SEEDS} \
    --n-bootstrap "${EVAL_BOOTSTRAP}" \
    --max-edits 8 \
    --max-novelty-sources 0
done
```

可选：对模型生成输出做 decoded frame/uAUG proxy audit：

```bash
PYTHONPATH="$(dirname "${ROOT}")" CUDA_VISIBLE_DEVICES=auto "${PYTHON_BIN}" -m mrna_editflow.eval.run_multiseed_benchmark \
    --records-jsonl "${EVAL_RECORDS}" \
    --checkpoint "${EVAL_CKPT}" \
    --task-id T5 \
    --edit-budget 8 \
    --proposal-top-k "${EVAL_TOP_K}" \
    --proposal-temperature 1.0 \
    --limit "${EVAL_LIMIT}" \
    --device cuda \
    --seeds ${EVAL_SEEDS} \
    --n-bootstrap "${EVAL_BOOTSTRAP}" \
    --max-novelty-sources 0 \
    --resume \
    --out-dir "${ROOT}/benchmark/multiseed_t7_decoded_proxy_head1024_${MAX_RUN}_top64"
```

### 6.8 汇总 T1-T7 报告

上述 runs 完成后刷新专用报告生成器。注意：部分 builder 当前指向 canonical artifact
names；如果要评估新的 `MAX_RUN`，需要先检查每个 builder 的 CLI contract，再复制或传入
新路径。

```bash
PYTHONPATH="$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.eval.summarize_t1_runtime \
  --project-root "${ROOT}" \
  --out-json "${ROOT}/benchmark/t1_runtime_report_head256_head1024.json" \
  --out-md "${ROOT}/benchmark/t1_runtime_report_head256_head1024.md"

PYTHONPATH="$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.eval.summarize_t2_t3_distribution_novelty \
  --project-root "${ROOT}" \
  --out-json "${ROOT}/benchmark/t2_t3_distribution_novelty_report_head256_head1024.json" \
  --out-md "${ROOT}/benchmark/t2_t3_distribution_novelty_report_head256_head1024.md"

PYTHONPATH="$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.eval.summarize_t4_protein_identity_cai_gc \
  --project-root "${ROOT}" \
  --out-json "${ROOT}/benchmark/t4_protein_identity_cai_gc_report_head256.json" \
  --out-md "${ROOT}/benchmark/t4_protein_identity_cai_gc_report_head256.md"

PYTHONPATH="$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.eval.summarize_t6_length_curve \
  --project-root "${ROOT}" \
  --out-json "${ROOT}/benchmark/t6_length_curve_report_head256_head1024.json" \
  --out-md "${ROOT}/benchmark/t6_length_curve_report_head256_head1024.md"
```

## 7. 与当前强 baseline 做 paired compare

```bash
PYTHONPATH="$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.eval.compare_benchmarks \
  --baseline "te_only_top64=${ROOT}/benchmark/multiseed_t5_public_head1024_mo_te_only_top64/multiseed_summary.json" \
  --run "a100_max_ranker_top64=${ROOT}/benchmark/multiseed_t5_public_head1024_${MAX_RUN}_ranker_top64/multiseed_summary.json" \
  --metrics delta_oracle_te_vs_source mean_oracle_te mean_protein_identity within_budget_fraction reading_frame_intact_fraction \
  --out-json "${ROOT}/benchmark/compare_${MAX_RUN}_ranker_vs_te_only_head1024.json" \
  --out-md "${ROOT}/benchmark/compare_${MAX_RUN}_ranker_vs_te_only_head1024.md" \
  --n-bootstrap 1000 \
  --n-permutations 2000 \
  --require-default-matching-config

PYTHONPATH="$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.eval.compare_benchmarks \
  --baseline "hardneg_v2_top64=${ROOT}/benchmark/multiseed_t5_public_head1024_hardneg_v2_top64/multiseed_summary.json" \
  --run "a100_max_ranker_top64=${ROOT}/benchmark/multiseed_t5_public_head1024_${MAX_RUN}_ranker_top64/multiseed_summary.json" \
  --metrics delta_oracle_te_vs_source mean_oracle_te mean_protein_identity within_budget_fraction reading_frame_intact_fraction \
  --out-json "${ROOT}/benchmark/compare_${MAX_RUN}_ranker_vs_hardneg_v2_head1024.json" \
  --out-md "${ROOT}/benchmark/compare_${MAX_RUN}_ranker_vs_hardneg_v2_head1024.md" \
  --n-bootstrap 1000 \
  --n-permutations 2000 \
  --require-default-matching-config
```

## 8. 刷新 T1-T7 / SOTA / 论文报告

```bash
PYTHONPATH="$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.eval.audit_sota_readiness \
  --project-root "${ROOT}" \
  --out-json "${ROOT}/docs/sota_readiness_audit_head256.json" \
  --out-md "${ROOT}/docs/sota_readiness_audit_head256.md"

PYTHONPATH="$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.eval.sota_gap_report \
  --project-root "${ROOT}" \
  --out-json "${ROOT}/docs/sota_gap_report.json" \
  --out-md "${ROOT}/docs/sota_gap_report.md"

PYTHON_BIN="${PYTHON_BIN}" bash "${ROOT}/scripts/check_remote_sota_status.sh"
```

## 9. 外部 LinearDesign / UTRGAN / UTailoR head1024 复现与验收

LinearDesign 正式执行：

```bash
ROOT="${ROOT}" PYTHON_BIN="${PYTHON_BIN}" WORKERS=8 TIMEOUT_S=900 \
bash "${ROOT}/scripts/run_external_lineardesign_head1024.sh"
```

UTRGAN 10-step budgeted 执行、UTR-only constrained ceiling 和 T5 报告刷新：

```bash
ROOT="${ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
UTRGAN_GPU=2 UTRGAN_STEPS=10 LOCAL_WORKERS=16 \
bash "${ROOT}/scripts/run_external_utrgan_head1024.sh"
```

UTailoR 官方 SavedModel 严格输入域执行：

```bash
ROOT="${ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
bash "${ROOT}/scripts/run_external_utailor_head1024.sh"
```

该命令只接收官方支持的 25--100 nt 5'UTR。当前 head1024 输入中有 315 条满足
条件，其余 709 条作为 ineligible 证据保存，不做截断、填充或静默丢弃。

MEF UTR-specific adapter 的 5'UTR-only 10-seed head1024 执行：

```bash
ROOT="${ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
SHARDED=1 SHARD_GPUS="0 2" \
bash "${ROOT}/scripts/run_mef_utr5only_head1024.sh"
```

该命令在 candidate enumeration 阶段设置 `--editable-regions utr5`，不会枚举
CDS 或 3'UTR 位置。seeds `0..4` 与 `5..9` 分别在两个 GPU shard 上运行，随后由
`merge_multiseed_shards` 合并；不要把旧的 unrestricted T5 candidate 目录作为
该运行的 resume 输入。

在与 UTailoR 完全相同的 315 条严格子集上，以 hard edit budget 5 运行最佳
pure UTR teacher：

```bash
ROOT="${ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
SHARD_GPUS="0 2" \
bash "${ROOT}/scripts/run_mef_utailor_subset_budget5.sh"
```

该 runner 会先生成并校验
`benchmark/utailor_strict_25_100_sources.{jsonl,summary.json}`，再执行 10 seeds。
budget 5 只是让 MEF 的平均编辑数接近 UTailoR，不代表严格预算匹配，因为官方
UTailoR 没有逐条 hard cap。

EnsembleDesign 的可恢复 budgeted head1024 执行：

```bash
ROOT="${ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
WORKERS=8 BEAM_SIZE=100 NUM_ITERS=3 NUM_RUNS=1 RESUME=1 \
bash "${ROOT}/scripts/run_external_ensembledesign_head1024.sh"
```

该预算不是论文默认配置。论文/官方默认需要
`BEAM_SIZE=200 NUM_ITERS=30 NUM_RUNS=20`；只有这些参数才允许
`protocol_fidelity_sufficient_for_sota_reproduction=True`。当前实测首条 326-aa
蛋白在 `100/3/1` 下耗时 `229.68s`，在 `200/30/1` 下超过 10 分钟，故完整
paper-default head1024 属于多日级任务。

codonGPT 官方 Hugging Face checkpoint 的安装与 head1024 执行：

```bash
ROOT="${ROOT}" bash "${ROOT}/scripts/setup_external_codongpt.sh"

ROOT="${ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
CODONGPT_GPU=6 BATCH_SIZE=64 LIMIT=1024 RESUME=1 \
bash "${ROOT}/scripts/run_external_codongpt_head1024.sh"
```

该流程固定 `naniltx/codonGPT` revision
`ee7017c4bdd285206b87be2e65a28272ff4ac88e`，权重 SHA-256 为
`df41546883e31ba13598d5ae74044666502a89ba34630d6f6c32943836e6f454`。
采样参数遵循模型卡：BOS 起始、逐 codon 同义 mask、temperature `1.0`、
top-k `50`、top-p `0.9`。公开 HF artifact 是预训练 checkpoint，不包含论文
HLA-A/ACTB 等任务的 RL policy 与 reward-training 产物；因此该结果只能称为
**official pretrained checkpoint constrained generation**，不能称论文 RL 复现。

codonGPT seeds `1..9` 双 GPU 补跑并与 canonical seed0 聚合：

```bash
ROOT="${ROOT}" PYTHON_BIN="${PYTHON_BIN}" \
CODONGPT_GPUS="6 7" BATCH_SIZE=64 \
bash "${ROOT}/scripts/run_external_codongpt_multiseed_head1024.sh"
```

UTRGAN 论文默认 10000-step 独立运行（不覆盖 10-step canonical artifact）：

```bash
ROOT="${ROOT}" PYTHON_BIN="${PYTHON_BIN}" UTRGAN_GPU=4 \
bash "${ROOT}/scripts/run_external_utrgan_paper10000.sh"
```

当前 paper-default 产物为 `1024/1024`、zero failures、elapsed `2798.396s`、
proxy TE delta `+0.040681`、mean UTR edit distance `72.48535`、normalized
edit distance `0.63032`、mean length delta `+8.08887`，固定
CDS/3'UTR/protein fractions 均为 `1.0`。相对独立保存的 10-step 结果，
TE delta 增加 `+0.011960`；由于两批输出都不是逐 source 条件生成，只能报告
描述性差异，不能报告 paired p。

只刷新验收报告而不重跑外部工具：

```bash
PYTHONPATH="$(dirname "${ROOT}")" "${PYTHON_BIN}" \
  -m mrna_editflow.eval.audit_external_sota_real_runs \
  --project-root "${ROOT}" \
  --out-json "${ROOT}/docs/external_sota_real_run_audit.json" \
  --out-md "${ROOT}/docs/external_sota_real_run_audit.md"

PYTHONPATH="$(dirname "${ROOT}")" "${PYTHON_BIN}" \
  -m mrna_editflow.eval.build_t4_external_cds_comparison \
  --project-root "${ROOT}" \
  --out-json "${ROOT}/docs/t4_external_cds_baseline_comparison.json" \
  --out-md "${ROOT}/docs/t4_external_cds_baseline_comparison.md"

PYTHONPATH="$(dirname "${ROOT}")" "${PYTHON_BIN}" \
  -m mrna_editflow.eval.build_t5_external_utr_comparison \
  --project-root "${ROOT}" \
  --out-json "${ROOT}/docs/t5_external_utr_baseline_comparison.json" \
  --out-md "${ROOT}/docs/t5_external_utr_baseline_comparison.md"
```

验收边界：

- LinearDesign 当前是 single-lambda measured run，不是完整 lambda sweep。
- codonGPT GitHub 链接当前匿名访问为 404，但官方 HF checkpoint 可下载；不能再写
  “无公开模型”，也不能把 HF pretrained checkpoint 冒充未公开的 RL policy。
  当前 official pretrained head1024 为 `1024/1024`、zero failures、valid CDS/
  protein exact-1=`1.0`、`0.18175s/sequence`、codon accuracy `0.41952`、
  CAI `0.69824`、GC `0.56005`、GC3 `0.64961`。10-seed 聚合为 codon
  accuracy `0.42030`、CAI `0.70449`、GC `0.56475`、GC3 `0.66242`；
  CAI delta vs native `+0.01079`，95% CI `[+0.00912,+0.01239]`，
  sign-flip `p=0.00450`。
- Prot2RNA 截至 2026-07-16 仍未找到官方 executable/checkpoint；其 ICLR 2026
  投稿为 reject，且评审明确指出 CAI/相似度不能证明真实表达提升。
- UTRGAN 的 10-step budgeted 与 paper-default 10000-step 均已独立保存；后者使
  UTRGAN 的 protocol-fidelity gate 通过，但不能把 proxy TE 当作湿实验 TE，
  也不能据此打开 MEF superiority gate。
- UTailoR 只在官方 25--100 nt 输入域的 315 条子集上测量；不能外推到 1024 条全集。
- MEF budget-5 严格子集结果为 TE delta `+0.007774`，而 UTailoR 为
  `+0.036105`；MEF - UTailoR 为 `-0.028332`，95% CI
  `[-0.029271,-0.027301]`，paired `p=0.00450`。平均编辑数分别为
  `4.79619` 和 `4.41905`，因此当前主要差距是单位编辑效率，而非原 budget 3。
- UTR local search 直接优化共享 proxy oracle，只能作为 constrained ceiling。
- UTRGAN 为 de novo batch generation，按输入顺序配对不构成 source-conditioned
  paired inference；因此 paired p 必须记为 `NA`。

## 10. 当前分布统计覆盖范围与缺口

现有 T2/T3 评估已经覆盖：

- 生成序列与 source/training slice 的 `kmer_js`；
- `codon_usage_kl`；
- candidate/source 的 mean GC；
- candidate/source 的 mean length；
- GC quantile distance；
- length quantile distance；
- combined GC/length distribution distance；
- embedding Frechet proxy；
- novelty、exact source match、unique fraction、pairwise diversity。

现有报告：

```bash
cat "${ROOT}/benchmark/t2_t3_distribution_novelty_report_head256_head1024.md"
```

当前缺口：

- 没有显式输出 A/C/G/U 四种碱基的分布表；
- 没有 region-wise base composition，即 5'UTR/CDS/3'UTR 分区碱基分布；
- 没有自动生成长度直方图、GC 分布图、碱基组成柱状图、k-mer/codon-pair 分布图；
- 没有把上述图表自动纳入 paper figure/table。

新增的 `multi_scale_sequence_spectrum_audit` 用于补齐这些统计和图表：

```bash
PYTHONPATH="$(dirname "${ROOT}")" "${PYTHON_BIN}" -m mrna_editflow.eval.multi_scale_sequence_spectrum_audit \
  --candidate-glob "${ROOT}/benchmark/multiseed_t5_public_head1024_${MAX_RUN}_ranker_top64/seed_*/candidates.jsonl" \
  --sources "${ROOT}/benchmark/multiseed_t5_public_head1024_${MAX_RUN}_ranker_top64/sources.jsonl" \
  --out-json "${ROOT}/benchmark/multi_scale_sequence_spectrum_${MAX_RUN}.json" \
  --out-md "${ROOT}/benchmark/multi_scale_sequence_spectrum_${MAX_RUN}.md" \
  --out-fig-dir "${ROOT}/benchmark/multi_scale_sequence_spectrum_${MAX_RUN}_figures"
```

如果某个 multiseed 目录没有保存 `candidates.jsonl`，必须先重新运行对应 benchmark
并开启候选保存，不能只从 summary 反推图表。

## 11. 必须使用的汇报措辞

在更强证据出现前，使用下面的表述：

```text
mRNA-EditFlow 当前是一个基于已有转录本的、约束安全的 full-length local
optimization / reranking 系统。A100 max run 只是训练基础设施证据；只有完成
proposal-ranking、T1-T7、paired comparison、external baseline 和真实数据审计后，
才能进入更强 claim。
```

不能声明：

- full-length de novo SOTA；
- wet-lab TE/stability validation；
- 真实 external SOTA reproduction；
- true data/model/step scale law；
- head1024 相对强 `te_only` control 的严格显著提升，除非 paired `p < 0.05`。
