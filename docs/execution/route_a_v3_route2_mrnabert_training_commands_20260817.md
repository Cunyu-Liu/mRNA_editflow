# Route 2 mRNABERT 主模型：参数、训练命令与实验记录

## 1. 当前模型与数据

- 主冻结编码器：`YYLY66/mRNABERT`，固定版本 `a1eb7df25804d23f08646e1cb996b234d7208a40`。
- RNA-FM 仅保留为历史对照，不再是新的主训练配置。
- 冻结 mRNABERT：113,389,056 参数；Huber/固定方差 critic：9,342,914 个可训练参数；总有效规模 122,731,970。
- 可学习方差 critic：9,343,299 个可训练参数；总有效规模 122,732,355。
- Development pool：126,165 条。HPO 只使用 TRAIN/VALIDATION；Development TEST 只在冻结模型选择后使用；Evaluation pool 不进入训练、调参或模型选择。

## 2. 参数在哪里修改

| 参数类别 | 文件 | 主要字段/位置 |
|---|---|---|
| mRNABERT 模型版本、GPU、分块、特征输出 | `configs/route_a_v3_route2_mrnabert_full_development_feature_cache_v1.json` | `mrnabert_model_path`、`physical_gpu_index`、`maximum_chunk_nucleotides`、`maximum_sequences_per_batch`、`batch_token_budget` |
| critic 架构 | `core/route2_delta_predictor.py` | `Route2PretrainedEditCenteredDeltaPredictor`；预训练全局背景、edit-centered pooling、差分表征、反对称输出 |
| Huber 主训练 | `configs/route_a_v3_route2_mrnabert_edit_max_mean_only_gpu6_v1.json` | `hidden_dim=384`、`depth=10`、`batch_size=16`、`learning_rate=1e-4`、`weight_decay=1e-4`、`epochs=100` |
| 固定方差对照 | `configs/route_a_v3_route2_mrnabert_edit_max_fixed_variance_gpu6_v1.json` | 同结构，`loss_kind=fixed_variance_gaussian_nll`，等待 GPU 5 空闲后运行 |
| 可学习方差对照 | `configs/route_a_v3_route2_mrnabert_edit_max_learned_variance_gpu6_v1.json` | 同结构，`loss_kind=learned_variance_gaussian_nll`，等待 GPU 3 空闲后运行 |
| 冻结 TEST | `configs/route_a_v3_route2_mrnabert_edit_max_mean_only_frozen_test_gpu6_v1.json` | `result_stage=FROZEN_DEVELOPMENT_TEST` |
| 全部 Development 最终拟合 | `configs/route_a_v3_route2_mrnabert_edit_max_mean_only_all126165_gpu6_v1.json` | `result_stage=FINAL_ALL_DEVELOPMENT_REFIT` |

换 GPU 时同时修改：

```json
"device": "cuda:4",
"physical_gpu_index": 4
```

项目直接使用物理 GPU 编号，不使用 `CUDA_VISIBLE_DEVICES` 重映射。

## 3. 每次训练的自动记录

训练入口会在启动、完成和失败时自动更新：

- 中央 Excel 可读 CSV：`/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiment_tracking/route2_training_attempts.csv`
- 每次运行的详细记录：`<output_directory>/training_attempt.json`
- 完整配置：`<output_directory>/training_config.json`
- 每轮指标：`<output_directory>/metrics.jsonl`
- 最终结果：`<output_directory>/training_summary.json`
- 失败证据：`<output_directory>.failed.json`

中央表一行对应一次训练尝试，记录：数据集和数据路径、TRAIN/VALIDATION/TEST 数量、模型与预训练模型、可训练/冻结/总参数量、loss、epoch、batch size、学习率、权重衰减、seed、GPU、更新次数、选择 epoch、验证/测试指标、显存、耗时、运行目录和失败原因。不会记录序列或逐行预测。

对在自动记录功能上线前已经启动的运行，可补登记：

```bash
$PY scripts/route_a_v3/sync_route2_training_attempt_ledger_v1.py \
  --ledger /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiment_tracking/route2_training_attempts.csv \
  --run-dir /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/mrnabert_scaleup_v1/max_mean_only_seed20260816_gpu6_v1
```

一次性回填整个 Route 2 历史运行目录：

```bash
$PY scripts/route_a_v3/sync_route2_training_attempt_ledger_v1.py \
  --ledger /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiment_tracking/route2_training_attempts.csv \
  --runs-root /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs
```

## 4. 服务器环境与测试

```bash
ssh A100
cd /home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_route2_method_repair_20260817
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python

$PY -m pytest -q \
  tests/route_a_v3/test_route2_experiment_ledger_v1.py \
  tests/route_a_v3/test_build_route2_mrnabert_feature_cache_v1.py \
  tests/route_a_v3/test_route2_delta_predictor_v1.py
```

## 5. 构建 mRNABERT 冻结特征

```bash
$PY scripts/route_a_v3/build_route2_mrnabert_feature_cache_v1.py \
  --config configs/route_a_v3_route2_mrnabert_full_development_feature_cache_v1.json
```

输出：

- `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/pretrained_features/mrnabert_full_development_v1.pt`
- `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/pretrained_features/mrnabert_full_development_v1.summary.json`

## 6. 三种 loss 的 HPO 命令

```bash
$PY scripts/route_a_v3/train_route2_delta_predictor_v1.py \
  --config configs/route_a_v3_route2_mrnabert_edit_max_mean_only_gpu6_v1.json

$PY scripts/route_a_v3/train_route2_delta_predictor_v1.py \
  --config configs/route_a_v3_route2_mrnabert_edit_max_fixed_variance_gpu6_v1.json

$PY scripts/route_a_v3/train_route2_delta_predictor_v1.py \
  --config configs/route_a_v3_route2_mrnabert_edit_max_learned_variance_gpu6_v1.json
```

每个配置训练 100 epochs；TRAIN 为 89,580 条，约 559,900 次 optimizer updates；只使用 VALIDATION 选择 checkpoint。三个 loss 同结构、同数据和同 seed，用于直接检查可学习 uncertainty 是否吸收误差并导致 mean collapse。

三个 run 全部完成后，用同一条冻结规则做汇总；不按 NLL 大小选择 learned-variance 模型：

```bash
$PY scripts/route_a_v3/summarize_route2_mrnabert_loss_comparison_v1.py \
  --summary /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/mrnabert_scaleup_v2/max_mean_only_seed20260816_gpu0_bf16_v1/training_summary.json \
  --summary /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/mrnabert_scaleup_v2/max_fixed_variance_seed20260816_gpu5_bf16_v1/training_summary.json \
  --summary /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/mrnabert_scaleup_v2/max_learned_variance_seed20260816_gpu3_bf16_v1/training_summary.json \
  --output /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/comparisons/mrnabert_loss_comparison_seed20260816_v1.json
```

选择后的 loss 还必须运行两个同预算 signal controls：

- `WITHIN_EXACT_SOURCE_TASK_TRAIN_CANDIDATE_PERMUTATION`：候选 token 与其冻结 mRNABERT 表征一起按 source/task 内打乱；
- `PARAMETER_MATCHED_PRETRAINED_SOURCE_ONLY`：参数规模匹配，但不读取 candidate-specific 信息。

长任务的低频接力入口为：

```bash
nohup scripts/route_a_v3/schedule_route2_mrnabert_postselection_controls_v1.sh \
  >/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/schedulers/mrnabert_postselection_controls_v1.log \
  2>&1 </dev/null &
```

它每 15 分钟检查三个 summary，并按固定顺序接力：loss VALIDATION 选择 → GPU 0/5 两个 signal controls → GPU 0/3/5 三个固定验证 seeds → 单次冻结 Development TEST → 全 126,165 条最终拟合 → 7 个 Development study × 3 seeds 的 TEST-preserving mRNABERT LOSO → 同样 21 个 matched global-scaled baseline LOSO → 三个 seed 的对齐汇总。随后等待 Flow V2 和在线 mRNABERT candidate encoder 的独立验证，组装 readiness；只有 `CRITIC_READY_FOR_GUIDANCE` 与 `FLOW_G0_READY` 同时成立，才会在空闲 GPU 0 上启动一次 guided XEditFlow Development run。任一门槛不通过，后续阶段停止。该调度器始终不会读取 GSE232572 或 E-MTAB-10902 的 Evaluation outcomes。

三个 loss 完成后、启动 controls 之前，调度器还会生成 `mrnabert_uncertainty_absorption_seed20260816_v1.json`。它从 Development VALIDATION 逐 task 重算 mean Spearman、正向 task 数、standardized MAE、prediction-spread ratio，以及 learned uncertainty 与绝对残差的相关性。这样不会让分钟、log-fold 和 usage 的不同原始量纲混成一个无意义的全局 spread。该报告只诊断 learned uncertainty 是否伴随 mean collapse；loss 选择仍只使用预冻结的 mean 性能规则，不会因 NLL 或预测方差看起来更好而选中 learned-variance。

## 7. 冻结 TEST 与最终拟合

冻结 TEST 不再直接使用预写死为 Huber 的旧配置。它由三个 final seeds 的冻结判定动态生成：

```bash
$PY scripts/route_a_v3/adjudicate_route2_mrnabert_three_seeds_v1.py \
  --protocol configs/route_a_v3_route2_mrnabert_three_seed_gate_v1.json \
  --summary <seed-20260822-training-summary.json> \
  --summary <seed-20260823-training-summary.json> \
  --summary <seed-20260824-training-summary.json> \
  --output <three-seed-adjudication.json>

$PY scripts/route_a_v3/prepare_route2_mrnabert_frozen_test_config_v1.py \
  --selected-config <selected-loss-seed-20260823-config.json> \
  --three-seed-adjudication <three-seed-adjudication.json> \
  --gpu 0 \
  --output-directory <new-frozen-test-run-directory> \
  --output-config <new-frozen-test-config.json>

$PY scripts/route_a_v3/train_route2_delta_predictor_v1.py \
  --config <new-frozen-test-config.json>
```

该配置使用 TRAIN+VALIDATION 共 107,873 条拟合，最后评测 18,292 条 TEST。

TEST 完成后，最终拟合配置由 TEST 记录生成；这里只验证一次性 TEST 已完成，不根据 TEST 数值重新选择结构、loss、seed 或轮数：

```bash
$PY scripts/route_a_v3/prepare_route2_mrnabert_all_development_refit_config_v1.py \
  --frozen-test-config <frozen-test-config.json> \
  --frozen-test-summary <frozen-test-training-summary.json> \
  --gpu 0 \
  --output-directory <new-all126165-run-directory> \
  --output-config <new-all126165-config.json>

$PY scripts/route_a_v3/train_route2_delta_predictor_v1.py \
  --config <new-all126165-config.json>
```

该结果不重新产生内部模型选择结论。

## 8. 当前底层优化状态

已经启用：

- mRNABERT 只前向一次并缓存冻结特征，100-epoch critic 训练不重复运行 113M encoder；
- mRNABERT 特征构建使用 BF16 autocast、按 token 数量动态组 batch、对长序列分块；
- mRNABERT 官方模型采用 ALiBi 相对位置偏置，不依赖固定长度的 learned absolute position embedding；
- critic 显式加入归一化绝对位置和 edit-gated position channels；
- critic 使用 edit-centered attention pooling、edit max pooling、source mean/max 全局背景、region FiLM 和 source/candidate 反对称结构。
- A100 数值对照通过后，正式 loss 比较已启用 BF16 autocast、fused AdamW、pinned memory 和 non-blocking transfer；
- workers 0/4/8 的同口径 batch32 基准显示 `num_workers=4` 最快：`310.94 records/s`，高于 workers0 的 `290.58 records/s`；workers8 为 `291.50 records/s`，继续增加 worker 没有收益。后续新的正式 cohort 使用 workers4；已经完成的 run 不追溯重跑。
- 新生成候选不在 canonical feature cache 中，因此已增加冻结 mRNABERT 在线编码器；它按 candidate batch 运行、在 trajectory 内做 sequence memoization，并先与 canonical cache 做数值对齐验证。
- 在线编码器和 guided potential cache 只在当前 source/budget cohort 内保留；进入下一个 source 时主动清空，避免对成千上万候选长期累积 768 维 embedding 而耗尽主存。
- guided XEditFlow 的合法动作仍由 SUB+STOP kernel 枚举，hard mask 不交给 critic；base transition rate 使用冻结 critic 的 clipped mean-potential difference 做一致性倾斜，critic uncertainty 不进入 guidance。

当前未启用：

- FlashAttention：官方仓库附带的旧 Triton kernel 与服务器当前 Triton API 不兼容，因此现在使用官方 PyTorch attention fallback；
- `torch.compile` 尚未启用；当前没有证据表明其编译开销能被这批固定长度的 critic 训练摊薄。

FlashAttention 主要影响一次性的冻结特征构建和后续候选在线编码，不能提升 100-epoch critic 的主体训练。当前在线路径优先用 candidate batching 与 memoization 降低重复前向；在没有证明新的 attention kernel 与冻结 cache 数值一致前，不替换官方 PyTorch fallback。当前使用的 BF16/fused 路径已经先通过独立吞吐与数值一致性测试，再用于全新的正式 runs；没有中途改变既有 run 的数值路径。

已增加一个不读取项目数据的 attention operator 基准，比较官方 `QK-softmax-V` fallback 与 PyTorch 2.5 SDPA 的 AUTO、Flash、memory-efficient 和 math 后端：

```bash
CUDA_VISIBLE_DEVICES=4 /home/cunyuliu/miniconda3/envs/editflow/bin/python \
  scripts/route_a_v3/benchmark_route2_mrnabert_attention_backend_v1.py \
  --config configs/route_a_v3_route2_mrnabert_attention_backend_benchmark_gpu4_v1.json
```

该基准只筛查 `50/64/96/128/164 nt × batch 1/4/8` 下的数值误差、真实融合后端和速度。即使通过，也只允许继续做全编码器 cache 对齐与端到端吞吐验证，不会自动切换正式 encoder。位置编码继续使用官方双向 ALiBi；不会在加载同一预训练权重时改成 RoPE 或新的绝对位置编码。

## 9. 2026-08-19 01:20 运行快照

| 运行 | 终态/当前状态 | task-macro Spearman | task-macro standardized MAE | 说明 |
|---|---:|---:|---:|---|
| RNA-FM Huber | **100/100 完成，best epoch 94** | 0.124236 | 2.141658 | 历史 encoder 对照；global Spearman 0.202329 |
| RNA-FM learned variance | **100/100 完成，best epoch 49** | 0.130418 | 1.935143 | global Spearman 0.200763；prediction std / target std 0.031654 |
| RNA-FM fixed variance | **100/100 完成，best epoch 92** | 0.098003 | 2.306655 | global Spearman 0.174160；prediction std / target std 0.138453 |
| mRNABERT Huber | **100/100 完成，best epoch 44** | **0.149988** | **2.108870** | global Spearman 0.198122；prediction std / target std 0.074666 |
| mRNABERT fixed variance | **100/100 完成，best epoch 54** | 0.120695 | 2.428299 | global Spearman 0.171786；prediction std / target std 0.080062 |
| mRNABERT learned variance | **100/100 完成，best epoch 84** | 0.123583 | 2.523861 | global Spearman 0.176032；prediction std / target std 0.149625 |

三种 mRNABERT loss 已按同一数据、split、seed、架构和训练预算完成。冻结选择规则选中 Huber：其 task-macro Spearman `0.149988` 高于 fixed variance 的 `0.120695` 和 learned variance 的 `0.123583`。learned variance 的预测不确定性与绝对残差存在正相关（Spearman `0.490600`），但均值任务相关性比 Huber 低 `0.026406`，standardized MAE 也更差；因此该 uncertainty head 有吸收残差尺度的迹象，却没有改善均值预测，不能因为 NLL 或方差诊断看起来合理而胜出。

同信息最强已完成 baseline 的 task-macro Spearman 为 `0.131714`。Huber 的 candidate-permutation 与 parameter-matched source-only controls 均通过冻结联合判定，但 three-seed confirmation 没有复现单 seed 的正向增量：seed 20260822、20260823、20260824 的 task-macro Spearman 分别为 `0.116129`、`0.116908`、`0.137384`，相对 strongest baseline 的 margin 分别为 `-0.015586`、`-0.014806`、`+0.005669`。只有 1/3 seed 为正，因此终态为 `THREE_FINAL_SEEDS_DO_NOT_SUPPORT_FROZEN_DEVELOPMENT_TEST`。调度器已按预冻结规则停止，Development TEST、Evaluation outcomes、全量 refit、LOSO readiness 和 guided XEditFlow 均未打开。中央训练表当前为 94 个唯一尝试、72 列：86 completed、3 failed、3 incomplete、2 stopped。

## 10. Guidance readiness 与执行入口

Base Flow V2 不依赖 critic，可以继续完成 unguided `SUB+STOP` 工程验证。原训练等待器只盯 GPU 2，在该卡被其他正常作业持续占用时空等超过 60 小时。当前规则扫描 GPU 0–5，只按剩余显存选择显存最多的一张卡，不再设置利用率门槛。新 run 启动后的实际占用约 1.04GB，因此保留 2GB 的任务最低可运行线以避免明显 OOM；这不是资源 gate。该修改不改变数据、模型、seed、30 epochs、BF16/fused 路径或输出目录。训练与验证分别使用：

```bash
nohup scripts/route_a_v3/schedule_route2_base_flow_v2_training_v1.sh \
  >/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/schedulers/base_flow_v2_dynamic_gpu_training_v1.log \
  2>&1 </dev/null &

nohup scripts/route_a_v3/schedule_route2_base_flow_v2_validation_v1.sh \
  >/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/schedulers/base_flow_v2_dynamic_gpu_validation_v1.log \
  2>&1 </dev/null &
```

实际选中的 physical GPU 会写入 runtime config、training summary 和 CUDA provenance；目录名中的 `v2` 表示该实验版本，不再表示固定使用 GPU 2。

冻结 guidance policy：

```text
configs/route_a_v3_route2_mrnabert_guidance_reward_policy_v1.json
```

它只使用 standardized predicted mean；learned uncertainty 只做诊断，不进入 reward。在线 candidate encoder 的单次验证由下列低频调度器在 GPU 0–5 中选择剩余显存最多且至少有 4GB 可用的卡执行，不使用利用率门槛：

```bash
nohup scripts/route_a_v3/schedule_route2_mrnabert_online_encoder_validation_v1.sh \
  >/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/schedulers/mrnabert_online_encoder_validation_v1.log \
  2>&1 </dev/null &
```

readiness 和 guided runner 分别为：

```text
scripts/route_a_v3/build_route2_mrnabert_guidance_readiness_input_v1.py
scripts/route_a_v3/adjudicate_route2_readiness_v1.py
scripts/route_a_v3/run_route2_guided_xeditflow_v1.py
```

guided runner 在 readiness 未全通过时会在读取 source manifest、base-flow checkpoint 和 critic checkpoint 之前停止；即使完成，它仍只代表 Development generation 完成，不代表 biological improvement，generated candidates 也不增加 canonical records。

guided runner 现在另外生成 `guided_compute_by_source.jsonl`，只记录 source key、编辑预算、候选预算和计算量，不记录序列。搜索基线不再使用旧的统一 `256` 次 critic 预算，而是逐 source 读取 guided run 的 `critic candidate forwards + generator NFE` 总预算，并把这个完整额度全部允许给搜索基线做 critic 查询。这是对搜索基线更有利的保守匹配。六种搜索方法依次运行，使用完全相同的最终 mRNABERT critic、reward policy、source pool、candidate budget 和逐 source 总 forward-equivalent 上限：

```text
scripts/route_a_v3/run_route2_mrnabert_matched_search_suite_v1.py
configs/route_a_v3_route2_mrnabert_matched_search_development_gpu0_v1.json
```

这个步骤只生成 matched-budget candidates；在独立评估器合格前不选择“最强生成方法”，也不把 critic 自评分写成科学收益。

候选生成完成后，主调度器会调用：

```text
scripts/route_a_v3/run_route2_mrnabert_generation_comparison_suite_v1.py
configs/route_a_v3_route2_mrnabert_generation_comparison_development_gpu0_v1.json
```

它把 guided XEditFlow 与 random、greedy、beam、genetic、local search、generate-then-rerank 和 unguided learned base flow 一起交给预冻结的独立 evaluator。先在七个 baseline 中用 source-paired bootstrap 冻结 strongest baseline，再比较 guided 与 strongest；未知的生成候选不会被当作 measured zero。该结果仍只是 Development independent-evaluator evidence，不是外部 Evaluation，也不是 measured biological improvement。

## 11. 独立评估器修复

旧独立评估器曾直接混合不同 endpoint 的原始量纲，分钟级 half-life 误差会压倒 log-fold 和 usage，因此该 run 已作为 `STOPPED_PRETERMINAL_METHOD_INVALID` 保留，不能用于生成方法选择。新的单次预冻结 run 使用 TRAIN-only task robust scaling、独立的 0.51M Siamese CNN 和 Development VALIDATION；它在 Base Flow validation 完成后从 GPU 0–5 中选择剩余显存最多且至少有 1GB 可用的卡，不再按利用率等待，也不读取 mRNABERT 特征、Development TEST 或 Evaluation：

```bash
nohup scripts/route_a_v3/schedule_route2_independent_evaluator_gpu2_v3.sh \
  >/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/schedulers/independent_evaluator_gpu2_v3.log \
  2>&1 </dev/null &
```

该调度器等待 Base Flow V2 完成后再启动，不与当前训练争卡。它只运行一次，不按结果追加 HPO；只有 task-macro Spearman 超过预冻结 candidate-permutation reference `0.1012476`、至少 5 个 task 为正且所有数据隔离检查通过，才标记为 qualified。无论 PASS 或 NO-GO，尝试都会自动写入中央训练表。

## 12. 2026-08-19 01:20 追加快照

- 三种 mRNABERT loss 已全部完成，正式选择 Huber；learned uncertainty 与残差相关但均值性能下降，因此不用于 loss 选择或 guidance reward；
- 两项 signal controls 虽然通过，但 three-seed confirmation 失败：三个 task-macro Spearman 为 `0.116129 / 0.116908 / 0.137384`，只有 1/3 超过 strongest baseline；冻结 TEST、全量 refit和 guided XEditFlow 均未启动；
- 三个 RNA-FM 历史对照均已完成并同步中央表，仅作为 encoder 历史对照；
- Huber 实际 wall time 为 39,616.45 秒，约 396.16 秒/epoch、70.76 ms/step 和 226.12 TRAIN records/s；相对 batch16 BF16 微基准约为 93%，没有严重 DataLoader 饥饿证据；
- GPU 0–5 当前均有项目任务，显存状态正常；control 启动验收未见 CUDA/NaN/提前退出；
- workers 0/4/8 的 batch32 数据管线测试已完成：workers4 为 310.94 records/s，优于 workers0 的 290.58；workers8 为 291.50，没有继续扩 worker 的收益。该结果只用于后续新正式 cohort；Base Flow V2、在线 mRNABERT encoder validation 和独立 evaluator 继续低频等待各自 GPU 条件；
- 独立评估器将等待 Base Flow 验证完成后再运行；Base Flow 训练/验证已改为在 GPU 0–5 中选择剩余显存最多且至少有 2GB 可用的卡，不再永久绑定 GPU 2，也不再以利用率为启动条件；
- 当前正式数据与 claim 状态仍为 `ordinary=1 / A1=1 / true-A2=0 / canonical=6,547 / NOT_ESTABLISHED`。

## 13. 2026-08-20 在线编码器终态与并行工程进度

在线 mRNABERT 编码器验证已于 `2026-08-20T12:17:39+08:00` 在动态选择的 physical GPU 0 上完成，终态为 `ONLINE_FROZEN_MRNABERT_MATCHES_CANONICAL_CACHE`。本次验证没有读取 Development TEST 或 GSE232572/E-MTAB-10902 Evaluation outcomes，也没有参数更新。

| 检查项 | 结果 |
|---|---:|
| 冻结 mRNABERT 参数量 | 113,389,056 |
| canonical-cache 对比 embedding 数 | 64 |
| 最大绝对差 | 0.00951385 |
| 预冻结容差 | 0.01 |
| 中位编码吞吐 | 74.55 sequences/s |
| novel candidate 在线编码 | 支持 |
| Evaluation records read | 0 |

这只闭合“新候选能否用同一个冻结 mRNABERT 在线编码，并与 canonical cache 在容差内一致”的工程条件；它不改变 three-seed 失败结论，也不授权 Development TEST、all-126,165 refit、LOSO、guided XEditFlow 或外部 Evaluation。当前 official PyTorch fallback backend 已能通过一致性验证；后续 SDPA/Flash Attention 检查只属于速度优化，不得被写成新的科学结果。

Base Flow V2 在本次检查开始时尚未启动，因为旧调度规则仍要求 `free_memory >= 24GB` 与 `utilization <= 70%`。该规则随后按负责人指令被替换：当前只按剩余显存选择 GPU 0–5 中显存最多的一张卡，Base Flow 默认最低 2GB，不再检查利用率。现有训练作业不会被终止；新任务允许与高利用率 GPU 上的既有任务共享计算时间。

两个旧的 SDPA/attention backend 等待器当时仍固定等待 GPU 4，存在空闲窗口到来时先于 Base Flow 抢占该卡的风险。它们均只是等待 shell、没有 CUDA 子进程，已安全停止；Base Flow 训练、Base Flow validation 和独立 evaluator 三个主调度器保持运行。attention backend screen 与 full-encoder SDPA alignment 延后到 Base Flow validation 之后再恢复，不改变已完成的 official PyTorch fallback 一致性结论。

Base Flow validation 的一次性汇总现在同时记录：candidate-budget mismatch、unique/duplicate candidate 数、global 与 source-macro unique rate、每条轨迹平均 generator NFE、候选输出/秒、含固定 seed replay 的 sampling invocation/秒与 generator NFE/秒。validation wall time 明确包含一次逐候选 replay，因此该吞吐是带可重放性检查的保守工程吞吐，不会冒充生产采样速度。任何 edit-budget violation、candidate-budget mismatch、replay failure 或 numerical failure 都使 `FLOW_G0_READY` 失败；无需为补采样效率另跑第二次 validation。

matched-budget suite 现已把“生成候选”和“用独立 evaluator 选择最强方法”拆开：若 evaluator 合格，继续七种方法的候选生成、独立评分、Development open-support evaluation 与 strongest selection；若 evaluator 为 NO-GO，仍按冻结预算生成七种候选并保留计算账本，但立即停止在评分与 selection 之前，终态写为 `MATCHED_GENERATION_CANDIDATES_COMPLETED_EVALUATOR_NO_GO`。因此 evaluator 失败不会让 Flow/搜索工程倒退，同时也不能被绕过来宣称最强方法或科学收益。

Base Flow V2 训练已于 `2026-08-20T13:24:55+08:00` 完成并自动写入中央训练表。终态为 `LEARNED_BASE_FLOW_GPU_UPDATE_COMPLETE`：68,294 TRAIN、15,924 VALIDATION、18,292 withheld Development TEST、Evaluation read=0；817,957 个可训练参数，30 epochs、32,040 optimizer updates、BF16 + fused AdamW，GPU 0 峰值显存 289.49MB，总 wall time 2,196.71 秒，平均约 73.22 秒/epoch。position/progress features 均启用，critic 未进入训练。

该 run 的 best validation NLL 为 5.51248，selected epoch=1。TRAIN NLL 从 5.12598 持续下降到 0.43646，而 VALIDATION NLL 在第 1 epoch 后明显恶化，说明 learned base-rate model 存在强过拟合；这不是 biological optimization 成功。工程上仍使用冻结的 epoch-1 checkpoint 继续一次合法性、可重放性、小图与采样效率 validation，验证终态将决定 `FLOW_G0_READY`，不会因为训练 loss 下降就提前通过。

新的 matched-generation v2 调度器会等待 Flow validation 与独立 evaluator 终态，然后只按剩余显存在 GPU 0–5 中选择卡，使用新的 position/progress Base Flow checkpoint 与新的独立 evaluator checkpoint，在独立 v2 输出目录运行七种 matched-budget 方法。启动命令为：

```bash
nohup scripts/route_a_v3/schedule_route2_matched_generation_suite_v2.sh \
  >/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/schedulers/matched_generation_suite_v2.log \
  2>&1 </dev/null &
```

该调度器不会读取外部 Evaluation outcomes；evaluator NO-GO 时仍生成候选但停止在独立评分之前，qualified 时才完成评分与 strongest selection。旧 v1 输出不覆盖。

中央训练尝试表当前为 96 个唯一尝试、72 列：87 completed、3 failed、3 incomplete、2 stopped，以及 1 个正在运行的独立 evaluator。online encoder 是零参数更新的工程验证，不伪装成训练尝试；Base Flow V2 的训练终态已进入中央表。

## 14. 低频查看进度

运行中只在事件节点低频查看：

```bash
tail -n 5 \
  /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/mrnabert_scaleup_v2/max_mean_only_seed20260816_gpu0_bf16_v1/metrics.jsonl
```

结束后查看一次汇总：

```bash
python -m json.tool \
  /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/mrnabert_scaleup_v2/max_mean_only_seed20260816_gpu0_bf16_v1/training_summary.json
```

## 15. 2026-08-20 执行顺序复盘与当前裁定

本节用于把原定 12 步与真实产物逐项对齐。它不根据后续结果改写原门槛，也不把未执行步骤写成完成。

| # | 原定步骤 | 当前裁定 | 直接证据或原因 |
|---:|---|---|---|
| 1 | Huber 跑满 100 epochs 并保留 best checkpoint | **完成** | 100/100 epochs；best epoch 44；`best.pt`；wall time 39,616.45 秒；Development TEST 未读 |
| 2 | workers 0/4/8 DataLoader benchmark | **完成** | workers4 最快：310.94 records/s；workers0 290.58；workers8 291.50 |
| 3 | 实际 epoch 时间与瓶颈分析 | **完成** | 约 396.16 秒/epoch、70.76 ms/step；full loop 达 batch16 BF16 微基准的 93.1%，没有严重 loader starvation；batch64 实测退化，batch32 被选中 |
| 4 | fixed-variance 与 learned-variance | **完成** | 两项均 100/100 epochs，与 Huber 使用同一数据、split、seed、架构和预算 |
| 5 | 三种 loss 同 Validation 比较 | **完成** | Huber task-macro Spearman 0.149988；fixed 0.120695；learned 0.123583。learned uncertainty 与残差相关，但 mean 更差，因此选 Huber |
| 6 | permutation、source-only、anchor-only、parameter-matched controls | **完成两次独立运行；术语需澄清** | candidate-permutation 已运行；source-only 已运行且参数量匹配。若 anchor-only 指“只保留 source/anchor”，它就是同一个 source-only control，不应重复计为第三次实验；若指“source + edit metadata、去掉 candidate sequence”，则目前 **未定义、未运行** |
| 7 | 三个 final seeds | **完成但未通过** | 三个 margin 为 -0.015586、-0.014806、+0.005669；只有 1/3 为正；终态 `THREE_FINAL_SEEDS_DO_NOT_SUPPORT_FROZEN_DEVELOPMENT_TEST` |
| 8 | 单次冻结 Development TEST | **按门槛停止，未打开** | three-seed gate 未通过；TEST outcomes read=0 |
| 9 | 全部 126,165 records 最终 refit | **按门槛停止，未启动** | 该步骤依赖单次 TEST 记录；不能跳过失败门槛产生“最终模型” |
| 10 | frozen-critic guided XEditFlow | **未授权、未启动** | critic readiness 未通过；不得用单 seed checkpoint 冒充冻结 critic。与 critic 无关的 Base Flow G0 已独立完成 |
| 11 | 同预算搜索/生成 baseline | **正在准备** | Base Flow validation 已达到 `FLOW_G0_READY`；独立 evaluator 正在 Development TRAIN/VALIDATION 上训练，终态后由已启动的调度器进入七种 matched-budget 方法 |
| 12 | GSE232572 / E-MTAB-10902 独立 Evaluation | **未开始，继续隔离** | 两个 Evaluation outcomes 均未读取；只有 Development 上的方法与预算冻结后才允许进入 |

### 15.1 反思

1. **记录中的 DataLoader 结论曾自相矛盾。** 早期文字写成 workers0 最快，但正式 benchmark 明确是 workers4 最快；本文件已修正，后续以正式 summary 为准。
2. **“controls 通过”不应掩盖术语重叠。** 当前只有 candidate-permutation 与 parameter-matched source-only 两次真实运行；source-only 在本项目里就是只看 source anchor 的对照。除非先定义一个不同的信息边界，否则不制造一个同义的第三次 anchor-only run。
3. **单 seed 的最好结果没有经受 three-seed 复现。** 这说明当前 critic 不能用于打开 TEST、最终 refit 或指导生成；增加 epoch、参数量或事后挑 seed 都不能修复这一结论。
4. **工程线与科学线要分开。** Base Flow V2 的 legality、budget、replay 与 small-graph checks 已通过，说明生成器工程骨架可用；它不说明 biological optimization 成功，也不替代 critic gate。
5. **当前独立 evaluator 是 Step 11 的必要基础设施，不是外部 Evaluation。** 它只读 Development TRAIN/VALIDATION，用来避免用生成器自己的 critic self-score 选择生成方法；无论 PASS 或 NO-GO，都不得读取 GSE232572/E-MTAB-10902。

### 15.2 当前正在做什么、下一步做什么

当前唯一新的 GPU 训练是独立 evaluator。它与 mRNABERT critic、生成器训练梯度和外部 Evaluation 隔离；中央表状态为 RUNNING。Base Flow V2 已完成，V4 validation 终态为 `FLOW_G0_READY`：891 个 source cohorts、28,512 条 trajectories、hard legality 100%、budget violation 0、replay failure 0、small-graph TV distance 0，source-macro unique candidate rate 0.882891。

后续顺序冻结为：

1. 等独立 evaluator 自然结束，只读取一次终态 summary/adjudication；
2. evaluator qualified 时，运行并独立评分七种 matched-budget generation/search 方法；evaluator NO-GO 时仍完成候选生成与 compute ledger，但停止在方法评分和 strongest selection 之前；
3. 保持冻结 TEST、all-126,165 refit、critic-guided XEditFlow 与外部 Evaluation 关闭；
4. 根据生成 baseline 结果和 critic 失败原因，单独设计下一版 critic 的前瞻性假设，不复用 TEST、不事后放松 three-seed gate；
5. 只有新的 critic cohort 重新满足 controls + three-seed readiness，才恢复步骤 8→9→10；步骤 12 永远最后执行。

## 16. V3.3.2 freshness、generation terminal 与 Critic V2 冻结（2026-08-20 19:32）

一次性 freshness check 已确认此前排在最前的两个任务都已经 terminal，故不得重复运行：

- 独立 evaluator 的正式状态为 `INDEPENDENT_GENERATION_EVALUATOR_QUALIFIED`。它是 509,845 参数的独立 Siamese CNN，8 epochs / 22,400 updates，Development VALIDATION task-macro Spearman `0.102565536`，超过预冻结 exclusive threshold `0.101247575` 的 margin 为 `+0.001317961`，5/9 tasks 为正；Development TEST 和 Evaluation outcome 均未读取。该 margin 很小，只授权 Development 生成方法选择，不是 biological validation。
- 七种 matched-budget generation/search 方法全部完成，suite 状态为 `MATCHED_GENERATION_BASELINE_SUITE_COMPLETED`。891 个 source、每 source 32 candidates、256 critic-forward budget 与 320 total forward-equivalents 的冻结口径未变；全部方法 hard legality=100%，edit/candidate budget violation=0。独立 evaluator 的 point leader 和冻结 strongest baseline 都是 `genetic`，source-macro independent-evaluator max uplift 为 `1.097825`。这仍是 `INDEPENDENT_EVALUATOR_ONLY_MEASURED_OUTCOME_NOT_ESTABLISHED`，不能写成 biological improvement。
- 中央训练尝试表共有 96 个唯一 attempts，最新一行 evaluator 已从 RUNNING 更新为 COMPLETED；Development TEST 18,292 条继续 withheld，Evaluation record count 仍为 0。

generation terminal 还显示：genetic 的 source-macro unique rate 为 `1.0`、pairwise Hamming diversity 为 `0.0682601`、平均 total forward-equivalents/source 为 `231.4669`；unguided Base Flow 的 measured-candidate recovery 和 measured top-k recovery 分别为 `0.202862` 与 `0.0979731`，但这些是 Development measured-neighborhood recovery，不是新 external Evaluation outcome。

因此当前最靠前的未完成任务转为 Critic V2。其前瞻冻结假设只改变优化与信息 control，不改 encoder 或容量：

```text
fixed TRAIN draws:
task -> study -> source-context-endpoint group -> record
    + per-batch equal task-macro Huber aggregation
    + frozen mRNABERT edit-centered 9.343M critic
```

四臂 screen 为 full、exact-source/task candidate permutation、parameter-matched source-only，以及独立的 `source + edit identity/position/context、无 candidate global mRNABERT representation` control。后者明确不是 source-only 的同义重复。现有 strongest same-information baseline `0.131714395` 继续冻结并复用，不重复运行。screen seed 为 `20260825`；只有 controls gate 通过，才允许再次运行预冻结的三个 seeds `20260822/20260823/20260824`，不得增加第四个 seed。

Critic V2 代码、协议和调度器进入 Git 后，A100 只启动一份：

```bash
nohup scripts/route_a_v3/schedule_route2_mrnabert_critic_v2_controls_v1.sh \
  >/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/schedulers/mrnabert_critic_v2_controls_v1.log \
  2>&1 </dev/null &
```

调度器只在 GPU0–5 中按剩余显存选择四张满足实际最低显存的卡，不使用 utilization gate；每个 arm 启动和结束时由训练器自动更新中央尝试表。controls terminal 前不创建 three-seed 结果，Development TEST、all-126,165 refit、LOSO、guided XEditFlow 和新的 final Evaluation 全部保持关闭。GSE232572 只保留 historical transfer/diagnostic 角色；E-MTAB-10902 仍为 `UNCONVERTIBLE_FOR_ROUTE2_V1`，最终独立确认必须使用新的、可转换且 outcome 未暴露的 replacement study。

## 17. Critic V2 control RUNNING 与条件式 three-seed 路径（2026-08-20 20:05）

Critic V2 control scheduler 已以 PID `4104921` 启动并完成四臂初始化：full
在 GPU2、candidate permutation 在 GPU4、source-only 在 GPU3、source+edit
metadata control 在 GPU5。中央训练表已出现四条 RUNNING，唯一 attempts 总数
为 100；四条记录的 sampling 都是
`TASK_STUDY_SOURCE_GROUP_BALANCED_FIXED_DRAWS`，loss aggregation 都是
`TASK_MACRO_MEAN`。本节只记录启动事件，不读取 epoch 或 Validation outcome。

三种子后继协议在 control terminal 前冻结，配置为：

```text
controls adjudication PASS
  -> exact seeds 20260822 / 20260823 / 20260824
  -> Development TRAIN/VALIDATION only
  -> one prospective three-seed adjudication
  -> stop after PASS/NO-GO; do not open TEST in this scheduler
```

后继 scheduler 每 15 分钟只检查 control adjudication 这个 terminal artifact。
NO-GO 时不创建 seed runtime/run 目录；PASS 时才从 GPU0-5 选择三张剩余显存
至少 4096 MiB 的卡。裁决除 3/3 strongest-baseline margin 外还完整输出
standardized MAE、prediction spread、positive-task count、相对 permutation /
source-only / source+edit-metadata controls 的 macro gap，以及 non-finite、mean
collapse 和可重放尺度诊断。control gap 没有新增事后阈值，因为 candidate-
specific signal 已由前一 control gate 冻结裁决。

聚焦验证结果为 `74 passed, 4 skipped`；4 项均为本机无 CUDA 的既有 skip，
另有一条 PyTorch Transformer warning，不影响判定。A100 项目环境的 control
聚焦测试为 57 项通过。两个 scheduler 的 shell 语法检查均通过。Development
TEST 与新 final Evaluation outcome 都未读取，guided generation 仍未授权。

上述条件式 watcher 已在 A100 启动为 PID `4148582`，日志路径为
`/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/schedulers/mrnabert_critic_v2_three_seeds_v1.log`。
首条日志确认它正在等待 control adjudication；验收时 three-seed runtime、run
和 adjudication 均未创建。A100 新增的 7 个 confirmation focused tests 全部通过。

## 18. Route 2 V3.3.2 manuscript evidence packet（2026-08-20）

等待 Critic V2 control terminal 期间，已建立本地、未向外部服务传输的
evidence-bound Methods/Results draft、evidence manifest、consistency manifest
和七方法全精度 CSV。所有事实句保留 claim/evidence markers，状态明确为
`INTERNAL_EVIDENCE_BOUND_NOT_SUBMISSION_READY`，需负责人逐项 human
verification 后才能并入正式稿。

该 packet 保留两个不能被润色抹掉的结果：genetic 是独立 evaluator uplift
criterion 的 Development strongest baseline；unguided Base Flow 则在 sparse
measured-neighborhood candidate recovery 与 top-k recovery 上领先。两种排序不
混写，也都不解释为 biological improvement。另记录一个真实 reporting gap：
六个 search 方法的 per-method generation wall time 未持久化，所有 source-level
字段均为 null；只报告 overall suite wall、独立 scoring wall 和 Base Flow 的独立
341.56 秒汇总，不通过文件时间戳倒推缺失值，也不重跑 terminal suite。

该论文任务没有参数更新，因此不向中央训练尝试表增加伪训练行；最近一次事件
节点仍为 100 个唯一 attempts，其中四个 Critic V2 arms 为 RUNNING。packet
一致性检查通过：13 个 claim markers、12 个 evidence sources、7 个唯一方法，
所有 evidence references 均闭合。

并行 stage runner 随后补充了未来运行的逐方法
`wall_time_seconds`：每个 subprocess 从实际启动到退出由独立 wait 记录，不再
用整阶段时长代替各方法时长。该改动不重跑 terminal baseline，也不回填既有
缺失值。focused suite tests 为 6/6 通过；额外的通用 entrypoint 检查中 21 项
通过、2 项因本机系统 Python 缺少既有 `h5py` 依赖失败，失败脚本为 APARENT
和 external-prediction baseline，与本次计时路径无关，未据此扩展修改范围。

## 19. Critic V2 单次冻结 Development TEST 门前瞻冻结（2026-08-20）

在 Critic V2 control screen 尚未 terminal、three-seed 结果尚不存在时，已完成
V2-only 单次 Development TEST 配置门的前瞻冻结。审计确认历史 V1 preparer
会把 checkpoint policy 改成 `FINAL_EPOCH`，且只绑定旧 V1 loss/裁决状态，不能
复用于 Critic V2；V2 路径因此使用独立协议与独立 preparer，不修改历史负队列。

冻结协议固定 seed `20260823`、Development TEST 记录数 18,292、GPU0-5 和
完整 Critic V2 训练策略。confirmation 的选择来源保持 `BEST_VALIDATION`；单次
TEST 会把 TRAIN+VALIDATION 折入训练，因此其可执行规则在 outcome 出现前固定为
100 epochs + `FINAL_EPOCH`，TEST 不参与 epoch 或 checkpoint 选择。preparer 只有在
以下条件全部成立时才允许写出唯一 runtime config：control adjudication PASS；
三个固定 seed `20260822/20260823/20260824` 的 adjudication PASS；3/3 seed 相对
同一 strongest same-information baseline 的 margin 均为正；三个协议、两个裁决
和被选 seed config 的 policy/seed/baseline/protected-outcome 字段完全一致。输出
明确声明 TEST 不参与 epoch、checkpoint、模型或策略选择，Evaluation 保持关闭，并拒绝
覆盖既有 runtime config 或 run directory。

本次仅实现和验证 gate，没有调用 preparer，没有创建 `/mnt` runtime config 或
run directory，更没有运行或读取 Development TEST。focused tests 与既有
three-seed config tests 合并执行为 14/14 通过。该任务没有参数更新，因此不向
中央训练尝试表增加伪 attempt；最近记录的 100 个唯一 attempts/4 个 Critic V2
RUNNING 状态不因本任务改变。

两个 V2 gate 将来真实 PASS 后，唯一允许的配置准备命令为：

```bash
$PY scripts/route_a_v3/prepare_route2_mrnabert_critic_v2_frozen_test_config_v1.py \
  --selected-confirmation-config /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/mrnabert_critic_v2/runtime_configs/task_study_macro_confirmation_seeds_v1/seed20260823.json \
  --control-protocol configs/route_a_v3_route2_mrnabert_critic_v2_protocol_v1.json \
  --confirmation-protocol configs/route_a_v3_route2_mrnabert_critic_v2_three_seed_protocol_v1.json \
  --frozen-test-protocol configs/route_a_v3_route2_mrnabert_critic_v2_frozen_test_protocol_v1.json \
  --control-adjudication /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/comparisons/mrnabert_critic_v2_task_study_macro_controls_adjudication_v1.json \
  --confirmation-adjudication /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/comparisons/mrnabert_critic_v2_task_study_macro_three_seed_adjudication_v1.json \
  --gpu "$GPU_INDEX"
```

其中 `GPU_INDEX` 必须在执行时设置为 GPU0-5 中显存足够的一张卡。这条命令只准备
配置，不执行训练。当前 control/confirmation gate 未知，因此不得
提前运行该命令，也不得从旧 V1 TEST→guided scheduler 绕过它。

## 20. Critic V2 all-126,165 Development refit 门前瞻冻结（2026-08-20）

在 Critic V2 three-seed 与单次 TEST outcome 均不存在时，已前瞻冻结独立的
all-Development refit protocol 和 V2-only preparer。合同规定单次 TEST 后按固定
结构/loss/policy refit，却没有规定再用 TEST 数值设一个新阈值；同时明确禁止按
TEST 重选结构、loss、seed、epoch 或 threshold。因此 preparer 要求合法、完整、
provenance-matched 的 TEST summary 和 metrics 字段存在，但完全不读取指标值做分支。

refit 固定 seed `20260823`、全部 126,165 条 Development records、100 epochs、
`FINAL_EPOCH`、同一 Critic V2 model/loss/sampling/aggregation/scaling policy 和
GPU0-5。它要求 TEST 训练确实使用 TRAIN+VALIDATION 107,873 条并只在 TEST 上
评估 18,292 条，CUDA 参数发生更新，Evaluation read=0；任何 stage、seed、
baseline identity、policy、record counts 或 checkpoint provenance 漂移都会拒绝。
输出声明 TEST metrics 未用于 refit selection，Evaluation/guidance 仍关闭，并拒绝
覆盖既有 runtime config 或 run directory。

本次没有调用真实 preparer，没有读取真实 TEST summary，也没有创建或运行 refit。
新门 focused tests 13/13 通过；与 TEST gate、three-seed config 和训练器 split 合并
验证为 81 passed、4 个本机无 CUDA 的既有 skips。该任务没有参数更新，因此中央
训练 CSV 不新增伪 attempt。

仅在单次 V2 TEST 合法 terminal 后，允许准备 refit config：

```bash
$PY scripts/route_a_v3/prepare_route2_mrnabert_critic_v2_all_development_refit_config_v1.py \
  --frozen-test-config /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/mrnabert_critic_v2/runtime_configs/single_frozen_development_test_v1/seed20260823.json \
  --frozen-test-summary /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/mrnabert_critic_v2/single_frozen_development_test_v1/seed20260823/training_summary.json \
  --frozen-test-protocol configs/route_a_v3_route2_mrnabert_critic_v2_frozen_test_protocol_v1.json \
  --refit-protocol configs/route_a_v3_route2_mrnabert_critic_v2_all_development_refit_protocol_v1.json \
  --gpu "$GPU_INDEX"
```

这条命令只写唯一 runtime config，不启动 refit；当前上游 gate 未 terminal，不得
提前调用。

## 21. Critic V2 primary TEST-preserving LOSO 门前瞻冻结（2026-08-20）

已在 Critic V2 outcome 不存在时冻结 V2-only primary LOSO protocol/preparer。
历史 V1 preparer 只依赖旧 three-seed PASS，不能证明单次 TEST 与 all-Development
refit 已按顺序 terminal；新门必须先验证 exact V2 refit config 和 terminal summary，
因此不能绕过 `TEST → refit → LOSO`。

LOSO cohort 固定为 7 个非空 Development studies × seeds
`20260822/20260823/20260824`，共 21 runs；study-major/seed-minor 按共享 schedule
round-robin 分配 GPU0-5。每折只使用原始 TRAIN/VALIDATION，held-out study 作为
assessment fold，跨入 holdout 的 connected source components 从训练中排除；原始
Development TEST 18,292 条在每折继续 withheld。模型、Huber、task→study→source
fixed draws、task-macro loss、robust scaling、batch16、100 epochs 均与 Critic V2
冻结 policy 一致。checkpoint 固定为 `FINAL_EPOCH`，避免用 held-out study 选择
checkpoint。

preparer 还要求 refit 使用全部 126,165 条、100-epoch final checkpoint、CUDA
参数更新且 Evaluation read=0。输出的每个 LOSO config 明确记录 earlier single
TEST 已发生但其 metrics 未用于 LOSO selection；当前 LOSO run 自身的 TEST/Evaluation
access 均为 false，guided generation 仍未授权。runtime config root 或任何 run
target 已存在时均拒绝覆盖。

本次没有调用真实 preparer、没有创建 21 个 `/mnt` runtime configs，也没有运行
LOSO 或读取任何 protected outcome。新门 focused tests 12/12 通过；包含共享六 GPU
配对和 trainer split 的扩展 suite 为 70 passed、4 个本机无 CUDA 的既有 skips。
本任务没有参数更新，中央训练 CSV 不新增伪 attempt。

仅在 exact V2 all-Development refit terminal 后，允许一次准备 21 个 configs：

```bash
$PY scripts/route_a_v3/prepare_route2_mrnabert_critic_v2_test_preserving_loso_configs_v1.py \
  --refit-config /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/mrnabert_critic_v2/runtime_configs/all_development_refit_v1/seed20260823.json \
  --refit-summary /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/mrnabert_critic_v2/all_development_refit_v1/seed20260823/training_summary.json \
  --refit-protocol configs/route_a_v3_route2_mrnabert_critic_v2_all_development_refit_protocol_v1.json \
  --loso-protocol configs/route_a_v3_route2_mrnabert_critic_v2_test_preserving_loso_protocol_v1.json
```

该命令只准备 configs，不调度 LOSO；当前上游 gate 未 terminal，不得提前运行。

## 22. Critic V2 matched strongest-baseline LOSO 门前瞻冻结（2026-08-20）

已冻结 V2-only matched-baseline LOSO protocol/preparer。历史 baseline preparer
只检查旧 three-seed PASS 布尔值，不能证明新 primary LOSO 已经由 TEST/refit 顺序
合法生成，也不能验证逐 fold 配对；新门以未来 21 份 primary runtime configs 为
只读输入，要求每个 `(holdout study, seed)` 唯一且完整，并逐一匹配相同 physical
GPU、TEST-preserving split 与数据边界。

matched 的含义固定为同一 study/seed/GPU/fold，而不是虚构相同模型容量或训练
预算。baseline 保留使其成为 strongest same-information comparator 的 native
policy：anchored position-aware antisymmetric model、transferable context、
task→source weighting、TRAIN-task robust scaling、pairwise-Huber、batch32、8 epochs、
FP32。held-out study 不用于 checkpoint 选择，因此 LOSO checkpoint 前瞻固定为
`FINAL_EPOCH`。每个 baseline config 交叉记录 paired primary identity/path，且
Development TEST 18,292 条、TEST metrics selection、Evaluation 和 guidance 均关闭。

本次没有读取 primary outcome，没有调用真实 preparer，没有生成 baseline runtime
configs 或启动训练。新门 focused tests 15/15 通过；包含共享 primary/baseline
pairing、历史 baseline config 和 LOSO aggregation provenance 的扩展 suite 为
27/27 通过。该任务没有参数更新，中央训练 CSV 不新增伪 attempt。

仅在合法的 21 个 primary configs 已生成后，允许一次准备配对 baseline configs：

```bash
$PY scripts/route_a_v3/prepare_route2_mrnabert_critic_v2_matched_baseline_loso_configs_v1.py \
  --base-config configs/route_a_v3_route2_method_repair_global_scaled_seed20260821_gpu0_v1.json \
  --primary-config-root /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/mrnabert_critic_v2/runtime_configs/test_preserving_loso_v1 \
  --primary-protocol configs/route_a_v3_route2_mrnabert_critic_v2_test_preserving_loso_protocol_v1.json \
  --baseline-protocol configs/route_a_v3_route2_mrnabert_critic_v2_matched_baseline_loso_protocol_v1.json
```

该命令只准备 configs，不运行 baseline LOSO；当前上游 gate 未 terminal，不得提前
调用。

## 23. Critic V2 guidance readiness 总闸门前瞻冻结（2026-08-20）

已新增独立的 V2-only readiness protocol、证据包 builder 与 adjudicator，替代只
认识历史 V1 signal-control 的旧 readiness 路径。新总闸门把整条已冻结因果链绑定
为：Critic V2 control PASS → 三个固定 seed PASS → 单次冻结 Development TEST
合法完成 → all-126,165 refit 合法完成 → 三个固定 seed 各自完成 7-study primary /
matched-baseline LOSO 聚合且 macro Spearman improvement 全部大于 0。LOSO 聚合器
已经逐 run 验证 21 个 primary 与 21 个 matched-baseline provenance，因此 readiness
只接受三个完整聚合结果，不重复读取或重新解释 42 份训练 summary。

单次 TEST 在此只要求合法完成、指标字段存在且没有被用于结构、loss、seed、epoch、
threshold 或 policy 选择；TEST 指标值本身完全不作为 readiness 阈值。最终 guidance
解锁仍须同时满足 `CRITIC_READY_FOR_GUIDANCE` 与 `FLOW_G0_READY`，并要求冻结的
reward policy、online frozen mRNABERT、final refit checkpoint、Base Flow checkpoint
和 Evaluation-unused provenance 全部成立。readiness 裁决不会运行 guidance，也不把
Flow G0 工程合法性或 critic readiness 写成 biological optimization claim。

本次只完成前瞻冻结和实现：没有调用真实 builder/adjudicator，没有读取 TEST 数值、
新的 final Evaluation outcome 或 LOSO outcome，没有创建 `/mnt` readiness packet，
也没有授权 guided generation。focused tests 14/14 通过；与历史 readiness builder、
adjudicator 和共享 LOSO aggregation 合并的回归为 41/41 通过。该任务没有参数更新，
中央训练 CSV 不新增伪 attempt；最近记录的 100 个唯一 attempts/四个 Critic V2
RUNNING 状态不因本任务改变。

未来只有在上述所有真实前置产物均 terminal 后，才允许依次调用以下两个 V2-only
入口一次：

```text
scripts/route_a_v3/build_route2_mrnabert_critic_v2_guidance_readiness_input_v1.py
scripts/route_a_v3/adjudicate_route2_mrnabert_critic_v2_readiness_v1.py
```

builder 的 required CLI 参数须逐一绑定八份冻结协议、control/three-seed/TEST/refit、
三个 LOSO 聚合、reward、encoder 与 Flow 的 exact terminal artifacts；输出位置由
readiness protocol 固定，且不得覆盖。当前上游 gate 未 terminal，因此不得提前构建
packet 或裁决，也不得回退到旧 V1 readiness。

## 24. Route 2 V3.3.2 paper packet 的 Critic V2 readiness 绑定（2026-08-20）

等待 Critic V2 control terminal 期间，Methods/results evidence draft 已补入完整的
V2 前瞻顺序：control → exact three seeds → single report-only TEST → all-126,165
refit → three positive matched LOSO aggregations → critic/Flow dual readiness。该段只
描述已冻结方法，不添加任何 Critic V2 result；draft 仍明确写为 control RUNNING、
Development TEST closed、guided XEditFlow unauthorized 和 final Evaluation closed。

evidence manifest 新增 V2 readiness protocol-chain source，consistency manifest
新增对应 prospective method；当前 packet 为 14 个唯一 claim markers、13 个唯一
evidence sources，所有 draft/manifest 引用闭合。focused consistency tests 2/2
通过，并验证 seed、single-TEST report-only policy、LOSO study count 与双 readiness
条件同冻结协议完全一致。该论文任务无参数更新，不向中央训练 CSV 增加伪 attempt；
最近记录的 100 个唯一 attempts/四个 Critic V2 RUNNING 状态不因本任务改变。

## 25. Critic V2 guided/matched/comparison 执行链硬切换（2026-08-20）

下游审计发现一个真实阻塞：历史 guided runner/config、matched-search runner/config
和 generation-comparison config 均绑定旧 V1 readiness、旧 refit checkpoint、旧
guided method 或旧 candidate 目录。新 V2 readiness 即使未来 PASS，也会被旧入口
拒绝或在后续消费者处断链。

当前执行链已直接硬切换到 V2，不保留双 schema 兼容分支。三个历史 config 均标记
为 `RETIRED_*_NOT_AUTHORIZED` 并指向各自 V2 replacement；guided 与 matched-search
runner 会在读取任何 readiness artifact 前拒绝旧 config。新 V2 configs 精确绑定：

- `mrnabert_critic_v2_guidance_readiness_{input,adjudication}_v1.json`；
- V2 all-Development refit 的 seed-20260823 final checkpoint；
- V2-specific guided output 与逐 source compute；
- V2-specific matched-search candidate 目录；
- `frozen_mrnabert_critic_v2_guided_xeditflow_v1` method identity。

生成算法、mean-potential reward、legal `SUB + STOP`、fixed-seed replay、逐 source
forward-equivalent matching、六种 search 方法、独立 evaluator 和 paired-bootstrap
Development 比较规则均未改变。matched search 仍只生成 candidates，不在该 stage
选择 strongest method；comparison 仍不构成 measured biological 或 external
Evaluation success。

本次只冻结并验证入口，没有构建真实 readiness packet，没有运行 guided/matched/
comparison，没有创建 `/mnt` candidates/output，也没有读取 TEST、LOSO 或新的 final
Evaluation outcome。三入口 focused suite 16/16 通过；定向检查确认 V2
configs/runners/tests 中不含旧 V1 readiness/refit/method/candidate bindings。本任务
没有参数更新，中央训练 CSV 不新增伪 attempt；最近记录的 100 个唯一 attempts/
四个 Critic V2 RUNNING 状态不因本任务改变。

未来三个 V2-only 配置依次为：

```text
configs/route_a_v3_route2_mrnabert_critic_v2_guided_xeditflow_development_gpu0_v1.json
configs/route_a_v3_route2_mrnabert_critic_v2_matched_search_development_gpu0_v1.json
configs/route_a_v3_route2_mrnabert_critic_v2_generation_comparison_development_gpu0_v1.json
```

只有 V2 readiness 真实 `guided_unlocked=true` 后，才可动态选择 GPU0-5 并从第一项
开始；后两项分别还须等待前一项 terminal。当前不得调用任何一个入口。

## 26. Critic V2 readiness-to-comparison 合成端到端合同验证（2026-08-20）

此前 readiness、guided、matched-search 与 comparison 各自的 focused tests 使用
独立夹具，不能单独证明真实上游输出字段可由下游连续消费。现已增加一项合成端到端
contract test：真实 readiness builder 组装完整 V2 packet，真实 adjudicator 产生
dual-ready 裁决，再依次送入 production guided config/validator、matched-search
consumer 和 comparison config boundary。测试还统一核对 guided output method ID 与
comparison 所需 method ID，避免两端字符串漂移。

该测试只使用合成 terminal evidence 和临时 checkpoint；为检查 production path
contract，仅把 packet 中 checkpoint path 绑定到已冻结的未来 V2 config，不读取或
创建任何真实 `/mnt` outcome。新的 readiness focused suite 为 15/15 通过；下游链、
paper 与 readiness 合并回归为 33/33 通过。未运行 TEST/refit/LOSO/generation，未
打开 Evaluation，也没有参数更新，因此中央训练 CSV 不增加伪 attempt；最近记录的
100 个唯一 attempts/四个 Critic V2 RUNNING 状态不因本任务改变。

## 27. Critic V2 三 seed LOSO aggregation-input 门前瞻冻结（2026-08-20）

审计确认通用 LOSO aggregator 本身可以验证 V2 primary/baseline terminal training
provenance，但历史三 seed input builder 硬编码旧 V1 的 `seed..._huber_v1` 与
`seed..._global_scaled_v1` 路径，无法读取 V2 preparers 生成的 21+21 run identities。

现已冻结独立 V2 aggregation protocol，并实现 V2-only input builder。它读取两侧
runtime config roots 各 21 份 JSON，以 config 的 `output_directory` 为未来 terminal
summary 路径依据，按 `(study, seed)` 核对 exact physical GPU、primary baseline ID、
paired primary output、LOSO stage/split、TEST-preserving 与 Evaluation-unused 字段。
每个 terminal summary 的 `validation_metrics` 被包装成 `LOSO::<study>` evaluation，
交给既有通用 aggregator 统一处理 study 对齐、undefined Spearman 和 macro
improvement；不复制统计实现。

聚合协议固定 seeds `20260822/20260823/20260824`、7 个非空 Development studies、
`GSE256185` 为 zero-record study，以及 V2-specific input/result roots。readiness
protocol/builder/adjudicator 已显式绑定该协议，不能再由旧 V1 input builder 提供结果。

本次没有调用真实 builder/aggregator，没有读取 LOSO summary，没有创建 `/mnt`
inputs/results。新 builder + shared aggregator + readiness focused suite 26/26 通过；
三个合成 seed 均得到 7-study aligned complete，且 protected boundary 保持关闭。
paper evidence manifest 新增这份 prospective protocol，当前为 14 个 evidence
sources，claim 数仍为 14。该任务没有参数更新，中央训练 CSV 不新增伪 attempt；
最近记录的 100 个唯一 attempts/四个 Critic V2 RUNNING 状态不因本任务改变。

只有 42 个 LOSO runs 全部合法 terminal 后，才允许先执行一次：

```bash
$PY scripts/route_a_v3/build_route2_mrnabert_critic_v2_loso_aggregation_inputs_v1.py \
  --primary-protocol configs/route_a_v3_route2_mrnabert_critic_v2_test_preserving_loso_protocol_v1.json \
  --baseline-protocol configs/route_a_v3_route2_mrnabert_critic_v2_matched_baseline_loso_protocol_v1.json \
  --aggregation-protocol configs/route_a_v3_route2_mrnabert_critic_v2_loso_aggregation_protocol_v1.json
```

随后才可对三个固定 input 各调用一次共享 `aggregate_route2_loso_v1.py`，输出到协议
冻结的 V2 aggregation result root。当前上游 gate 未 terminal，不得调用。

## 28. Critic V2 paired LOSO 六 GPU stage runner 前瞻实现（2026-08-20）

在补齐 aggregation input 后继续审计发现：V2 尚无可执行 launcher，唯一旧 scheduler
仍硬编码 V1 config names、run roots 和 aggregation builder。现已新增 V2-only LOSO
stage runner，但没有启动。

runner 在任何 log/run 创建前读取并验证两侧各 21 份 runtime configs 与 V2 三份
protocol，构造冻结的 6 个 physical-GPU queues。每个 `(study, seed)` 先在指定 GPU
完成 Critic V2 primary，再在同一 GPU 完成 exact matched baseline；每张 GPU 内串行，
GPU0-5 之间并行。每次启动前只检查 assigned GPU 的 free memory，默认阈值 4096 MiB、
poll 900 秒，不使用 utilization gate。任何 worker 失败都会保留现有 evidence 并阻止
aggregation；只有 42 runs 全部成功后，runner 才调用 V2 input builder 和共享
aggregator 生成三个固定 seed results，然后停止，不自动进入 readiness/guidance。

runner 拒绝已有 primary/baseline run、log、aggregation input 或 result root，避免
重复 terminal 或覆盖 partial evidence。focused tests 8/8 通过，覆盖 21 unique
pairs、六 GPU queues、V2 config filenames、primary-before-baseline、existing-root
拒绝和 all-training-before-aggregation 顺序。该测试没有启动 subprocess/GPU。

本次没有创建 `/mnt` stage artifact、没有运行 LOSO、没有读取 TEST/LOSO/Evaluation
outcome，也没有参数更新或中央训练 CSV 新行。最近记录状态不因本任务改变。

未来只有合法 21+21 configs 已准备且所有对应 run roots 均不存在时，允许人工显式
调用一次：

```bash
$PY scripts/route_a_v3/run_route2_mrnabert_critic_v2_loso_stage_v1.py \
  --primary-protocol configs/route_a_v3_route2_mrnabert_critic_v2_test_preserving_loso_protocol_v1.json \
  --baseline-protocol configs/route_a_v3_route2_mrnabert_critic_v2_matched_baseline_loso_protocol_v1.json \
  --aggregation-protocol configs/route_a_v3_route2_mrnabert_critic_v2_loso_aggregation_protocol_v1.json
```

当前 TEST/refit/config gates 未 terminal，因此不得调用。

## 29. Critic V2 Development generation 单 GPU stage runner 前瞻实现（2026-08-20）

下游 V2 guided、matched-search 与 Development comparison 虽已完成硬切换，但三个
入口仍各自固定 `cuda:0`，且没有在 GPU0-5 中按实时 free memory 选择可用卡并保持
严格阶段顺序的统一 launcher。现已新增 V2-only Development generation stage
runner，但没有启动。

显式调用后，runner 首先读取 V2 readiness input/adjudication，并由 production
guided validator 验证真实 dual readiness；校验未通过时，在 runtime config、log
或 output 创建前停止。只有 readiness 通过后，才在 physical GPU0-5 中选择 free
memory 最大且不少于 4096 MiB 的卡。三份 runtime config 只把冻结模板中的 `device`
和 `physical_gpu_index` 改为所选 GPU，随后依次运行 guided XEditFlow、六方法 matched
search 和 frozen independent-evaluator Development comparison。每个 child 启动前仅按
900 秒间隔等待同一卡恢复 minimum free memory，不使用 utilization gate；任一 child
失败即保留已有 evidence 并阻止后续阶段。

runner 拒绝覆盖已有 runtime root、log root 或三个 stage 的 output directory。其完成
状态仍明确 `development_test_opened=false`、`evaluation_opened=false`、generated
candidates 无 canonical credit，且不建立 biological optimization claim。

本次没有调用 runner、没有查询远端 GPU、没有创建 `/mnt` runtime/log/candidates/
comparison artifacts，也没有读取 TEST、LOSO 或 Evaluation outcome。focused tests
4/4 通过；与 V2 readiness、guided、matched-search、comparison 和 paper evidence
合并的相邻合同回归为 37/37 通过。该任务没有参数更新，中央训练 CSV 不新增伪
attempt；最近记录状态不变。

未来只有真实 V2 readiness 已输出 `guided_unlocked=true` 且三个 stage 尚未开始时，
才允许人工显式调用一次：

```bash
$PY scripts/route_a_v3/run_route2_mrnabert_critic_v2_development_generation_stage_v1.py
```

当前 control 及其下游 gates 未 terminal，因此不得调用。

## 30. Critic V2 post-confirmation 全链 runner 与条件 watcher（2026-08-20）

继续审计可执行衔接后确认一个真实缺口：Critic V2 three-seed watcher 在 PASS 后只
输出“single frozen TEST authorized not started”并退出；TEST、refit、LOSO、readiness
和 generation 虽各有冻结组件，但没有一个严格按 V3.3.2 顺序连接它们的 V2-only
入口。同时，历史 `schedule_route2_mrnabert_postselection_controls_v1.sh` 仍保留旧
V1 controls/TEST/refit/LOSO/guided 可执行路径。该历史 scheduler 现已在入口立即报
retired 并退出，不提供兼容分支。

新 post-confirmation runner 在任何 runtime/log/output 写入前，先由 production
frozen-TEST builder 验证 control PASS、exact three-seed PASS、seed-20260823
confirmation config 和完整八协议绑定，然后一次检查从 TEST 到 generation 的 19 个
future targets 均未开始。唯一执行顺序固定为：

```text
single report-only Development TEST
  -> all-126,165 Development refit
  -> prepare 21 primary + 21 exactly matched baseline LOSO configs
  -> paired six-GPU LOSO + three fixed-seed aggregations
  -> Critic/Flow readiness input + adjudication
  -> Development guided/matched/comparison only if guided_unlocked=true
```

TEST 与 refit 各自在 GPU0-5 中选择 free memory 最大且不少于 4096 MiB 的卡，等待
只按 900 秒 free-memory poll，不使用 utilization。LOSO 与 generation 复用各自已冻结
runner。TEST 指标只作为 terminal report 进入既有 refit/readiness schema，不触发
结构、loss、seed、epoch、threshold 或 policy 分支；readiness NO-GO 时 generation
不启动。全链始终保持 final Evaluation closed，generated candidates 无 canonical
credit，也不建立 biological optimization claim。

另新增一份 900 秒 conditional watcher：three-seed adjudication 不存在时只等待；
NO-GO 时退出且不创建 stage artifacts；PASS 时才调用上述 runner。代码推送并在
A100 通过同组 103/103 回归后，该 watcher 已于 22:05 启动一份，实际脚本 PID
`380389`（launch wrapper PID `380388`），日志为
`/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/schedulers/mrnabert_critic_v2_post_confirmation_v1.log`。
首条日志是 `waiting_for_critic_v2_three_seed_adjudication`；启动验收确认 19 个 downstream
targets 全部不存在。

本次没有调用 post-confirmation runner、没有查询新 GPU、没有打开 TEST，也没有创建
stage runtime/log/run/LOSO/readiness/candidate/comparison artifact；新增的 `/mnt` 文件
只有 scheduler 等待日志。focused tests 7/7，完整相邻生产合同回归在本机和 A100
均为 103/103。没有参数更新，中央训练 CSV 不新增伪 attempt；21:54 最近低频状态
仍为 100 个唯一 attempts/四个 Critic V2 RUNNING，control adjudication absent。

已使用的唯一 watcher 启动命令为：

```bash
nohup scripts/route_a_v3/schedule_route2_mrnabert_critic_v2_post_confirmation_v1.sh \
  >/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/schedulers/mrnabert_critic_v2_post_confirmation_v1.log \
  2>&1 </dev/null &
```

watcher 的存在不授权 TEST；唯一授权仍来自真实 three-seed adjudication PASS。

## 31. Development generation bootstrap 统计报告补全（2026-08-20）

等待下一次 control 低频窗口期间，只读复核了已经 terminal 的 frozen strongest-
generation-baseline artifact；没有访问 Development TEST 或 final Evaluation。原 paper
draft 已说明 genetic 是唯一 bootstrap uncertainty-equivalent 方法，但未完整报告
analysis unit、bootstrap seed 和六个 leader-advantage intervals。

现新增全精度 `route2_v332_generation_bootstrap_table_v1.csv`，固定记录 source-level
paired bootstrap 的 891 个 analysis units、seed `20260816`、10,000 iterations 和六个
竞争方法的 point advantage/95% interval。所有 10,000 次均定义，六个 interval 的
lower bound 均大于 0；最近竞争者 generate-then-rerank 的 genetic advantage 为
`0.040228719771844945`，95% interval 为
`[0.018934459870160632, 0.06168808431503615]`。该结果只支持 frozen Development
independent-evaluator separation，不提供 measured biological 或 external validation。

Methods/results draft、evidence manifest 和 consistency manifest 已同步；当前 packet
为 15 个唯一 claim markers、14 个唯一 evidence sources。paper focused tests 3/3，
与 frozen selection producer/input 合并回归 17/17 通过。本任务没有参数更新，不向
中央训练 CSV 增加伪 attempt；Critic V2 运行状态不因本论文任务改变。

## 32. Independent evaluator 九 task 异质性报告补全（2026-08-20）

继续只读复核已经 terminal 的 Development independent-evaluator training summary
和 adjudication；Development TEST 与 final Evaluation 未访问。原 draft 只给出
task-macro Spearman `0.10256553571558498`、standardized MAE
`1.8078550850617527` 和 5/9 positive，尚未展开九个 task-region groups。

现新增全精度 `route2_v332_independent_evaluator_task_table_v1.csv`。九行 record count
合计 18,293；task Spearman 范围为 `-0.10916458562634956` 至
`0.7619576378536184`，其中 5 个正、4 个非正。`RNA_HALF_LIFE_MINUTES::region=0`
的 standardized MAE 为 `9.220029033415157`，是九组最大值。因此 paper 现明确：
evaluator narrowly passing macro gate 与 task-level heterogeneity 同时成立，不能把
qualification 写成均匀跨 task 的可靠性或 biological validation。

evidence manifest locator 也改为 terminal artifact 的精确字段：training summary 的
`validation_metrics.task_metrics`/`evaluation_outcomes_read=0`，以及 adjudication 的
`development_test_outcomes_accessed=false`、`evaluation_outcomes_accessed=false`。
当前 packet 为 16 个唯一 claim markers、14 个 evidence sources。paper focused tests
4/4，与 evaluator scorer/adjudicator 合并回归 10/10 通过。本任务没有参数更新，中央
训练 CSV 不增加伪 attempt；Critic V2 运行状态不因本论文任务改变。

## 33. Independent evaluator global spread 限制与冻结 gate 核对（2026-08-20）

九-task 表补全后进一步核对 terminal evaluator summary 与 prospective qualification
protocol。raw global `prediction_std=1.6834847809909463`、
`target_std=764.2945302862793`，ratio 为 `0.0022026649600126917`，是必须保留的
compression diagnostic。

冻结 qualification gate 只包含 task-macro Spearman exclusive threshold、minimum
positive-task breadth、model independence、CUDA completion 和 protected-outcome
closure；没有 prediction-spread threshold。因此不允许事后添加阈值或撤销已经合法
terminal 的 qualification。另一方面，九个 endpoint 原始尺度高度异质，terminal
summary 没有持久化 per-task prediction/target spread，故 global ratio 也不能单独
证明每个 task 都 mean-collapse。paper 现同时报告该 diagnostic 与不可定位到 task
的限制，并记录 `PER_TASK_SPREAD_NOT_RECORDED_NO_TERMINAL_RERUN`。

evidence manifest 新增 exact prospective qualification protocol，当前 packet 为
17 个唯一 claim markers、15 个 evidence sources。paper focused tests 5/5，与
evaluator scorer/adjudicator 合并回归 11/11 通过。没有读取 Development TEST/final
Evaluation、没有修改 gate 或参数，因此中央训练 CSV 不增加 attempt，Critic V2
运行状态不因本任务改变。

## 34. Independent evaluator qualification checks 补充表（2026-08-20）

只读提取 terminal independent-evaluator adjudication 的 exact `checks`，新增
`route2_v332_independent_evaluator_qualification_checks_v1.csv`。表中 12 项均为 true：
run completion、frozen Development Validation、TEST withheld、Evaluation closed、
CUDA update、与 guiding critic 的 architecture/feature independence、TRAIN-only
task-robust scaling、全部 task defined、超过 exact source-permutation threshold、
positive-task breadth 和 exact frozen evaluator identity。

adjudication 的 `candidate_rerun_authorized=true` 只解释为已完成的 Development
candidate rerun 授权；同一 artifact 的 `scientific_claim_status=NOT_ESTABLISHED`
继续保留，不能写成 biological success。当前 paper packet 为 18 个唯一 claim
markers、15 个 evidence sources。paper focused tests 6/6，与 evaluator
scorer/adjudicator 合并回归 12/12 通过。本任务没有读取 Development TEST/final
Evaluation、没有参数更新或新训练行，Critic V2 状态不因本任务改变。

## 35. Critic V2 post-confirmation 生产只读输入 preflight（2026-08-20）

按低频监控要求，等待 control terminal 期间没有查询 epoch/validation 进度，而是对
post-confirmation 唯一 runner 的生产只读输入做一次存在性 preflight。检查范围为八份
冻结 Critic V2 协议、reward/baseline configs、三个执行入口、三份 Development
generation templates、冻结 Development manifest、八份 canonical records、已 terminal
的 strongest baseline/online encoder/Flow readiness inputs 与 Flow checkpoint，共
30 个路径；结果为 30/30 存在、missing=0。

路径枚举来自本地冻结协议和 runner 源码；远端核查只使用文件存在性元数据，没有
打开生产数据或 outcome 文件内容、没有读取训练进度、Development TEST 或 final
Evaluation outcome，也没有创建 future runtime/run/candidate artifact。
因此不修改冻结协议、数据或 runner；focused static-input preflight 状态为 `PASS`，
action 为 `NO_PATH_REPAIR_REQUIRED`。本任务没有参数更新，中央训练 CSV 不新增伪
attempt；Critic V2 运行状态不因本任务改变。

## 36. Paper evidence source locator 闭合核查（2026-08-20）

等待 control terminal 期间，对 `route2_v332_evidence_manifest_v1.json` 的 15 个
source locator 做一次存在性核查：本地仓库或合同路径 8/8、A100 `/mnt` 路径 7/7，
合计 15/15 存在、missing=0。核查只看文件存在性，不打开 evidence 内容、训练进度、
Development TEST 或 final Evaluation outcome。

evidence manifest 新增 `source_path_preflight`，明确该 PASS 只表示 locator 闭合，
不表示 human content verification 已完成，也不改变 submission readiness；
`human_verification_required=true` 与 `submission_ready=false` 均保留。paper focused
tests 现为 7/7。本任务没有参数更新，中央训练 CSV 不新增伪 attempt，Critic V2
运行状态不因本任务改变。

## 37. Critic V2 control terminal NO-GO 与后继链关闭（2026-08-22）

Critic V2 四臂 screen 已全部完成，每臂均为 100 epochs、559,900 optimizer updates，
中央训练表四行都由训练器原位更新为 `COMPLETED`。full 的 task-macro Spearman 为
`0.11637066318689378`，虽高于 candidate permutation `0.08018546242383856`、
parameter-matched source-only `0.017976235482461158` 和 source+edit-metadata
`0.08655782657012488`，也通过两个 permutation-supported tasks 与 task-breadth
要求，但未超过冻结 strongest same-information baseline `0.13171439492559175`；
margin 为 `-0.015343731738697977`。唯一前瞻裁决因此为
`CRITIC_V2_CONTROLS_DO_NOT_SUPPORT_THREE_FROZEN_SEEDS`，
`supports_three_frozen_seeds=false`，scientific claim 继续为 `NOT_ESTABLISHED`。

按冻结 gate，seed `20260822/20260823/20260824` 均未启动，three-seed runtime config
数为 0、three-seed adjudication 不存在。生产 runner 列出的 19 个 TEST→generation
future targets 在 terminal 后逐项检查均不存在；Development TEST、all-Development
refit、TEST-preserving LOSO、readiness、guided generation 与新的 final Evaluation
全部保持关闭。不得补第四个 seed、改阈值、回退旧 TEST→guided 顺序或重复运行已
terminal 的四臂 screen。

终态同时暴露一个真实工程缺口：post-confirmation watcher 原先只等待 three-seed
adjudication，在 control NO-GO 后会永久空等。提交 `990f941` 让 watcher 在每轮先读
control adjudication，NO-GO 时以 0 退出且不创建 downstream artifact。focused tests
在本机和 A100 均为 8/8；旧 waiter PID `380389` 已精确终止，修复后脚本写入
`critic_v2_control_gate_terminal_no_go_post_confirmation_not_started` 并正常退出。

中央训练表当前为 100 个唯一 attempts：92 completed、3 failed、3 incomplete、
1 stopped for throughput repair、1 stopped for priority reallocation。完整终态证据保存于
`audits/route_a_v3_route2_critic_v2_control_terminal_no_go_v1.json`；没有读取
Development TEST 或 final Evaluation outcome。

## 38. Critic V2 数据/任务 failure geometry（2026-08-22）

按主合同对 terminal predictor failure 的要求，只读比较四臂 Critic V2 与冻结
strongest same-information baseline 的 Development VALIDATION task metrics。九 task
共 18,293 records，单 task 数量从 48 到 12,048，相差 251 倍。full 在冻结的两个
candidate-permutation eligible tasks 上分别取得 `+0.10750582874835171` 与
`+0.13788530735618243` Spearman margin，且有 7/9 positive tasks、全局
prediction/target spread ratio `0.09643439549490583`；因此不能把失败简化成“完全没有
candidate-specific signal”或单一 mean collapse。

真正未满足的是跨 task 的 strongest-baseline superiority 与 calibration。full 相对
strongest baseline 只赢 4/9 task、输 5/9，九-task mean Spearman margin 为
`-0.01534373173869797`；standardized MAE 在 9/9 task 都更差，macro margin 为
`+0.4161680105385127`（越低越好）。两个各 48-record tasks 的 Spearman margin 合计
`-0.21149352683453532`；仅作 post hoc geometry diagnostic 排除它们时，其余七 task
mean margin 为 `+0.010485705883750515`。该敏感性不能用于删除 task、重定义 gate 或
授权新 seed，只支持“局部 candidate signal 被异质 task/低样本几何与跨任务校准不足
抵消”的受限诊断。

全精度九行表与解释边界保存于
`docs/paper/route2_v332_critic_v2_task_diagnostic_table_v1.csv` 和
`audits/route_a_v3_route2_critic_v2_task_failure_diagnostic_v1.json`。forward route 固定为
Benchmark+historical transfer/generation limits+negative result+data/action-space geometry；
不启动 TEST、更多 seed、guided XEditFlow 或 final Evaluation。本任务没有参数更新，
中央训练 CSV 不增加 attempt。

## 39. 七法 Development generation action-space geometry（2026-08-22）

沿 Critic V2 terminal NO-GO 后的合同 forward route，只读聚合 terminal v2
matched-generation selection input；没有读取 Development TEST 或 new final Evaluation，
也没有打开 guided stage。七法均使用 891-source cohort、`SUB + STOP`、candidate cap
32、critic cap 256 和 total forward-equivalent cap 320。candidate-count、edit-distance、
terminal-cause 三组守恒检查均为 7/7，hard legality 全为 1.0，edit/candidate budget
violation、`NO_LEGAL_ACTION`、`NUMERICAL_FAILURE` 全为 0。

终态 geometry 显示：observed source-relative edit distance 为 0--5；zero-edit 来自合法
immediate STOP，不表示 INS/DEL。greedy/beam explicit-STOP rate 最高，均为
`0.7048260381593715`；unguided Base Flow budget-exhaustion rate 最高，为
`0.8702651515151515`。local search 未强制填满 candidate cap，每 source 3--32 个、
均值 `23.5993265993266`；Flow 28,512 rows 中有 25,173 unique、3,339 duplicates，
其余六法 within-source duplicates 为 0。

所有方法的 support mode 均为 `OPEN_GENERATED_SUPPORT`，closed measured NDCG defined
source count 均为 0；因此 independent-evaluator uplift、measured-neighborhood recovery、
stopping、edit depth、uniqueness 和 compute 继续分列，未知候选不赋 zero gain 或
canonical intervention credit。per-candidate algorithmic STOP time 与六 search 方法
generation wall time未被 terminal input 保留，明确记为缺口而不重跑或用时间戳反推。

新增 `route2_v332_generation_action_space_geometry_table_v1.csv`、
`route_a_v3_route2_generation_action_space_geometry_v1.json`，paper packet 现为 20 个唯一
claim markers、17 个 evidence sources。本机 focused suite 为 39 passed、5 skipped、
0 failed；A100 完整环境同组为 44/44 passed。本任务不新增中央训练 attempt，100-row
终态分布不变；forward claim 保持 Benchmark+limits+negative result+data/action-space
geometry。

## 40. Historical transfer 与 18-item minimum benchmark package（2026-08-22）

按主合同的降级路线，仅复核既有、已 outcome-exposed 的 GSE232572 historical
zero-shot summary；未读取 Development TEST 或新的 final Evaluation。8,068 records、
三 frozen seeds 相对 neural-medium Siamese strongest baseline 的 task-macro Spearman
点差均为正，但 seed 20260816 的 paired CI 跨零；三 seed 的 MAE 全部显著劣于 baseline。
终态 `preregistered_pass=false`，只能报告 historical negative/null transfer，不能写成
independent final confirmation。

新增 `route2_v332_minimum_benchmark_package_table_v1.csv` 与
`route_a_v3_route2_v332_minimum_benchmark_package_v1.json`，逐项映射合同的 18 个最低包
要求：13 complete/complete-with-declared-limits、4 partial、1 unavailable。五个当前
blockers 为 MBP-10 guided/first-order comparison NO-GO、MBP-13 replacement Evaluation
缺失、MBP-14 final zero-shot/adaptation 未发生、MBP-15 terminal timing 字段缺失、
MBP-17 figure builders 未建立。因此状态固定为
`MINIMUM_BENCHMARK_PACKAGE_NOT_COMPLETE`、`submission_ready=false`；Outcome C 只是
conditional target route，不能由旧 GSE summary 的 provisional label 自动冻结。

paper packet 已同步为 22 个唯一 claim markers、19 个 evidence sources，并显式记录
GSE232572 historical role 与旧 inventory role/delivery/running-state 字段由 V3.3.2
authority 覆盖。本机与 A100 inventory/split/baseline/E-MTAB/paper focused suite 均为
30/30 passed。本任务不运行训练、guided、E-MTAB outcome 或新 final Evaluation，
不新增中央 attempt，100-row 终态不变。

## 41. Route 2 V3.3.2 provisional manuscript figure builders（2026-08-22）

新增 `scripts/route_a_v3/build_route2_v332_manuscript_figures_v1.py` 与 focused
tests，仅消费冻结的 generation/action-space table、完整九任务 Critic V2 diagnostic
table 和已 outcome-exposed 的 GSE232572 historical summary。构建器输出两张通用稿件
图，各含 PNG/PDF/SVG；另写 provenance manifest 与 alt text，并显式记录未读取
Development TEST/new final Evaluation、未运行 guided XEditFlow。

A100 已从 GitHub commit `a27e04a` 快进同步并在
`/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/figures/route2_v332_v1/` 一次性生成
8 个产物。A100 builder focused test 为 2/2；同步审计记录后，A100 与本机
paper-packet + builder 联合 focused suite 均为 14/14。视觉复核已修正图例遮挡和
标题/轴标签裁切；格式核查确认 PNG 为
2160-px wide、300-dpi、全像素不透明，PDF 单页且字体嵌入，SVG 无 raster image。
target journal/article type/submission phase 尚未选择，故
`publisher_compliance_claimed=false`。

MBP-17 更新为 `COMPLETE_WITH_PROVISIONAL_GENERAL_FIGURES`，18-item 汇总变为
14 complete/complete-with-declared-limits、3 partial、1 unavailable；总体仍是
`MINIMUM_BENCHMARK_PACKAGE_NOT_COMPLETE`。四个 blocker 保持 MBP-10 guided NO-GO、
MBP-13 replacement Evaluation 缺失、MBP-14 final zero-shot/adaptation 未执行、
MBP-15 terminal timing 缺失。此任务不训练或更新参数，不新增中央 attempt；100-row
终态仍为 92 COMPLETED、3 FAILED、3 INCOMPLETE_NO_TERMINAL_RECORD、1
STOPPED_FOR_THROUGHPUT_REPAIR、1 STOPPED_PRIORITY_REALLOCATION。

## 42. Route 2 V3.3.2 dataset qualification/development table（2026-08-22）

新增 `build_route2_v332_dataset_qualification_table_v1.py`、14-row paper CSV、审计
JSON 与 focused tests。构建器只读取冻结的 14-study inventory，不打开 canonical rows、
Development TEST、新 final Evaluation outcome、E-MTAB outcome 或 sealed GSE246381。
冻结 inventory 的 GSE232572 `EVALUATION` 字段由 V3.3.2 authority 显式规范化为
`HISTORICAL_OUTCOME_EXPOSED_TRANSFER_DIAGNOSTIC_NOT_FINAL_CONFIRMATION`；旧 YAML
`AUDIT_PENDING` registry 只作历史 lineage，不覆盖当前 terminal inventory。

表内 8 个 Development study units 合计 126,165 records；历史 GSE232572 为 8,068；
新 outcome-unexposed final Evaluation 为 0。只有 GSE200304 贡献 qualified credit：
6,547 records、ordinary/A1/true-A2=`1/1/0`。Development-relaxed/listwise rows 不增加
qualified credit，6 个 zero-record studies 保留 unconvertible/auxiliary/aggregate-only/
sealed 原因，generated candidate canonical credit 为 0。paper packet 保持 22 个 claim
markers，evidence sources 从 21 增至 23；本机与 A100 dataset-table + paper-packet +
figure-builder 联合 focused suite 均为 17/17。

本任务不训练或更新参数，不新增中央 attempt；100-row 终态仍为 92 COMPLETED、3
FAILED、3 INCOMPLETE_NO_TERMINAL_RECORD、1 STOPPED_FOR_THROUGHPUT_REPAIR、1
STOPPED_PRIORITY_REALLOCATION。guided、Development TEST 与 new final Evaluation 继续关闭。

## 43. Canonical conversion flow figure（2026-08-22）

新增独立 conversion-flow builder 与 focused tests，从已冻结的 14-row dataset
qualification table 和 method-repair split protocol 构建两面板通用稿件图。Panel A
分离 8 个 Development study units/126,165 records、1 个 outcome-exposed historical
study/8,068 records，以及 5 个 zero-canonical-record terminal roles；Panel B 分离
qualified 6,547、Development-relaxed 88,652、listwise 30,966、unconvertible 0，并显示
TRAIN/VALIDATION/TEST-withheld=`89,580/18,293/18,292`。箭头只编码 workflow，不按混合的
study/record 单位缩放；generated candidate credit 与 new final Evaluation records 均为 0。

A100 commit `647de6f` builder focused test 为 2/2，并在
`/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/figures/route2_v332_v1/` 追加 PNG/PDF/SVG、
独立 manifest 与 alt text，未覆盖原两图。视觉复核修正标题裁切/重叠；PNG 为
2160×1860、300 dpi、全像素不透明，PDF 字体嵌入，SVG 无 raster image，主轮廓色对白底
均超过 3:1。target journal/article type/submission phase 待定，继续明确
`publisher_compliance_claimed=false`。

paper packet 仍为 22 个 claim markers，evidence sources 从 23 增至 25；本机与 A100
四组联合 focused suite 均为 19/19。manuscript figures 现为 3 张、2 个 builders；MBP-17 status 和
最低包 14/3/1 汇总不变，四个 blocker 仍为 MBP-10/13/14/15。本任务不训练或更新参数，
不新增中央 attempt；100-row terminal 分布和 protected-outcome closure 不变。

## 44. Development/Evaluation architecture figure（2026-08-22）

新增独立 Development/Evaluation architecture builder 与 focused tests，仅消费冻结的
14-row dataset qualification table、method-repair/readiness protocols 和 minimum package
audit。Panel A 固定 Development 126,165 与 TRAIN/VALIDATION/TEST-withheld=
`89,580/18,293/18,292`，区分 fitting、selection、Critic V2 current NO-GO 与 conditional
single report-only TEST；Panel B 区分 historical GSE232572、conversion-failure-only
E-MTAB-10902、sealed GSE246381 和尚不存在的 replacement Evaluation。

A100 已从 GitHub commit `3ccccef` 快进同步，builder focused test 2/2 passed，并在
`/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/figures/route2_v332_v1/` 追加 PNG/PDF/SVG、
manifest 与 alt text，未覆盖已有产物。视觉复核修正条件标签位置和底部说明裁切；PNG
为 2160×2040、300 dpi、全像素不透明，PDF 单页且 DejaVu 字体嵌入，SVG 无 raster
image，主轮廓色对白底均超过 3:1。target journal/article type/submission phase 待定，
继续明确 `publisher_compliance_claimed=false`。

paper packet evidence sources 从 25 增至 27，figure count 为 4、builder count 为 3；
22 个 claim markers、MBP-17、最低包 14/3/1 与四个 blockers 均不变。本机与 A100
五组联合 focused suite 均为 21/21。本任务不训练或更新参数，不新增中央 attempt；
Development TEST、new final Evaluation、E-MTAB outcome、sealed GSE246381 和 guided
XEditFlow 均未打开。

## 45. Predictor–Legal XEditFlow–Independent Evaluator architecture figure（2026-08-22）

新增独立系统架构 builder 与 focused tests。证据输入限于冻结的小型组件配置、资格/奖励
协议和 terminal summary audits；显式拒绝旧 generation-readiness audit 的陈旧 Base Flow
参数。Panel A 分离 frozen mRNABERT Delta critic、position/progress legal `SUB+STOP` Base
Flow、distinct Siamese evaluator；Panel B 分离 prospective frozen-critic potential、已执行的
independent post-generation scoring 与尚不可用的 measured outcome，并标记全部无梯度边界。

A100 从 GitHub commit `6798b6c` 快进，builder focused test 2/2 passed，在
`/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/figures/route2_v332_v1/` 追加三格式、manifest
与 alt text。视觉复核修复了旋转标签遮挡和 evaluator→measured outcome 的误导连线；PNG
2160×2040、300 dpi、全不透明，PDF 单页且 DejaVu 嵌入，SVG 无 raster，所有主轮廓色
对白底超过 3:1。`publisher_compliance_claimed=false`。

paper evidence sources 从 27 增至 29，figure count=5、builder count=4；22 个 claims、
MBP-17、最低包 14/3/1、四个 blockers 和 `submission_ready=false` 不变。本机与 A100
六组联合 focused suite 均为 23/23。Development TEST、new final Evaluation、E-MTAB
outcome、sealed GSE246381 与 guided XEditFlow 均未打开。

## 46. V3.3.2 Prediction/Generation baseline inventory matrix（2026-08-22）

按合同 baseline-matrix 顺序新增独立 builder、focused tests、45-row paper CSV 与审计。
该产物是覆盖/状态矩阵而非结果表：Prediction 34 行覆盖 11 internal controls、6 classical、
7 neural 和 10 task-specific/foundation references；Generation 11 行覆盖 random、
exhaustive-small-space、greedy、beam、genetic、local search、generate-N-rerank、unguided
Flow、first-order guidance、frozen-critic XEditFlow 与 masked discrete flow/diffusion。

矩阵不把近似名称当成完成证据：anchored CNN 绑定已冻结选择的
`delta_anchored_position_aware_antisymmetric`；ordinal/listwise 仅有配置而没有独立 terminal
selection；study mean 是明确受限的 composite mapping；absolute-candidate predictor 只作为
absolute-difference pipeline component。旧 baseline inventory 的 mRNABERT-running 和
generation-preterminal 标签分别由 Critic V2 terminal NO-GO 与七方法 terminal table 覆盖。
first-order/frozen-critic guided rows保持 `NOT_RUN_CRITIC_V2_NO_GO`。

GitHub commit `589d263` 已推送，A100 快进同步后 builder test 2/2 passed；paper
integration commit `4734eae` 后，本机与 A100 七组联合 focused suite 均为 25/25。
paper evidence sources=31（local/contract 19、A100 `/mnt` 12），
claim markers=22、figures/builders=5/4、最低包=14/3/1、blockers=MBP-10/13/14/15、
`submission_ready=false` 均保持。此任务不训练或更新参数，不新增中央 attempt；
Development TEST、new final Evaluation、E-MTAB outcome、sealed GSE246381、generated
candidates 与 guided XEditFlow 均未打开。

## 47. V3.3.2 NATIVE / COMMON / ARCH_CONTROLLED 三轨结果表（2026-08-22）

按 Goal 7 顺序新增 terminal-input snapshot、独立 builder、focused tests、52-row CSV 与
audit。表格严格拆分三轨：Track A 10 行全部为 status-only，native numeric=0；Track B
12 行中 9 行有 Development 数值、8 行可在相同 task scope 内进入 headline 横向比较；
Track C 30 行中 26 行有 Development 数值，剩余为 guided NO-GO 或未匹配因果对照。

表中纳入六个 external common-task adapters、Critic V2 full/strongest same-information
baseline/legacy reference、absolute/candidate controls、六个 neural HPO architecture rows、
sequence/context/region/A1/single-study/study-scale/study-balance ablations、四个 Critic V2
controls和七方法 generation suite。明确禁止跨 5′UTR/3′UTR/nine-task scope 排名；aligned
A1 direct result 未 materialize；scratch-vs-frozen 和 generic-trunk-vs-region-adapter 不能
由当前 terminal runs 作因果归因。

GitHub commit `e2a9b63` 已推送，A100 builder focused test 2/2 passed；paper integration
commit `99136f6` 后，本机与 A100 八组联合 focused suite 均为 27/27。paper evidence
sources=34（local/contract 22、A100 `/mnt` 12），
claim markers=22、figures/builders=5/4、最低包=14/3/1、blockers=MBP-10/13/14/15、
`submission_ready=false` 均保持。`reporting_table_complete=true` 但
`three_track_benchmark_execution_complete=false`。本任务不新增中央 attempt，也未打开
Development TEST、new final Evaluation、E-MTAB outcome、sealed GSE246381、generated
candidates 或 guided XEditFlow。

## 48. V3.3.2 frozen Development learning curves（2026-08-22）

按 Goal 7 顺序新增独立 learning-curve builder 与 focused tests，不训练、不监控实时进度，
只读取已经 terminal 的 Development histories。四面板分别为六个 selected predictor
profiles（各 8 epochs）、Critic V2 四个 control arms（各 100 epochs）、independent
evaluator（8 epochs）和 Base Flow G0（30 epochs）。所有曲线均按原始 epoch 顺序绘制，
没有平滑、插值或删点。

预测器 per-epoch history 只有 pooled Validation Spearman，最终架构标签则来自独立计算的
Development Validation task-macro Spearman；图和 manifest 明确禁止把两者混成同一指标或
跨 panel 排名。Critic V2 full selected epoch 98 的 task-macro Spearman 为
`0.11637066318689378`，仍低于 same-information hurdle `0.13171439492559175`，NO-GO
不变。evaluator final epoch 为 `0.10256553571558498`，仅跨过 Development method-selection
threshold `0.1012475745988908`。Base Flow selected epoch 1 Validation NLL 为
`5.512483521877043`，epoch 30 为 `9.939703254814608`，图中保留 train 降、validation
恶化的 engineering overfitting pattern；不声称 biological optimization。

GitHub commits `f28d04f` 与 `9659da7` 已推送；A100 两次 builder focused test 均为 2/2，
终态 PNG/PDF/SVG、manifest 与 alt text 保存于
`/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/figures/route2_v332_v1/`。视觉与格式审计确认
PNG `2400×2520`、约 300 dpi、全不透明，PDF 字体嵌入且无 raster image，SVG 无 raster
image；`publisher_compliance_claimed=false`。paper integration commit `f94372b` 后，
evidence sources=36（local/contract 23、A100 `/mnt` 13），claim markers=22、
figures/builders=6/5；本机与 A100 九组联合 focused suite 均为 29/29，最低包 14/3/1、
blockers=MBP-10/13/14/15 与
`submission_ready=false` 不变。此非训练任务不新增中央 attempt，100-row 的 92/3/3/1/1
终态分布不变；Development TEST、new final Evaluation 与 guided XEditFlow 未打开。

## 49. V3.3.2 A1 numeric-task / true-A2 availability and result-boundary table（2026-08-22）

按 Goal 7 顺序先完成交叉核查：已有 Figure 2 已覆盖 GSE232572 outcome-exposed historical
zero-shot transfer，且明确不作 final confirmation，因此不重复生成；replacement Evaluation
study 不存在，few-shot adaptation 依法不可用。当前最靠前且可真实完成的缺项是 A1 与
true-A2 task results 的估计量分离表。

新增独立 builder、focused tests、14-row CSV 与 audit。9 个 A1 Development Validation
numeric task rows 共 18,293 records，task Spearman 范围为
`[-0.10916458562634956, 0.7619576378536184]`，5/9 为正；这只用于 Development
generation-method selection。5 个 true-A2 rows 全部为 availability/result-boundary：
GSE269595 有 30,966 Development-exposed listwise records 但 qualified true-A2 study
credit=0；evaluator implementation complete，listwise ranker 仅 configured-not-terminal；
open generated support 下七方法 closed measured NDCG defined-source count=0；新 independent
Evaluation unexposed records=0。因此 terminal true-A2 numeric performance rows=0，缺失值保持
空白，不以 0 performance 代替，也不作 A1/true-A2 cross-estimand numeric ranking。

GitHub builder commit `363c741` 已推送，A100 快进后 builder focused test 2/2 passed；
paper integration commit `410053d` 推送并同步后，本机与 A100 十组 paper/table/figure
联合 focused suite 均为 32/32。paper evidence sources=38
（local/contract 25、A100 `/mnt` 13），claim markers=22、figures/builders=6/5、最低包
14/3/1、blockers=MBP-10/13/14/15、`submission_ready=false` 均保持。本任务不训练、不监控
实时训练进度、不新增中央 attempt；100-row 的 92/3/3/1/1 终态分布不变，Development TEST、
new final Evaluation、generated-candidate outcome 与 guided XEditFlow 均未打开。

## 50. V3.3.2 Prediction/Generation matched-budget baseline matrix（2026-08-22）

先按 Goal 7 做 terminal crosswalk：既有 predictor–XEditFlow–evaluator architecture figure
已呈现 `SUB+STOP`、INS/DEL out-of-scope、1/3/5 budgets、legal mask、position/progress Base
Flow、`FLOW_G0_READY` 与 guided NOT RUN；既有 generation Figure 1 已呈现 explicit STOP
versus budget-exhaustion fractions。因此 legal action/STOP-budget/base-guided method-figure
项由两张既有 terminal 图共同闭合，不重复生成。

下一真实缺口新增只读 terminal compute snapshot、独立 builder、focused tests、14-row ×
47-column CSV 与 audit。Prediction 5 行中，Critic V2 full/candidate-permutation/source-only/
source-edit-metadata 四臂为 exact within-screen matched：同 seed 20260825、100 epochs、
559,900 updates、9,342,914 trainable + 113,389,056 frozen parameters、同 grouped split；
strongest same-information hurdle 虽是冻结 gate comparator，但只有 22,120 updates，因此明确
标为 `SAME_INFORMATION_HURDLE_NOT_UPDATE_BUDGET_MATCHED_TO_CRITIC_V2`。full-minus-hurdle
task-macro Spearman 仍为 `-0.015343731738697977`。

Generation 9 行包含七个 terminal 方法和两个 guided NO-GO。七方法共享 891 sources、
`SUB+STOP`、1/3/5 edit budgets、candidate cap 32、critic-forward cap 256 和 total-forward
cap 320；六方法达到 28,512 candidates，local search 为 21,027。algorithm-specific
training/HPO 不是共同 numeric budget，六个 search generation wall times 仍为
`NOT_RECORDED_NO_TERMINAL_RERUN`，first-order 与 frozen-critic guidance 仍因 Critic V2
NO-GO 不运行。因此 `reporting_matrix_complete=true`，但
`matched_budget_benchmark_execution_complete=false`、fully contract-matched headline rows=0。

GitHub builder commit `11d3ec0` 已推送，A100 focused test 2/2 passed；paper integration
commit `a4d4f64` 推送并同步后，本机与 A100 十一组联合 focused suite 均为 35/35。
paper evidence sources=41（local/contract 28、A100 `/mnt` 13），
claim markers=22、figures/builders=6/5、MBP=14/3/1、blockers=MBP-10/13/14/15 与
`submission_ready=false` 不变。本任务不训练、不新增中央 attempt；100-row 的 92/3/3/1/1
终态分布不变，Development TEST、new final Evaluation、generated-candidate outcome 与
guided XEditFlow 均未打开。

## 51. V3.3.2 Generation critic / independent / measured 三层结果表（2026-08-22）

按 Goal 7 顺序完成 measured / independent / critic-only generation result table。该任务
不训练、不轮询 GPU 或训练进度，只对既有 terminal selection input 做一次只读聚合字段核查；
没有展开 generated candidate payload，没有读取 Development TEST、新 final Evaluation 或
guided outcome。版本化输入为
`audits/route_a_v3_route2_v332_generation_three_layer_terminal_snapshot_v1.json`；新增 builder、
2 项 focused tests、9-row × 31-column CSV 与独立 audit。Spreadsheet artifact 检查确认
6 个 critic numeric rows、7 个 independent rows、7 个 measured-recovery rows、2 个 guided
NO-GO rows；9 行 closed measured NDCG 均为空，guided 数值字段也全部为空。

六个 critic-driven 方法的 guiding-critic self-score 均覆盖 891/891 sources；unguided Base
Flow 的 critic 层为 `NOT_APPLICABLE_NO_CRITIC_CALLS`。genetic 同时领先 critic max uplift
`1.1912207428186161` 与 independent-evaluator uplift `1.0978248587628674`，但 measured
candidate recovery 由 unguided Base Flow 以 `0.20286195286195285` 领先，genetic 为
`0.05443322109988777`。conditional recovered measured NDCG 仅覆盖 11--400 sources，且七
方法 closed measured NDCG defined-source count 均为 0；因此禁止 self-score/independent
替代 measured outcome，也不建立跨层 numeric ranking 或 biological improvement claim。

GitHub commits `adcd9d4`/`3af0ac0` 已推送；A100 一次性同步到 `3af0ac0` 后十二组联合
focused suite 38/38，本机同为 38/38。paper evidence sources=44（31 local/contract、13
A100 `/mnt`），claim markers=22、figures/builders=6/5、MBP=14/3/1、
blockers=MBP-10/13/14/15、`submission_ready=false` 与中央 100-row 的 92/3/3/1/1 终态
均不变。first-order 与 frozen-critic guided generation 继续遵守 Critic V2 terminal NO-GO。

## 52. V3.3.2 Generation diversity / quality–cost / failure analysis figure（2026-08-22）

按 Goal 7 顺序交叉核查后发现真实图形缺口：既有 Figure 1 没有 performance-versus-cost
坐标，也没有独立 Hamming-diversity panel。新增
`build_route2_v332_generation_quality_cost_diversity_failure_figure_v1.py` 与 2 项 focused
tests，只读取冻结的 7-method action-space geometry table/audit，不训练、不轮询、不展开
generated candidates，不读取 Development TEST/new final Evaluation/guided outcome。

四面板分别为 independent-evaluator uplift–forward-equivalent cost、sparse measured
recovery–cost、Hamming diversity + unique rate，以及 duplicate/candidate-cap-shortfall
failure geometry。point-estimate Pareto front 分别为 random/Flow/genetic 与 random/Flow；
Base Flow 同时显示最高 recovery/diversity 和 0.117109 duplicate fraction，local search 显示
0.262521 cap shortfall。所有 legality/budget/no-legal/numerical failure 为 0。没有 per-method
uncertainty、六个 search wall time 或 closed measured NDCG，guided rows 不伪装成 executed。

GitHub commits `559952c`/`e7af043` 已推送；A100 正式 PNG/PDF/SVG/manifest/alt text 位于
`/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/figures/route2_v332_v1/`。视觉与文件检查确认
PNG 2160×2100、约 300 dpi、全不透明，PDF 无 raster image resource 且字体嵌入，SVG 无
raster image；`publisher_compliance_claimed=false`。本机/A100 十三组 focused suite 均为
41/41；evidence sources=46（32 local/contract、14 A100 `/mnt`）、claims=22、
figures/builders=7/6、MBP=14/3/1、blockers=MBP-10/13/14/15、`submission_ready=false` 和
中央 100-row 的 92/3/3/1/1 终态不变。

## 53. V3.3.2 Error / domain-shift analysis（2026-08-22）

按 Goal 7 顺序交叉核查后确认真实缺口：Critic V2 九任务 control diagnostic、independent
evaluator 九任务 error table 与 GSE232572 outcome-exposed historical transfer 分散在不同
artifact，尚无统一的 region/assay/context boundary table。新增
`build_route2_v332_error_domain_shift_analysis_table_v1.py`、2 项 focused tests、12-row ×
41-column CSV 与 audit；不训练、不轮询 GPU/训练进度，不读取 Development TEST/new final
Evaluation/guided outcome。

表格严格拆成 9 个 Development task-region rows 和 3 个 historical seed rows。Development
部分覆盖 7 studies/7 assays、18,293 Validation records；Critic V2 task-level Spearman
相对 strongest same-information baseline 为 4 win/5 loss，standardized MAE 为 0 win/9
loss。task size 为 48--12,048；剔除两个 n=48 task 的七任务 margin 只保留为 post hoc
geometry，不替换 frozen gate。independent evaluator 为 5/9 positive tasks，Spearman 范围
[-0.10916458562634956, 0.7619576378536184]。

5′UTR 的 4-task descriptive Critic margin/evaluator macro Spearman 为
-0.031473208428386926 / 0.02852402190396145；3′UTR 的 5-task 对应值为
-0.0024401503869468017 / 0.16179874676488382。两者被 study、assay、context、endpoint、
task size 混杂，不作 region effect；terminal task aggregates 也不能识别 within-assay
context error。三条 GSE232572 seed 保持历史 outcome-exposed 状态，rank CI 仅 2/3 排除
零、MAE 三条均 favor baseline，不能作 final confirmation。Spreadsheet artifact 验证为
12×41，跨层缺失字段保持空白，12 行 external confirmation 均 false。

GitHub commit `39ce66d` 已推送；A100 自 `e7af043` 一次 fast-forward 到 `39ce66d`，本机/
A100 十四组联合 focused suite 均为 44/44。paper evidence sources=48（34 local/contract、
14 A100 `/mnt`）、claims=22、figures/builders=7/6、MBP=14/3/1、
blockers=MBP-10/13/14/15、`submission_ready=false` 与中央 100-row 92/3/3/1/1 均不变。

## 54. V3.3.2 Minimum benchmark package itemwise closure（2026-08-22）

按 Goal 7 顺序核对 18-row minimum benchmark package table 与现有 audit。任务不新增训练、
不轮询 GPU/训练进度、不读取 Development TEST/new final Evaluation/guided outcome。新增明确
状态分离：`itemwise_adjudication_complete=true` 表示 18/18 requirement IDs 完整唯一、
unadjudicated=0、unfinished-as-PASS=false；`minimum_package_complete=false` 则继续保留
14 complete / 3 partial / 1 unavailable 的真实包状态。

四个 blockers 不变：MBP-10 是 Critic V2 NO-GO 依赖，MBP-13/14 是 replacement study 与
其 zero-shot/adaptation 缺失，MBP-15 是不能重建且不能为补 timing 重跑 terminal suite 的
instrumentation gap。当前授权动作为 `NO_RERUN_NO_GUIDED_NO_PROTECTED_OUTCOME_READ`；这使
“最低包逐项闭合”任务诚实终结，但不解锁 submission-ready 或 final paper outcome。

GitHub commit `e6607f6` 已推送；A100 自 `39ce66d` 一次 fast-forward 到 `e6607f6`，本机/
A100 paper-evidence focused suite 均为 18/18。evidence sources=48、claims=22、
figures/builders=7/6、MBP=14/3/1、blockers=MBP-10/13/14/15、`submission_ready=false` 与
中央 100-row 92/3/3/1/1 均不变。

## 56. V3.3.2 Selected-outcome claim/evidence closure（2026-08-22）

按 Goal 7 顺序新增 selected Benchmark+limits claim-evidence 交付。任务不训练、不轮询 GPU/
训练进度、不读取 Development TEST/new final Evaluation/guided outcome。builder 自动解析
manuscript 中 22 个唯一 `C-R2-*` markers、对应段落与 evidence IDs，并用固定科学边界映射
生成表格；unmapped marker、duplicate ID 或 unregistered evidence 均为 hard failure。

正式 CSV 为 35 rows × 17 columns：22 rows supported-with-declared-boundary，13 rows
unsupported。后者显式冻结 A/B headlines、biological/guided success、external generation
validation、equal realized candidate count、INS/DEL、missing-as-zero、historical final
confirmation、package/submission completeness、unread E-MTAB outcome 与 causal region/context
mechanism 为不允许表述。所有 unsupported rows 的 allowed=false，minimum package/outcome
trigger/submission-ready 及 protected-outcome fields 也全部 false。

Spreadsheet artifact check 为 35×17，blank evidence=0、unsupported allowed=0。GitHub commit
`799e156` 已推送；A100 自 `e211212` 一次 fast-forward 到 `799e156`，本机/A100 claim+
paper focused suite 均为 22/22。evidence sources=51（37 local/contract、14 `/mnt`）、
claims=22、figures/builders=7/6、MBP=14/3/1、blockers=MBP-10/13/14/15、中央 100-row
92/3/3/1/1 均不变。

## 55. V3.3.2 Final manuscript-route outcome adjudication（2026-08-22）

按 Goal 7 顺序逐条核对合同三种 outcome。任务不训练、不轮询 GPU/训练进度、不读取
Development TEST/new final Evaluation/guided outcome。A 因 minimum package incomplete、
无 outcome-unexposed external confirmation、Critic V2 NO-GO、guided not run 与无 true-A2/
guided improvement 而 ineligible；B 因无 stable outcome-unexposed external prediction value、
guided comparison 不完整与 package incomplete 而 ineligible。

新增 `route_a_v3_route2_v332_paper_outcome_adjudication_v1.json`，将唯一 forward manuscript
route 冻结为 `BENCHMARK_PLUS_TRANSFER_AND_GENERATION_LIMITS_PAPER`。严格分开 route 与
eligibility：`final_paper_outcome_frozen=true`，但 `outcome_trigger_fully_satisfied=false`、
`submission_level_outcome_eligibility=false`、`submission_ready=false`。Outcome C 的负面
transfer/generation、controls/geometry/error 条件已支持；minimum package trigger 仍 false。

论文 Discussion 同步加入五类 next-data requirement：unambiguous source/candidate lineage、
closed dense measured pool、biological replicate/finite-positive SE、explicit frozen balanced
assay/context/endpoint/region，以及 new convertible outcome-unexposed external study。未知 generated
candidates 仍不赋零 gain，zero-shot 必须永久先于 adaptation。

GitHub commit `e211212` 已推送；A100 自 `e6607f6` 一次 fast-forward 到 `e211212`，本机/
A100 paper-evidence focused suite 均为 19/19。evidence sources=49（35 local/contract、14
`/mnt`）、claims=22、figures/builders=7/6、MBP=14/3/1、blockers=MBP-10/13/14/15、
中央 100-row 92/3/3/1/1 均不变。

## 57. V3.3.2 Data / rights / exposure limitations closure（2026-08-22）

本项无训练命令、无 GPU 轮询、无新 attempt。只读现有 14-study qualification table、paper
evidence confidentiality/human-verification boundary，以及 11 个当前 converter/preflight rights
locators；没有打开 canonical rows、Development TEST、new final Evaluation、sealed payload、
generated candidates 或 guided outcome。

新增 `build_route2_v332_data_rights_exposure_limitations_table_v1.py`、2 项 focused tests、
14-row × 22-column CSV 与独立 audit。表中保留 1 true / 8 false / 5 not-Boolean 的 operational
redistribution declarations，但 converter/preflight declaration 明确不等于 license verification；
14/14 license rows 保持 human-review pending，0/14 public release authorized。GSE217518 的
converter `true` 不能升级为 dataset license；GSE256185 的 raw/derived row-level redistribution
均保留 NOT_AUTHORIZED；GSE207584/GSE261709 的 aggregate/member rights 限制也逐项登记。

Spreadsheet import/inspect/formula scan 与三段 visual render 覆盖全部 15×22 cells；本机与
A100 全部 V3.3.2 paper/table/figure suite 均为 51/51。GitHub core commit `fe3fb6b` 已推送，
A100 自 `799e156` 一次 fast-forward。evidence sources=53（39 local/contract、14 `/mnt`），
claims=22、figures/builders=7/6、MBP=14/3/1、blockers=MBP-10/13/14/15、
`minimum_package_complete=false`、`submission_ready=false` 与中央 100-row 的 92/3/3/1/1
终态均不变。

## 58. V3.3.2 Methods section completion（2026-08-22）

本项无训练命令、GPU 轮询或新 attempt。Methods 从 draft 状态闭合为
`COMPLETE_INTERNAL_HUMAN_VERIFICATION_PENDING`，共 14 个 subsection；只使用现有合同、配置、
terminal summaries 的已登记字段与派生表，不读取 protected outcomes。

补齐 evaluator 的 depth/width/max length、Huber/weighting/scaling、AdamW/LR/WD/batch/BF16/seed，
generation 的 beam/genetic/oversampling/Base Flow 训练设置，以及 Critic V2 loss/optimizer/
checkpoint policy；新增统一 statistical-analysis 段，明确 task-macro/source-macro、standardized
MAE、paired source bootstrap、跨 evidence layer 禁止混排和 unknown outcome 不补零。配置预计
evaluator parameters=509,905、terminal actual=509,845 的 -60 差异进入 audit，终态实际值优先。

修正 figures/builders 陈旧文案为 7/6；claim/evidence 表从当前 manuscript 显式重建且保持
35 rows、22 supported markers、13 unsupported claims。GitHub core commit `7ae4e57` 已推送，
A100 自 `fe3fb6b` 快进；本机/A100 全部 V3.3.2 suite 均为 54/54。evidence=54
（40 local/contract、14 `/mnt`），MBP=14/3/1、blockers=MBP-10/13/14/15、中央 100-row
92/3/3/1/1、`minimum_package_complete=false` 与 `submission_ready=false` 均不变。

## 59. V3.3.2 Results section completion（2026-08-22）

本项无训练命令、GPU 轮询或新 attempt。Results 从 evidence draft 闭合为
`COMPLETE_INTERNAL_HUMAN_VERIFICATION_PENDING`，共 15 个 subsection；只改变 section 状态并
新增 completion audit/focused tests，没有打开或生成任何新 result outcome。

audit 固定现有 Development/negative evidence：baseline rows=45、three-track rows=52、
A1 tasks=9、true-A2 terminal numeric rows=0、fully contract-matched headline rows=0、generation
methods=7 且 legality=1.0、Critic V1 positive-margin seeds=1/3、Critic V2 status 为
`CRITIC_V2_CONTROLS_DO_NOT_SUPPORT_THREE_FROZEN_SEEDS`、historical transfer pass=false、
new outcome-unexposed Evaluation records=0。package 仍 14 complete-or-limited / 3 partial /
1 unavailable，blockers=MBP-10/13/14/15。

GitHub core commit `86e63bf` 已推送，A100 自 `7ae4e57` 快进；本机/A100 全部 V3.3.2 suite
均为 57/57。evidence=55（41 local/contract、14 `/mnt`），claims=22、figures/builders=7/6、
中央 100-row 92/3/3/1/1、`minimum_package_complete=false`、
`outcome_trigger_fully_satisfied=false` 与 `submission_ready=false` 均不变。

## 60. V3.3.2 Discussion section completion（2026-08-22）

本项无训练命令、GPU 轮询或新 attempt。Discussion 从单一 next-data 草稿闭合为
`COMPLETE_INTERNAL_HUMAN_VERIFICATION_PENDING`，共 5 个 subsection；只读取既有合同、paper
tables 和 terminal audits 的已登记字段，不打开 protected outcomes。

新增解释层明确：当前贡献是 comparison-boundary benchmark，而非 predictor/generator success；
Critic V2 的 localized candidate-specific rank signal 不等于 task-wide baseline/calibration
superiority；genetic 的独立 evaluator point lead 与 Base Flow 的 sparse measured-recovery point
lead 属于不同、非 biological 的开放支持端点；task/region summaries 被 study、assay、context、
endpoint 和 task size 混杂；GSE232572 仍为 historically outcome-exposed negative transfer
diagnostic。confirmatory cohort 需要 dense closed measured pool、replicate-level uncertainty、明确
context/region 和新的 outcome-unexposed Evaluation。

新增 completion audit 与 3 项 Discussion focused tests，并同步 Results section boundary、
consistency/evidence manifests；claim markers 保持 22。GitHub core commit `e6e807a` 已推送，A100
自 `86e63bf` 快进。A100 第一次文件选择使用了远端不存在的 `rg`，空选择使 pytest 误收集旧全
测试树并产生 39 个无关 collection errors；该次运行无效，不代表 V3.3.2 回归。改用远端已有的
精确文件查找后，本机/A100 V3.3.2 suite 均为 60/60。evidence=56（42 local/contract、14
`/mnt`），MBP=14/3/1、blockers=MBP-10/13/14/15、中央 100-row 92/3/3/1/1、model/
biological/external/guided success=false、`minimum_package_complete=false`、
`outcome_trigger_fully_satisfied=false` 与 `submission_ready=false` 均不变。

## 61. V3.3.2 Data Availability section completion（2026-08-22）

本项无训练命令、GPU 轮询或新 attempt。新增 `Data Availability` section、completion audit 与
3 项 focused tests；同步 consistency/evidence manifests 和未决清单，只读取已有 data-rights audit、
qualification table、合同 storage boundary 与 package 状态，不打开任何 protected outcome。

statement 记录第三方 accession/locator 但不替代 current access、redistribution 或 reuse authority；
当前 14-study inventory 没有 accountable-human-verified study-bound license，14/14 rights rows
pending、0/14 public payload release authorized。Git 中的小型 aggregate tables/manifests/audits 仍是
internal version-controlled evidence packet，不声明永久 archive；大型 canonical data、run products、
checkpoints、weights、generated candidates 和运行产物保持在 `/mnt/.../route2/`，不声明 public
release。没有 DOI/persistent identifier、availability-on-request promise 或未获授权的未来开放承诺。

GitHub core commit `2faff4a` 已推送，A100 自 `e6e807a` 快进；本机/A100 V3.3.2 suite 均为
63/63。evidence=57（43 local/contract、14 `/mnt`），claims=22、MBP=14/3/1、
blockers=MBP-10/13/14/15、中央 100-row 92/3/3/1/1、public release/right review/
stable repository/submission readiness=false 或 pending，`minimum_package_complete=false` 与
`outcome_trigger_fully_satisfied=false` 均不变。

## 62. V3.3.2 Code Availability section completion（2026-08-22）

本项无训练命令、GPU 轮询或新 attempt。新增 `Code Availability` section、completion audit 与
3 项 focused tests；同步 consistency/evidence manifests 和未决清单。只读 Git remote/branch/tag、
tracked environment descriptors 和 license/README 状态，不打开任何实验 outcome。

statement 将 `https://github.com/Cunyu-Liu/mRNA_editflow` 与
`route-a-v3-route2-method-repair-20260817` 写成 working repository locator/branch，不声称
unauthenticated public access。当前无 Route 2 V3.3.2 tag、persistent archive identifier 或 archived
container；`pyproject.toml`、`requirements-lock.txt`、`environment.yml`、`Dockerfile` 存在，但
未完成 Route 2 clean-environment reproduction/human verification。README 仍指向早期 v2 authority；
package metadata=`Proprietary`，无 standalone `LICENSE`。large artifacts 仍在 `/mnt/.../route2/`
且不属于 code release；没有 code-availability-on-request promise。

GitHub core commit `ceeae40` 已推送，A100 自 `2faff4a` 快进；本机/A100 V3.3.2 suite 均为
66/66。evidence=58（44 local/contract、14 `/mnt`），claims=22、MBP=14/3/1、
blockers=MBP-10/13/14/15、中央 100-row 92/3/3/1/1、release/license/environment/
submission readiness=false 或 pending，`minimum_package_complete=false` 与
`outcome_trigger_fully_satisfied=false` 均不变。

## 63. V3.3.2 internal GitHub branch release candidate（2026-08-22）

本项无训练命令、GPU 轮询或新 attempt。合同没有在 minimum package incomplete 时要求 formal
GitHub Release/tag；因此组装并推送的是 version-controlled internal branch candidate，而非软件
release 或 submission package。新增 RC audit 与 3 项 focused tests，更新 README branch notice、
manuscript title/header、Code Availability 和 evidence/consistency manifests。

candidate 固定 Methods/Results/Discussion=14/15/5 subsections、Data/Code Availability internal
complete、claims=22 supported markers + 13 unsupported rows、figures/builders=7/6、evidence=59
（45 local/contract、14 `/mnt`）、MBP=14/3/1 与 blockers=MBP-10/13/14/15。README notice 明确
Route A V3.3.2 高于后续 legacy/general v2 body；formal GitHub Release、tag、persistent archive、
unauthenticated public access 与 submission authorization 均未声称。

Git tracked-policy preflight 只查路径、size、history 和 references，不读取 payload 内容：1 个 46,498-
byte Parquet 与 4 个总计 34,739,577-byte legacy B0 JSONL 仍 tracked。它们与当前 Git release rule
冲突，但历史 v3.1 rule 要求 B0 JSONL 原样保留且 active loader fail；没有自动删除或迁移，作为
formal-release blocker。GitHub core commit `52a41cb` 已推送，A100 自 `ceeae40` 快进；本机/A100
V3.3.2 suite 均为 69/69。中央 100-row 92/3/3/1/1、model/biological/external/guided success、
`minimum_package_complete`、`outcome_trigger_fully_satisfied` 与 `submission_ready` 均为 false。

## 64. V3.3.2 official provider rights evidence and FAIR gap closure（2026-08-22）

本项无训练命令、GPU/训练进度轮询或新 attempt。只读取官方 provider policy、选定的 accession/
HTTP/license-field-count 元数据和既有 data-rights inventory；没有读取 Development TEST、new final
Evaluation、GSE246381、E-MTAB-10902 或 generated-candidate outcome，也没有运行 guided XEditFlow。

新增冻结 source snapshot、14×36 provider-evidence CSV、可重复 builder/audit 与 4 项 focused tests。
provider 分布为 NCBI GEO=12、ENCODE=1、EMBL-EBI BioStudies ArrayExpress=1；官方 accession route
resolved=14、analysis/publication/citation supported=14。一般 provider policy 没有提升为逐研究许可：
study-specific license records=0、project redistribution authorizations=0、human reviews pending=14。
GEO submitter-IP exception、ENCODE exact-bundle repackaging gap、迁移 E-MTAB record 的 license/release
field 缺失均进入逐行表。FAIR evidence 为 findable/accessibility/interoperability/reusability=
14/14/0/0；target journal-specific check 仍 pending。

Data Availability、completion audit、consistency/evidence manifests 和 internal GitHub RC audit 已同步；
evidence 从 59 增至 62（45→48 local/contract，`/mnt` 仍 14），public release 与 submission 边界不变。
CSV focused validation 为 14 rows × 36 columns、14 unique studies、boolean domains valid、0 formula-like
cells。由于 workspace 缺少 spreadsheets skill 要求的 artifact-operation marker，没有生成 XLSX 或
声称视觉工作簿验证，也没有猜测/安装替代路径。

focused provider/Data Availability/RC/evidence tests 为 31/31；本机/A100 精确 V3.3.2 suite 均为
73/73。GitHub core commit `2adb8c3` 已推送，A100 自 `52a41cb` 快进到该 commit。中央 100-row
92/3/3/1/1、claims=22、MBP=14/3/1、blockers=MBP-10/13/14/15、model/biological/external/guided
success=false、formal release/tag=false、`minimum_package_complete=false`、
`outcome_trigger_fully_satisfied=false` 与 `submission_ready=false` 均不变。

## 65. V3.3.2 accountable-human study-rights review packet preparation（2026-08-22）

本项无训练命令、GPU/训练进度轮询或新 attempt。只使用已冻结的 14-study provider-evidence CSV
生成 accountable-human decision register，不访问 accession outcome、canonical payload、Development
TEST、new final Evaluation、sealed GSE246381、E-MTAB-10902 outcome 或 guided output。

新增 14×42 CSV packet、review instructions、可重复 builder/validator、audit 和 8 项 focused tests。
17 个 machine fields 从 provider evidence 逐字段冻结，20 个 human fields 初始为空，5 个 protected
fields 固定 false；当前 pending/completed/hold=14/0/0，signoff=0，target-journal check=0，public
release authorization=0。validator 拒绝机器证据改写、无身份/证据/signoff 的伪 `COMPLETED`、未列
exact-file scope 的授权，以及任何 `PENDING`/`HOLD` exact-file authorization。完整人工 review 仍不
自动成为 project release decision。

Data Availability completion audit、consistency/evidence manifests 与 internal GitHub RC 已同步；
evidence 62→64（48→50 local/contract，`/mnt` 仍 14）。spreadsheets skill 规定的 artifact-operation
marker 再次因文件缺失返回 `MODULE_NOT_FOUND`；本项没有猜测/安装替代路径、没有使用替代 workbook
library、没有生成 XLSX，也没有声称视觉工作簿验证。bundled Python 只读 CSV schema/全行投影核查为
14×42、14 unique studies、machine/human/protected=17/20/5、0 formula-like cells。

validator focused tests=8/8；本机/A100 精确 V3.3.2 suite 均为 81/81。GitHub core commit
`b93ba20` 已推送，A100 自 `2adb8c3` 快进到该 commit。中央 100-row 92/3/3/1/1、claims=22、
MBP=14/3/1、blockers=MBP-10/13/14/15、model/biological/external/guided success=false、
human review complete=false、formal release/tag=false、`minimum_package_complete=false`、
`outcome_trigger_fully_satisfied=false` 与 `submission_ready=false` 均不变。

## 66. V3.3.2 legacy tracked-payload formal-release disposition audit（2026-08-22）

本项无训练命令、GPU/训练进度轮询或新 attempt。只检查 5 个目标路径是否 tracked、文件 size 和
非 payload 文本引用；没有打开 JSONL/Parquet 内容，没有 copy/delete/move，未改 legacy reader，
未改 Git history，也未创建 formal tag/Release。

tracked payloads 为 1 个 46,498-byte `excel_inventory.parquet` 和 4 个总计 34,739,577-byte legacy
B0 JSONL，5 文件合计 34,786,075 bytes。Parquet 只有历史 producer、无当前 Route 2 consumer；4 个
B0 JSONL 仍被 `audit_split_manifests.py`、`eval_tracks.py`、`leakage_audit.py`、
`fm0_exposure_audit.py` 直接读取。旧 v3.1 contract 要求 active loader reject，但当前 test tree 未找到
`SUPERSEDED_NOT_LOADABLE` / `LEGACY_B0_INVALIDATION_MANIFEST` negative-loader evidence。因此 formal
release blocker 不只是文件仍 tracked，还包括 4-reader fail-close 尚未实现。

disposition audit/memo 推荐：取得明确用户授权后，先 fail-close readers 并加 negative tests，再保存在
`/mnt/.../route2/legacy_repository_payloads/`、迁移 Parquet 默认输出、停止 current HEAD tracking、加入
窄 ignore 并重新裁决 RC；本任务不建议 shared-history rewrite，任何 history rewrite 需要独立授权。
Code Availability statement/audit、consistency/evidence manifests 和 RC audit 已同步，evidence
64→65（50→51 local/contract，`/mnt` 仍 14）。

legacy/Code Availability/RC/evidence focused tests=31/31；本机/A100 精确 V3.3.2 suite 均为 85/85。
GitHub core commit `5bd4424` 已推送，A100 自 `b93ba20` 快进到该 commit。中央 100-row 92/3/3/1/1、
claims=22、MBP=14/3/1、blockers=MBP-10/13/14/15、model/biological/external/guided success=false、
payload migration authorized=false、formal release/tag=false、`minimum_package_complete=false`、
`outcome_trigger_fully_satisfied=false` 与 `submission_ready=false` 均不变。

## 67. V3.3.2 legacy B0 active-loader fail-close（2026-08-22）

本项无训练命令、GPU/训练进度轮询或新 attempt。没有读取 Development TEST、new final Evaluation、
sealed GSE246381、E-MTAB-10902 outcome、generated-candidate outcome 或 guided output；中央 100-row
仍为 92/3/3/1/1。

新增 `d1_staging/scripts/b0/legacy_split_guard.py`，并接入
`audit_split_manifests.py`、`eval_tracks.py`、`leakage_audit.py` 与
`fm0_exposure_audit.py`。4 个入口请求保留的 repository-root `data/b0_splits` 时，在读取任何
canonical/manifest 输入前抛出 `SUPERSEDED_NOT_LOADABLE`；其他显式 split path 仍可解析。新增 7 项
negative-loader tests，覆盖 shared guard、4 个 CLI fail-before-read、旧 JSONL size 不变与 source
ordering。5 个 tracked payload 的内容未打开，未 copy/delete/move，未改 Git history，未创建 formal
tag/Release。

legacy disposition、Code Availability、evidence manifest、manuscript 和 internal RC 已同步为
4 guarded / 0 unguarded / negative-loader evidence=true。当前 formal-release payload boundary 仍不
合规，因为 1 个 Parquet 和 4 个 legacy B0 JSONL 仍 tracked；向 `/mnt` 保存并停止 current-HEAD
tracking 需要明确用户授权。evidence=65（51 local/contract、14 `/mnt`），claims=22、
figures/builders=7/6、MBP=14/3/1、blockers=MBP-10/13/14/15。

guard/disposition/Code Availability/RC focused tests=17/17；本机/A100 精确 V3.3.2 suite 均为
92/92。GitHub core commit `794df0d` 已推送，A100 自 `5bd4424` 快进到该 commit。中央
model/biological/external/guided success=false、payload migration authorized=false、formal
release/tag=false、`minimum_package_complete=false`、`outcome_trigger_fully_satisfied=false` 与
`submission_ready=false` 均不变。

## 68. V3.3.2 Excel inventory generated-output boundary repair（2026-08-22）

本项无训练命令、GPU/训练进度轮询或新 attempt。没有读取 Development TEST、new final Evaluation、
sealed GSE246381、E-MTAB-10902 outcome、generated-candidate outcome 或 guided output；中央 100-row
仍为 92/3/3/1/1。

核查确认 legacy `import_excel_inventory.py` 的默认 `--parquet` 仍指向 Git tree，显式 override 也会在
audit Markdown 中被硬编码旧路径掩盖。新增冻结的 Route 2 storage/default constants 和可测试 parser；
future default 改为 `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/data_registry/excel_inventory.parquet`，
小型 audit default 仍为 `docs/data/excel_inventory_audit.md`，renderer 记录实际输出路径。历史 audit
只加 provenance notice，原 legacy output 行保留为历史事实。

本项没有运行真实 importer，没有读取真实 Excel/Parquet，没有创建 `/mnt` artifact，也没有复制、删除、
移动或停止 tracking 现有 5 个 payload。legacy disposition、Code Availability、manuscript、evidence
manifest 与 internal RC 已同步为 future producer default inside Git=false；remaining migration actions=3，
但 formal-release payload boundary 仍不合规，明确迁移授权仍缺失。evidence=65
（51 local/contract、14 `/mnt`），claims=22、figures/builders=7/6、MBP=14/3/1、
blockers=MBP-10/13/14/15。

producer-boundary/legacy/Code Availability/RC focused tests=13/13；本机精确 V3.3.2 suite=95/95。
本机 Python 缺少可选 Parquet engine，未安装依赖；A100 项目环境一次完成 V3.3.2 95 项与 importer
8 项，合计 103/103。GitHub core commit `dc2ed02` 已推送，A100 自 `794df0d` 快进到该 commit。
中央 model/biological/external/guided success=false、payload migration authorized=false、formal
release/tag=false、`minimum_package_complete=false`、`outcome_trigger_fully_satisfied=false` 与
`submission_ready=false` 均不变。

## 69. V3.3.2 authorized legacy payload current-HEAD migration（2026-08-22）

本项无训练命令、GPU/训练进度轮询或新 attempt。用户授权的范围仅为五个精确 legacy payload 的
`/mnt` 保存、current-HEAD 停止 tracking 和窄 ignore；shared Git history rewrite、formal tag/Release
与 public payload redistribution 均未授权。Development TEST、new final Evaluation、sealed
GSE246381、E-MTAB-10902 outcome、generated-candidate outcome 和 guided output 均未读取。

A100 源 commit=`1d899dd`；五文件 preflight tracked/size 全部符合既有审计，目标目录原先不存在。
无覆盖复制后，`/mnt/.../route2/legacy_repository_payloads/` 含五个 payload 和 `PROVENANCE.md`；逐项
source/destination byte size 一致，无项目 checksum。随后对五个精确 Git path 执行 current-HEAD
untracking，并加入五条 exact ignore；没有 broad directory ignore，没有 history rewrite。

current-HEAD payload policy 已从 5 tracked / non-compliant 改为 0 tracked / 5 preserved / compliant。
四个 B0 reader 仍为 4 guarded / 0 unguarded，future Excel Parquet default 仍在 `/mnt`。同步 legacy
disposition、Code Availability、manuscript、consistency/evidence manifests 和 internal RC；evidence
65→66（51 local/contract、15 `/mnt`）。RC formal blockers 从 6 减至 5，但 minimum package、human
rights/license、clean-environment、immutable archive 和 venue/authorship/disclosures 仍未完成。

focused tests=42/42；本机精确 V3.3.2=96/96；A100 V3.3.2+importer=104/104。GitHub core commit
`b6fbdce` 已推送，A100 自 `1d899dd` 快进。中央 100-row 92/3/3/1/1、model/biological/external/guided
success=false、formal release/tag=false、`minimum_package_complete=false`、
`outcome_trigger_fully_satisfied=false` 与 `submission_ready=false` 均不变。

## XEdit V3 方法修复启动与 projection-first 边界（2026-08-23）

用户冻结 XEditCritic V3 的严格 `0.25 each seed / 0.30 median` Spearman gate，并选择 18M/42M
XEditSetFlow capacity screen 与 soft-value SMC。机器协议位于
`configs/route_a_v3_route2_xedit_v3_method_repair_protocol_v1.json`；本轮不修改历史 V2 terminal
结果，也不授权额外 seed、阈值下调、TEST 后返调或 terminal rerun。

新训练的第一项实现是 `DevelopmentProjectionV3`：builder 在 canonical full decode 前用
`canonical_record_id` 查询 outcome-free frozen manifest，只有 TRAIN/VALIDATION 行才完整解析并写入
projection。TEST 行不解码；通用 TEST projection 被拒绝。V3 critic/flow 后续只接受 projection，
不再从 canonical outcome JSONL fallback。endpoint registry 同时提供给 full model 与 matched baseline，
并明确区分 GSE149487 的两个无 TRAIN 支持 endpoint。

实现 commit `1659633` 推送并同步 A100 后，真实 projection 已只运行一次并写入
`/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/projections/xedit_v3/development_train_validation_v1/`：
TRAIN=89,580、VALIDATION=18,293、TEST withheld=18,292，canonical full decode 的 TEST 计数为 0。
A100 focused tests=11/11、既有精确 V3.3.2 suite=96/96。该 materialization 不更新参数、不读取 TEST
outcome、不增加中央 training attempt，且不得再次运行覆盖。

## XEditCritic V3 edit-site token feature implementation preflight（2026-08-23）

本项实现 cache/online 共用的唯一 token-offset 与 chunk policy：局部表示使用 1000-nt chunk、64-nt
overlap、包含完整 clipped radius-16 window 的 most-centered chunk；site 与 local window 排除 special
token，旧 official masked whole-chunk mean 只保留为独立 global residual。cache 使用 ragged record/edit
offset，并以 shared sequence-position feature index 去重；持久化 payload 不写 raw sequence。

只读 TRAIN/VALIDATION projection 的几何核查为 107,873 records、346,862 record-edits、最大 38
edits/record、43,730 unique sequences、76,159 unique sequence-position pairs，证明多编辑不能按历史小
budget 截断。Development 最大序列长度为 837；通用实现仍覆盖大于 1000 nt 的 overlap/final-anchor 情形。

A100 GPU2 的真实 frozen mRNABERT smoke 验证得到 pretrained parameters=113,389,056、hidden=768、
tokenizer max length=1024；3 个请求位置全部返回 finite 768-d features，运行时 token-layout assertion 通过。
本机/A100 focused tests 均为 8/8，本机/A100 精确 V3.3.2 suite 均为 96/96。完整 cache 尚未
materialize，本项不训练、不新增中央 attempt，不读取 Development TEST 或新的 final Evaluation outcome。

## XEditCritic V3 C0/C1/C2/C3 architecture and LoRA preflight（2026-08-23）

已实现四臂共同的 outcome-free endpoint conditioner、hidden-65/depth-2 raw branch、C1 独立 global
residual、C2/C3 8-layer width-512/8-head/FFN-2048 edit transformer、attention+max pooling、low-rank
region/endpoint adapters 与 multiplicative-only study scale。study identity 不进入 shared effect trunk，
unknown study scale 严格为 1 且没有 study intercept；主模型以共享 directional scorer 的差保证 swap
antisymmetry和 identity zero。0.10 dropout 使用 forward/reverse 共享 mask，避免随机 mask 破坏差分约束。

按真实 Development vocab 实例化的 trainable 参数为 C0=486,784、C1=1,798,528、C2=29,489,049。
真实 mRNABERT last-four blocks 的目标模块为 combined Wqkv、attention output dense、gated FFN 与 FFN
output；固定 rank-16/alpha-32 LoRA 精确增加 983,040 参数，故 C3 总数为 30,472,089。协议中的
32–36M 是估计值；没有为追随估计而事后扩大 rank、block 数或 target projection。

source-only、edit-metadata-only 与 parameter-matched no-candidate-sequence control 使用相同 full-model
geometry；complete candidate-bundle permutation 留在 dataset layer。模型/LoRA focused tests 本机与
A100 均为 10/10，精确 V3.3.2 suite 均为 96/96。训练尚未启动，不新增 attempt，不访问
Development TEST 或新 final Evaluation outcome。

## XEditCritic V3 cache-arm training runner preflight（2026-08-23）

新增 projection-only C0/C1/C2 runner、train-only zero-anchored task-robust scaler、study→source-group
weights、sqrt(task-size) task-homogeneous pass sampler 和第 8 pass 的 cross-source-group pairwise logistic
loss。没有 canonical outcome fallback；两个无 TRAIN support 的 GSE149487 Validation task 使用 TRAIN
region、再 TRAIN global 的 scale fallback，不读取其 outcome 来拟合 scale。

真实 TRAIN preflight 为 89,580 draws/pass、2,802 batches/pass、8 passes 预计 22,416 updates。七个
TRAIN task 的 allocation 合计严格为 89,580；small task 达到 capacity 后自动重分配，任何 record 每
pass 不超过四次。complete candidate-bundle permutation 在 exact source/task strata 内覆盖 29,271
recipients，29,259 个 candidate sequence 实际改变，适用 task=6，超过 gate 所需的两个 task。

A100 GPU5 的真实 C0 batch-32 forward 为 one-task batch、finite prediction、486,784 parameters。训练
data/runner focused tests 本机/A100 均为 11/11，精确 V3.3.2 均为 96/96。C3 online-LoRA runner 仍待
实现，cache-arm screen 尚未启动，本项没有新 training attempt 或 protected outcome access。

## XEditCritic V3 C3 online-LoRA runner and screen gate preflight（2026-08-23）

C3 runner 以 physical microbatch=1、gradient accumulation 维持 effective batch=32 和与 cache arms 相同
optimizer update 数；前七 pass 只回传 standardized Huber，第八 pass 在同一 effective update 内重新
计算冻结的 disjoint cross-source-group ranking pairs。head/LoRA learning rates 分别固定为 3e-4/3e-5；
checkpoint 只保存 983,040-parameter LoRA delta，不复制 113M frozen encoder。

A100 GPU5 真实 mRNABERT 的 40-nt two-record forward/backward smoke 不执行 optimizer step：output finite、
identity exact zero、32 个 LoRA parameter tensors 均有 gradient、non-LoRA encoder gradient count=0，峰值
显存 884,616,192 bytes。完整 Development 最大长度 837 的正式显存需求仍必须等有足够空闲的 GPU
再启动，不进行 CPU fallback。

screen adjudicator 硬要求 C0/C1、C2/C3 full、每个 selectable arm 三项同几何 control 和 complete-bundle
permutation，共 12 个 exact artifacts；缺失/多余 artifact 或 protected outcome read 均硬失败。C2/C3
只有各自完整 gate 全通过才 eligible；差值大于 0.005 选高者，否则选 C2。screen PASS 只授权
confirmation，不授权 Development TEST。focused tests 本机/A100=19/19，V3.3.2=96/96；训练未启动。

## XEditCritic V3 cache materialization 与 XEditSetFlow V3 implementation preflight（2026-08-23）

Critic edit-site cache 已在 A100 physical GPU2 按低频监控纪律完成；完整载入验证为 107,873 records、
346,862 ragged edits、43,730 unique sequences、76,159 unique sequence-position、embedding width 768，
artifact size 433,780,150 bytes。payload 内 raw sequence、Development TEST 与 Evaluation record 计数均为
0。cache/online 数值对齐预设三条记录覆盖 133-nt/1-edit、164-nt/7-edit 和 837-nt/1-edit，容差在读取
差异前固定为 max abs 0.02 / mean abs 0.005；真实 A100 对齐需 current HEAD 同步后执行。

XEditSetFlow V3 已实现 projection-only、outcome-free 的 set-marginal state/target、task/source-group balanced
sampler、逐 source float16 mRNABERT token cache、F1 原 V2 两层 trunk 目标诊断、F2 8×384 与 F3 12×512
hybrid local-attention+dilated-depthwise trunk、hard mask before rate normalization、独立 STOP head、common
Validation set-NLL patience-2 早停、冻结 F0 epoch-1 common-NLL 只读回放、32-candidate unguided replay/G0/
recovery validation 和严格 F2/F3 screen gate。真实容量为 F2=16,179,014、F3=42,197,158；F1 不可入选。

SetFlow eligible geometry 为 TRAIN 68,294、VALIDATION 15,924、over-budget 排除 21,286/2,369、19,303
unique sources、2,817,781 source tokens、maximum length 837。source-token cache 与任何 F1/F2/F3 参数更新
均尚未启动；Base Flow V2 不重训。新增/相邻本机 focused tests=36/36、精确 V3.3.2=96/96。中央训练
attempt 仍为 100 total / 92 completed / 3 failed / 3 incomplete / 1 stopped-throughput / 1 stopped-priority；
Development TEST、新 final Evaluation、critic/generator success、guided authorization 与 submission-ready 均不变。

首次 raw-online cache alignment 按预冻结 max/mean tolerance 0.02/0.005 终态 FAIL 并保留在
`cache_online_alignment_v1.json`：overall max=0.0678864、mean=0.00028031；837-nt 样本 max=0.00360584，
越界集中于短序列在不同 BF16 batch-padding geometry 下的少量 local extrema。阈值没有放宽，cache 也不
覆盖重建。工程修复把 C3 表示定义为 `cached base + (online adapted - online zero-LoRA)`，并让 frozen
encoder 始终 eval、只启用 LoRA dropout；零 LoRA 时应精确返回 cache，非零时保留可微 LoRA delta。
原 FAIL 不删除；同一阈值的 v2 repair validation 需在新 current HEAD 上一次性执行。

repair commit `f1b6131` 推送并同步 A100 后，v2 在完全相同的三条记录和 0.02/0.005 阈值下 PASS，
observed max/mean difference 均为精确 0。独立真实 164-nt/7-edit backward smoke 无 optimizer step：
983,040 LoRA 参数的 32 个 tensor 均收到 gradient，non-LoRA encoder gradient count=0，loss finite，峰值
显存 684,938,752 bytes；因此 cache anchoring 没有把适配分支常数化。A100 修复 focused=15/15、
V3.3.2=96/96；SetFlow 全 focused=36/36、V3.3.2=96/96。Critic/SetFlow formal training 仍未启动。

Critic C0–C3 runner 的中央尝试记录已补齐：每个 exact screen artifact 使用唯一 attempt id，在第一个
optimizer step 前写 RUNNING，成功或异常时原子 upsert COMPLETED/FAILED；记录 TRAIN/VALIDATION/
withheld TEST=89,580/18,293/18,292、Evaluation=0、参数量、GPU、updates、最终 Validation metrics 与
wall/VRAM。cache arms 必须证明参数变化，C3 必须分别证明 head 与 LoRA 变化，否则不能写 COMPLETED。
ledger focused/V3.3.2 本机=13/13、96/96；正式 C0 尚未启动，因此中央表此刻仍为 100 行。

## XEdit V3 post-screen confirmation, atomic TEST, closed/guidance stack preflight（2026-08-23）

Critic 与 SetFlow 已各自补齐 screen-PASS-only confirmation config builder 和 exact three-seed gate。Critic
仅接受 20260831/20260901/20260902 的 selected C2/C3 + per-seed matched C0，并以 task 内 source-group
paired bootstrap 的 task-macro Spearman CI 执行 3/3、median、MAE 与 task-win gate；SetFlow 仅接受
20260904/20260905/20260906 的 selected F2/F3，三个 seed 均需满足 recovery/top-k/unique 和全 G0 correctness。
任何 NO-GO 均不生成下一阶段授权，也不允许第四个 seed。

一次性 frozen TEST runner 只有在 Critic three-seed PASS 后可运行。它先预载核验 6 个 checkpoint，再消费
一次授权；canonical scan 对 TRAIN/VALIDATION 仍只提取 record id，只有 TEST=18,292 行完整解析。TEST
edit-site base features 只在内存中按 frozen cache geometry 构建，C3 使用该 ephemeral base 加 online LoRA
delta；不会写通用 TEST projection/cache，失败也不自动重试。frozen TEST、all-Development refit、LOSO 与
guidance 当前均未授权、未执行。

closed benchmark 已实现最多 5 edit 的全部排列精确求和（上限 120 paths），并以 source 为单位报告 NDCG、
normalized regret、top-1 recall；少于两个 measured candidates 或 zero measured gain 保持 undefined，不填 0。
soft-value core 固定三 seed uncertainty reward、K=8、6×384 scalar-only value、18 个 κ/τ/β 组合、32-particle
SMC、ESS<16 stratified resampling、32 candidate cap、320 forward-equivalent ceiling 和三 critic member 分别
计费。Critic frozen TEST/refit/7-study LOSO 四个 predecessor 未全 PASS 前 guidance authorization 为 false。

本机 confirmation focused=39/39、closed=4/4、guidance core/gate=6/6+3/3、atomic TEST direct=3/3；本批
A100 测试等待 active C0/C1 terminal 后再同步，避免运行中的 ledger 把非实际代码 commit 写成 provenance。
C0/C1 已在 seed20260830 分别启动于 GPU0/GPU5；曾在 elapsed=64s 发生一次过早快照，已如实记为节奏错误，
其后未补查，下一次合并检查延后到启动满 30 分钟。SetFlow source-cache 在此期间未被轮询。

到期后只做了一次精简合并检查：C0/C1 elapsed=1,913 秒且均存活，SetFlow source-cache elapsed=2,332 秒
且仍存活，三个任务均无 terminal summary。没有读取 pass curve 或完整训练日志；下一次至少再间隔 30 分钟。
A100 不同步 post-screen commit，继续保持 active attempt 的 `22317ed` provenance。

## XEditFlow V3 value/SMC/closed runtime preflight（2026-08-23）

新增组件尚未执行，当前只作为 readiness-gated code path：

- `build_route2_xeditflow_value_targets_v3.py`：TRAIN-only，K=8，每 rollout 三 critic seeds；
- `train_route2_xeditflow_value_v3.py`：6×384 scalar potential，固定 8 pass BF16 Huber final checkpoint；
- `run_route2_xeditflow_smc_v3.py`：32 particles、ESS<16 stratified、candidate cap32、compute ceiling320，
  预留 3 critic forwards；不足 cap 时在余量内追加冻结 seed rounds；
- `evaluate_route2_xeditflow_closed_neighborhood_v3.py`：Development Validation、最多 5 edits/120 paths
  exact order-invariant probability、source-level NDCG/regret/top-1，undefined 不填 0。

SMC 使用 base proposal 与 scalar-potential importance weight，未增加 action-ratio head。所有入口要求
critic frozen TEST/refit/LOSO readiness 和 SetFlow confirmation 同时成立；当前调用会 fail closed。本机
合并 focused=89/89、V3.3.2=96/96、compile/diff-check PASS；A100 测试延后到 C0/C1 terminal。
