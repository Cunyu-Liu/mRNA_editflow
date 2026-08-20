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
