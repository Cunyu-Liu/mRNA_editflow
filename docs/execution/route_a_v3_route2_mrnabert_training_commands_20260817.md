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
- 实测 `num_workers=0` 对当前内存特征缓存最快，因此没有为了形式上的并行而增加 workers。
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

## 9. 2026-08-18 10:13 运行快照

| 运行 | 终态/当前状态 | task-macro Spearman | task-macro standardized MAE | 说明 |
|---|---:|---:|---:|---|
| RNA-FM Huber | 运行中 | — | — | 历史 encoder 对照；不使用其中间 epoch 做选择 |
| RNA-FM learned variance | **100/100 完成，best epoch 49** | 0.130418 | 1.935143 | global Spearman 0.200763；prediction std / target std 0.031654 |
| RNA-FM fixed variance | **100/100 完成，best epoch 92** | 0.098003 | 2.306655 | global Spearman 0.174160；prediction std / target std 0.138453 |
| mRNABERT Huber | **100/100 完成，best epoch 44** | **0.149988** | **2.108870** | global Spearman 0.198122；prediction std / target std 0.074666 |
| mRNABERT fixed variance | **运行中** | — | — | RNA-FM 前序完成后已自动在 GPU 5 启动 |
| mRNABERT learned variance | **运行中** | — | — | RNA-FM 前序完成后已自动在 GPU 3 启动 |

同信息最强已完成 baseline 的 task-macro Spearman 为 `0.131714`；Huber 单 seed 的暂时增量约为 `+0.01827`，还不能代替三种 loss、controls 或 three-seed 判断。三个 mRNABERT loss 当前为 `1/3` 完成，所以 controls、final seeds、TEST、全量 refit、LOSO readiness 和 guided XEditFlow 均未越级启动。中央训练表在两个新 mRNABERT loss 启动后为 88 个唯一尝试、72 列：77 completed、3 running、3 failed、3 incomplete、2 stopped。

## 10. Guidance readiness 与执行入口

冻结 guidance policy：

```text
configs/route_a_v3_route2_mrnabert_guidance_reward_policy_v1.json
```

它只使用 standardized predicted mean；learned uncertainty 只做诊断，不进入 reward。在线 candidate encoder 的单次验证由下列低频调度器等待空闲 GPU 4 后执行：

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

旧独立评估器曾直接混合不同 endpoint 的原始量纲，分钟级 half-life 误差会压倒 log-fold 和 usage，因此该 run 已作为 `STOPPED_PRETERMINAL_METHOD_INVALID` 保留，不能用于生成方法选择。新的单次预冻结 run 使用 TRAIN-only task robust scaling、独立的 0.51M Siamese CNN、Development VALIDATION 和 GPU 2；它不读取 mRNABERT 特征、Development TEST 或 Evaluation：

```bash
nohup scripts/route_a_v3/schedule_route2_independent_evaluator_gpu2_v3.sh \
  >/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/schedulers/independent_evaluator_gpu2_v3.log \
  2>&1 </dev/null &
```

该调度器等待 Base Flow V2 的 GPU 2 任务完成后再启动，不与当前训练争卡。它只运行一次，不按结果追加 HPO；只有 task-macro Spearman 超过预冻结 candidate-permutation reference `0.1012476`、至少 5 个 task 为正且所有数据隔离检查通过，才标记为 qualified。无论 PASS 或 NO-GO，尝试都会自动写入中央训练表。

## 12. 2026-08-18 10:13 追加快照

- mRNABERT Huber 已完成，三种 loss 的正式 summary 为 `1/3`，所以 loss 选择、controls、three seeds、冻结 TEST 和全量 refit仍未越级启动；
- RNA-FM fixed/learned-variance 历史对照已完成并同步中央表；对应 GPU 释放后，新的 mRNABERT fixed/learned-variance 已自动启动；
- Huber 实际 wall time 为 39,616.45 秒，约 396.16 秒/epoch、70.76 ms/step 和 226.12 TRAIN records/s；相对 batch16 BF16 微基准约为 93%，没有严重 DataLoader 饥饿证据；
- GPU 0–5 当前均有项目任务，显存状态正常，没有 CUDA/NaN/提前退出证据；
- workers 0/4/8、batch32/64 数据管线测试正在等待 GPU 0 降低利用率；Base Flow V2、在线 mRNABERT encoder validation 和主 post-selection scheduler 继续低频等待各自条件；
- 新增的独立评估器将等待 GPU 2 的 Base Flow 验证完成后再运行，不抢占现有任务；
- 当前正式数据与 claim 状态仍为 `ordinary=1 / A1=1 / true-A2=0 / canonical=6,547 / NOT_ESTABLISHED`。

## 13. 低频查看进度

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
