# Route A V3.3 Route 2 训练尝试记录表

## 表格位置

所有发生参数更新的 Route 2 训练统一追加或更新到：

`/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiment_tracking/route2_training_attempts.csv`

每个 run 目录同时保存一份 `training_attempt.json`。CSV 是跨实验总表，JSON 是该次实验的独立记录。训练中的行标记为 `RUNNING`，正常结束后原行更新为 `COMPLETED`，异常结束更新为 `FAILED`；不会为同一 run 重复增加多行。

## 必须记录的内容

| 类别 | 记录内容 |
|---|---|
| 身份 | attempt ID、baseline ID、用途、科学角色、代码 commit、输出目录 |
| 数据 | canonical 文件、实际 study、区域、TRAIN/VALIDATION/TEST 数、Evaluation 读取数 |
| 模型 | 模型类型、预训练模型、可训练参数、冻结参数、总有效参数 |
| 训练 | loss、loss aggregation、sampling/weighting mode、batch、epochs、学习率、weight decay、优化器、BF16/FP32、GPU、seed、optimizer steps、最佳 epoch |
| 数据管线 | workers、pinned memory、non-blocking transfer、预训练特征缓存 |
| 架构 | encoder attention backend、预训练位置编码、critic 位置特征、生成动作空间、生成器位置与轨迹时间特征 |
| 结果 | VALIDATION/TEST 指标、峰值显存、耗时、状态、失败原因与解释备注 |

## 当前解释边界

- Development 的训练、消融、control 与 generation G0 都进入此表。
- Evaluation outcome 的读取数必须为 0，除非进入已经冻结的最终评估任务；Evaluation 不用于训练或选择模型。
- `COMPLETED` 只表示工程运行完成，不表示模型具有科学预测价值。
- uncertainty head 的输出只记为模型诊断量，不冒充生物重复得到的标准误。
- unguided Base Flow 明确记录为工程 G0；在 critic 和独立评估完成前，不称为生物优化成功。
- 失败、显存不足、吞吐不合格和人工停止的尝试都保留，避免只报告成功 run。

## 维护方式

正式训练器在开始、完成和失败时自动维护总表。已有历史 run 使用：

```bash
python scripts/route_a_v3/sync_route2_training_attempt_ledger_v1.py \
  --ledger /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiment_tracking/route2_training_attempts.csv \
  --run-dir /ABSOLUTE/RUN/DIRECTORY
```

查看总表时优先使用 `status`、`result_stage`、`model_kind`、`loss_kind`、`seed` 和 `included_study_unit_ids` 筛选；不要把不同阶段或读过 Evaluation 的结果混在同一个最优模型比较中。

## 当前模型与框架优化状态

以下状态只描述工程实现与实测结果，不等同于科学效果已经成立。

| 项目 | 当前状态 | 采用或不采用的理由 |
|---|---|---|
| 冻结预训练编码器 | 已启用 mRNABERT | 预训练参数冻结；Development 的 source/candidate 表征预先缓存，critic 的每个 epoch 不重复运行 113M 参数编码器。 |
| 可训练 critic | 已启用约 9.34M 参数 edit-centered critic | 同时使用编辑位点注意力汇聚、编辑位点最大汇聚、全 source 均值/最大背景、预训练差分和共同背景；不是只看单个编辑位置。 |
| 数值精度 | 已启用 BF16 | A100 实测与 FP32 输出满足预设一致性门，同时吞吐明显更高。 |
| 优化器 | 已启用 fused AdamW | 已在当前 A100 训练 profile 中验证可运行；与 BF16 一起用于三种 loss 的匹配比较。 |
| 数据搬运 | 已启用 pinned memory 与 non-blocking transfer | Development 特征已缓存；匹配 benchmark 显示 batch32 下 workers4 比 workers0 提速约 7%，workers8 无进一步收益。后续新正式 cohort 使用 workers4；当前三种 loss 不改配置。 |
| 批处理 | 已使用 length-bucket batch sampler | 减少不同长度序列混批造成的 padding 浪费；排名 loss 另使用 source-group sampler。 |
| mRNABERT 位置编码 | 保留官方双向 ALiBi | 这是预训练模型的一部分；直接改成 RoPE 会造成架构与预训练权重不一致，因此不做无依据替换。 |
| critic 位置特征 | 已启用归一化绝对位置 + edit-gated 位置 | 让模型直接知道编辑发生在序列的相对位置，同时避免不同长度序列使用不可比的绝对索引。 |
| Flash/SDPA attention | 正在做候选后端验证 | 不修改 ALiBi 语义；先做 attention primitive，再做完整 encoder 输出对齐与速度比较。只有缓存与新序列一致、且完整编码器实测至少提速 10% 才启用 PyTorch SDPA，否则继续使用官方实现。 |
| `torch.compile` | 暂未启用 | 当前 trainer 明确记录为未实现；在没有端到端收益证据前不把编译开关写成已优化。 |
| uncertainty head | 三种匹配 loss 已完成，选择 Huber | learned variance 的不确定性与绝对残差相关，但 task-macro Spearman 和 standardized MAE 均弱于 Huber；NLL/方差诊断不覆盖均值预测选择规则。 |
| Edit Flow 计算 | 已加入重复状态的 rate cache | 逻辑 trajectory 决策次数与真实 generator forward 次数分开记账，使 matched-budget baseline 使用实际计算成本。 |

## 当前三种 loss 的冻结比较口径

- Development 共 126,165 条 canonical source-candidate records。
- 当前 loss 选择只使用冻结的 TRAIN 89,580 条与 VALIDATION 18,293 条；Development TEST 18,292 条保持未打开。
- 三个主实验均为 100 epochs、batch size 16、learning rate `1e-4`、weight decay `1e-4`、BF16、fused AdamW、相同 seed 和相同约 9.34M 可训练 critic。
- 比较顺序为：最大化 task-macro Spearman，其次最小化 task-macro standardized MAE，最后比较 global Spearman。
- 学习方差模型还报告 prediction spread、预测标准差与绝对残差的相关性，用于识别 uncertainty 是否吸收误差或伴随 mean collapse；这些诊断不改变上述选择规则。
- loss 选定后才运行 candidate permutation 与 parameter-matched source-only controls；controls 通过后才进入三个 final seeds、一次冻结 Development TEST 和全 126,165 条最终 refit。

## mRNABERT Huber 首轮终态（2026-08-18）

Huber 主实验已经完成 100 epochs。该终态只使用冻结 TRAIN/VALIDATION，Development TEST 18,292 条仍未读取，Evaluation outcome 读取数为 0，科学 claim 仍为 `NOT_ESTABLISHED`。

| 项目 | 终态 |
|---|---:|
| TRAIN / VALIDATION | 89,580 / 18,293 |
| optimizer updates | 559,900 |
| 选择的 checkpoint epoch | 44 |
| VALIDATION global Spearman | 0.198122 |
| VALIDATION task-macro Spearman | 0.149988 |
| VALIDATION task-macro standardized MAE | 2.108870 |
| prediction std / target std | 0.074666 |
| 总 wall time | 39,616.45 秒（约 11.0 小时） |
| 实际平均 epoch 时间 | 396.16 秒 |
| 实际平均 optimizer-step 时间 | 70.76 ms |
| TRAIN records / wall second | 226.12 |

此前 batch16 BF16/fused 微基准为约 242.88 records/s，因此完整 100-epoch 训练达到了微基准吞吐的约 93%。这说明当前没有严重的 DataLoader 饥饿；主要成本来自 batch16 下每 epoch 5,599 次参数更新。此前 batch32/64 对照中 batch32 约 339.45 records/s、batch64 反而降至约 207.52，结合下面的 workers 对照，后续新正式 cohort 采用 batch32/workers4，而不采用 batch64 或 workers8。

DataLoader 匹配 benchmark 随后在 batch32、BF16、fused AdamW、pinned memory、non-blocking transfer 下完成：workers0 为 290.58 records/s，workers4 为 310.94 records/s，workers8 为 291.50 records/s。三组 mean loss 完全一致且全部有限，Evaluation 读取数为 0。workers4 相对 workers0 提速约 7.0%，而 workers8 仅约 0.3%，说明 4 workers 是后续新正式训练的合理配置，继续增加到 8 只增加进程开销。该结论不追溯改变正在运行的三种 loss 比较。

Huber 的 prediction spread 已从早期近乎常数输出改善到目标标准差的约 7.47%，但仍明显偏窄；它当前不能单独通过 critic guidance gate。

## 三种 mRNABERT loss 正式终态（2026-08-19）

| loss | selected epoch | task-macro Spearman | task-macro standardized MAE | global Spearman | prediction std / target std |
|---|---:|---:|---:|---:|---:|
| Huber | 44 | **0.149988** | **2.108870** | **0.198122** | 0.074666 |
| fixed variance Gaussian NLL | 54 | 0.120695 | 2.428299 | 0.171786 | 0.080062 |
| learned variance Gaussian NLL | 84 | 0.123583 | 2.523861 | 0.176032 | 0.149625 |

冻结均值性能规则选择 Huber。learned variance 的预测标准差与绝对残差 Spearman 为 `0.490600`，说明方差头确实学到了一部分残差尺度；但它的 task-macro Spearman 比 Huber 低 `0.026406`，standardized MAE 也更差。因此当前证据支持“uncertainty 吸收了部分误差尺度、却没有改善均值预测”，而不是“uncertainty head 已解决 mean collapse”。Huber candidate-permutation 和 parameter-matched source-only controls 已启动；在它们完成并通过前，不进入 final seeds、冻结 Development TEST 或全量 refit。

## source-only control 终态（2026-08-19）

parameter-matched source-only control 已完成全部 100 epochs 和 559,900 次 optimizer updates，使用相同 mRNABERT cache、critic 参数规模、Huber loss、TRAIN/VALIDATION、seed 和训练预算。Development TEST 与 Evaluation outcomes 均未读取。

| 项目 | 完整 Huber | source-only control | 差值（完整－control） |
|---|---:|---:|---:|
| task-macro Spearman | 0.149988 | 0.025703 | +0.124285 |
| global Spearman | 0.198122 | 0.068596 | +0.129527 |
| task-macro standardized MAE | 2.108870 | 1.841805 | +0.267065 |
| prediction std / target std | 0.074666 | 0.001301 | +0.073365 |

结果说明 candidate/edit 分支为跨 task 排序提供了明显信息，完整模型不是只靠 source/context 背景取得相关性；source-only 输出几乎塌缩为常数。与此同时，source-only 的 standardized MAE 更低，反映“更保守但缺乏排序能力”的预测可以在绝对误差上占优。最终 signal-control 判定必须同时纳入 candidate-permutation、相关性和冻结的误差规则，不能提前把该单项结果写成 gate PASS。

## signal controls 联合终态与 three-seed 启动（2026-08-19）

candidate-permutation control 已完成 100 epochs / 559,900 updates，Development TEST 与 Evaluation outcomes 均未读取。

| 项目 | 完整 Huber | candidate permutation | source-only |
|---|---:|---:|---:|
| task-macro Spearman | **0.149988** | 0.100819 | 0.025703 |
| task-macro standardized MAE | 2.108870 | **1.812514** | 1.841805 |
| prediction std / target std | 0.074666 | 0.001155 | 0.001301 |

冻结联合裁决的六项检查全部为 true：完整模型超过 strongest same-information baseline；task median 为正；在两个预指定可判任务上都胜过 permutation，平均 margin `0.109332`；超过 source-only 的 macro margin 为 `0.124285`，并在 9 个任务中胜出 7 个。因此状态为 `MRNABERT_SIGNAL_CONTROLS_SUPPORT_FINAL_SEED_CONFIRMATION`。

这个状态只支持三个固定 seed 的 Development VALIDATION 复现，不是科学 claim、不是 TEST 结果，也不授权 guided generation。三个 Huber final seeds（20260822、20260823、20260824）已分别在 GPU 0、3、5 启动；只有 three-seed 冻结判定继续通过，才允许单次打开 Development TEST。

## three-seed confirmation 终态（2026-08-19）

三个 final seeds 均完成 100 epochs / 559,900 updates，只使用冻结的 Development TRAIN/VALIDATION；Development TEST 与 Evaluation outcomes 均未读取。

| seed | selected epoch | task-macro Spearman | strongest-baseline margin | task-macro standardized MAE | directional pass |
|---:|---:|---:|---:|---:|---|
| 20260822 | 6 | 0.116129 | -0.015586 | 1.799772 | false |
| 20260823 | 86 | 0.116908 | -0.014806 | 2.079968 | false |
| 20260824 | 25 | 0.137384 | +0.005669 | 1.956973 | true |

三个 seed 的 task median 均为正，但预冻结规则要求三个 seed 相对 strongest same-information baseline 的 improvement 方向全部为正；实际只有 1/3 达到。因此裁决为 `THREE_FINAL_SEEDS_DO_NOT_SUPPORT_FROZEN_DEVELOPMENT_TEST`，`supports_single_frozen_development_test=false`。

这表明先前 seed 20260816 的 `0.149988` 是不稳定结果，当前 mRNABERT critic 尚不能写成可重复优于强 baseline。调度器已在 VALIDATION 层停止：没有打开一次性 Development TEST，没有执行 all-126,165 refit，没有启动 LOSO readiness 或 guided XEditFlow。该负结果作为 Benchmark 与 generation-limits 论文路径的一部分保留，不追加临时 seed 或改阈值来追逐通过。

## 记录边界

- `training_attempts.csv` 是 Excel 可直接打开的跨实验总表；每个 run 的 `training_config.json` 保存该次尝试的完整参数，避免总表为了可读性遗漏长配置。
- attention benchmark、encoder 一致性验证等没有参数更新的工程实验保存独立报告，不冒充训练尝试。
- 长训练可能跨越代码更新。写表器现在保留任务启动时的 `code_commit`/`started_at`，并保留总表中它不认识的较新字段，避免旧进程结束时把新列静默删除。当前仍在运行的旧进程是在该修复前启动的；它们全部结束后要用各 run 的 `training_attempt.json` 做一次最终表头与生成器字段恢复。

## 2026-08-20 工程验证终态

在线冻结 mRNABERT 编码器已完成，状态为 `ONLINE_FROZEN_MRNABERT_MATCHES_CANONICAL_CACHE`：冻结参数量 113,389,056，64 个 embedding 对比的最大绝对差为 0.00951385（容差 0.01），中位吞吐 74.55 sequences/s，Evaluation records read 为 0。该任务没有参数更新，因此按本表既定边界保存在独立 validation summary，不增加中央训练尝试数；中央表仍为 94 个唯一尝试、86 completed、3 failed、3 incomplete、2 stopped。

该工程成功不覆盖 mRNABERT critic 的 three-seed 失败。Development TEST、all-126,165 refit、LOSO、guided XEditFlow 和 GSE232572/E-MTAB-10902 Evaluation outcomes 仍保持未打开。Base Flow V2 的调度已改为只按剩余显存选择 GPU 0–5 中显存最多的一张卡，按当前约 1.04GB 的实际占用保留 2GB 最低可运行线，不再设置利用率 gate；其后依次进入 validation 与独立 evaluator。

Base Flow V2 已作为正式工程训练尝试自动登记为 `COMPLETED`：817,957 trainable parameters、30 epochs、32,040 updates、BF16/fused AdamW、GPU 0、峰值显存 289.49MB、wall time 2,196.71 秒，selected epoch=1，best validation NLL=5.51248。TRAIN/VALIDATION 明显分叉，记录为 base-rate overfitting，不改变 `scientific_claim_status=NOT_ESTABLISHED`；后续只由一次 Flow G0 validation 裁决合法性和采样工程 readiness。

## Critic V2 control screen 启动与中央表状态（2026-08-20 20:05）

一次 freshness check 先确认 independent evaluator adjudication 和七方法
matched-generation suite 已经 terminal，故两者均未重复运行。Critic V2 的
四臂 screen 随后只启动一份，screen seed 固定为 `20260825`：

| arm | GPU | 中央表状态 | sampling | loss aggregation |
|---|---:|---|---|---|
| full | 2 | RUNNING | TASK_STUDY_SOURCE_GROUP_BALANCED_FIXED_DRAWS | TASK_MACRO_MEAN |
| candidate permutation | 4 | RUNNING | TASK_STUDY_SOURCE_GROUP_BALANCED_FIXED_DRAWS | TASK_MACRO_MEAN |
| parameter-matched source-only | 3 | RUNNING | TASK_STUDY_SOURCE_GROUP_BALANCED_FIXED_DRAWS | TASK_MACRO_MEAN |
| source+edit metadata、无 candidate global representation | 5 | RUNNING | TASK_STUDY_SOURCE_GROUP_BALANCED_FIXED_DRAWS | TASK_MACRO_MEAN |

中央训练尝试表当前为 100 个唯一 attempts，其中 4 个 Critic V2 arms 为
RUNNING。它们都使用 Development TRAIN/VALIDATION、Huber、batch 16、100
epochs、冻结 mRNABERT 和 9,342,914 参数 critic；此时尚无 terminal 指标，
不得提前比较 arm 或书写 PASS/NO-GO。Development TEST 18,292 条继续
withheld，Evaluation outcome 读取仍为 0。

在 control outcome terminal 前，exact three-seed confirmation protocol 也已
前瞻冻结：只允许 `20260822/20260823/20260824`，controls PASS 后动态选择
GPU0-5 中三张显存足够的卡；不得补第四个 seed。裁决统一报告 task-macro
Spearman、standardized MAE、prediction spread、strongest-baseline margin、
positive-task count、相对三个 controls 的差距和非有限值/mean-collapse/尺度
诊断。controls 必须 PASS 且 3/3 baseline margins 均大于 0，才只授权下一步
单次冻结 Development TEST；confirmation scheduler 本身不会打开 TEST。

条件式 watcher 已在 A100 以 PID `4148582` 启动，poll interval 为 900 秒；
首条日志为 `waiting_for_critic_v2_control_adjudication`。启动验收时，three-seed
runtime config root、run root 与 adjudication 文件均不存在，符合 gate 前不创建
seed 产物的冻结要求。A100 新增三种子配置/裁决的 7 个 focused tests 全部通过。

等待期间完成的 Route 2 V3.3.2 manuscript evidence packet 属于零参数更新的
论文/记录任务，不增加中央训练尝试。它把七方法 terminal Development 聚合表、
独立 evaluator qualification、原 three-seed negative result、Critic V2 RUNNING
状态和 protected-outcome boundary 绑定到 12 个 evidence IDs；一致性检查覆盖
13 个 claims 与 7 个方法。六个 search 方法缺失 per-method generation wall time
被明确记为 `NOT_RECORDED`，不据此重跑已 terminal 的 suite。

未来 matched/guided parallel stages 已补充逐方法 wall time 落盘；该记录修复
没有参数更新、不新增中央训练尝试，也不追溯填充已 terminal baseline 的缺失
字段。对应 focused suite tests 6/6 通过。

## Critic V2 单次冻结 TEST 配置门（2026-08-20）

Critic V2 control 尚在运行且 three-seed outcome 尚不存在时，已前瞻冻结独立的
single-TEST protocol 和 V2-only config preparer。它固定 seed `20260823`，完整
重放 control/three-seed/TEST 三份协议的同一训练 policy，保留
confirmation 的 `BEST_VALIDATION` 选择来源；由于 TEST 阶段会把 TRAIN+VALIDATION
折入训练，可执行 TEST checkpoint 规则前瞻固定为 100 epochs + `FINAL_EPOCH`，
不使用 TEST 选择 epoch。它要求两个裁决 PASS、精确三 seed、3/3
positive strongest-baseline margins、protected outcomes 未进入以及被选 confirmation
config 身份一致。preparer 只会写出一次性 runtime config；不会训练，也不会自行
读取 TEST。既有 runtime config 或 run directory 任一存在时均拒绝覆盖。

本任务没有发生参数更新，也没有调用真实 preparer，因此中央 CSV 不增加一行；
最近一次已记录的中央状态仍是 100 个唯一 attempts，其中四个 Critic V2 arms 为
RUNNING。Development TEST 与 Evaluation 仍关闭。focused verification 覆盖新门
和既有 three-seed 配置路径，共 14/14 通过。

## Critic V2 all-Development refit 配置门（2026-08-20）

已在 TEST outcome 不存在时冻结 V2-only all-126,165 refit protocol/preparer。
它只接受 exact seed-20260823 V2 TEST config 与合法 terminal summary，验证同一
model/loss/full policy、TRAIN+VALIDATION 107,873 条、TEST 18,292 条、固定 100
epochs/`FINAL_EPOCH`、CUDA 参数更新和 Evaluation read=0。合同没有新增 TEST
数值阈值，preparer 因而只要求 metrics 存在而不按其数值分支；合成负 TEST 指标
仍生成同一个固定 refit config，防止事后重选。

真实 preparer/refit 均未运行，所以中央 CSV 不增加记录，最近已记录的 100 个
唯一 attempts/四个 Critic V2 RUNNING 状态不因本任务改变。新门 focused tests
13/13 通过；包含 TEST gate、three-seed config 与 trainer split 的扩展 suite 为
81 passed、4 个本机无 CUDA 的既有 skips。Development TEST、refit、LOSO、
readiness、guided generation 与 Evaluation 仍关闭。

## Critic V2 primary TEST-preserving LOSO 配置门（2026-08-20）

已前瞻冻结 V2-only primary LOSO protocol/preparer。它只有在 exact
all-Development refit config 与 terminal summary 验证通过后才生成 7 studies ×
3 seeds 的 21 个 configs，不能再像历史路径那样只凭旧 three-seed status 绕过
TEST/refit。每折使用 TRAIN/VALIDATION，排除跨 holdout 的 connected source
components，Development TEST 18,292 条继续 withheld；checkpoint 为固定
100-epoch `FINAL_EPOCH`，held-out study 不参与 checkpoint 选择。

真实 LOSO preparer 和 21 个训练均未运行，因此中央 CSV 不增加行；最近已记录的
100 个唯一 attempts/四个 Critic V2 RUNNING 状态不因本任务改变。新门 focused
tests 12/12 通过；含共享六 GPU pairing 与 trainer split 的扩展 suite 为 70
passed、4 个本机无 CUDA 的既有 skips。TEST/refit/LOSO/readiness/guidance/
Evaluation 仍保持各自冻结边界。

## Critic V2 matched strongest-baseline LOSO 配置门（2026-08-20）

已前瞻冻结 V2-only matched-baseline LOSO protocol/preparer。它读取未来 21 个
primary configs 的配置元数据而非 outcome，并按 holdout study、seed、physical
GPU 和 TEST-preserving split 一一配对。baseline 保留 native 8-epoch FP32、
pairwise-Huber、anchored position-aware policy；matched 不被错误表述为相同容量或
训练预算。每个 config 明确 TEST/TEST-metric selection/Evaluation 均关闭。

真实 baseline configs 和训练未创建，因此中央 CSV 不增加行；最近已记录的 100
个唯一 attempts/四个 Critic V2 RUNNING 状态不因本任务改变。新门 focused tests
15/15、扩展 pairing/aggregation suite 27/27 通过。primary LOSO、baseline LOSO、
readiness、guidance 与 Evaluation 都尚未执行。

## Critic V2 guidance readiness 总闸门（2026-08-20）

已前瞻冻结 V2-only readiness protocol，并实现一次性 evidence packet builder 与
adjudicator。它不再接受旧 V1 signal-control 状态，而是逐级绑定 Critic V2
control、三个固定 seed、单次冻结 TEST、all-126,165 refit、三个固定 seed 的
7-study primary/matched-baseline LOSO 聚合、冻结 reward policy、online encoder
以及 Flow G0 训练/验证与 checkpoint provenance。

三个 LOSO seed 必须分别完整且 model-minus-baseline macro Spearman improvement
大于 0；单次 TEST 数值只报告，不参与任何 readiness 阈值或后续选择。只有
`CRITIC_READY_FOR_GUIDANCE` 与 `FLOW_G0_READY` 同时成立，才会输出 Development
guided generation allowed；该裁决自身不执行 generation，也不建立 biological
optimization claim。

真实 readiness packet/adjudication 尚未创建，TEST/LOSO/Evaluation outcome 均未
因本任务读取，guided generation 仍未授权。因此中央 CSV 不增加训练行；最近已记录
的 100 个唯一 attempts/四个 Critic V2 RUNNING 状态不因本任务改变。新 focused
tests 14/14、与历史 readiness 和 LOSO aggregation 的扩展回归 41/41 通过。

## Critic V2 readiness 论文证据绑定（2026-08-20）

Methods/results evidence packet 已把 V2 control、exact three seeds、单次
report-only TEST、all-Development refit、三 seed matched LOSO 和 critic/Flow 双
readiness 写成一条可核验的 prospective 方法链。没有写入 Critic V2 结果或读取新
outcome；当前文字仍为 control RUNNING、TEST/Evaluation closed、guidance
unauthorized。

更新后的 packet 包含 14 个唯一 claim markers、13 个唯一 evidence sources，引用
全部闭合；focused consistency tests 2/2 通过。该任务没有参数更新，因此中央 CSV
不增加训练行，最近已记录的 100 个唯一 attempts/四个 Critic V2 RUNNING 状态不变。

## Critic V2 Development generation 执行链硬切换（2026-08-20）

审计确认历史 guided、matched-search 与 generation-comparison 入口仍绑定旧 V1
readiness/refit/method/candidate paths，无法消费新 V2 readiness。现已直接切到
V2-only schemas/configs：三个历史 config 标记 retired，现 runners 在 artifact read
前拒绝旧 config；新链精确绑定 V2 readiness、final refit checkpoint、guided
summary/compute、matched candidates 与 V2 guided method identity。

真实 readiness packet、guided candidates、matched candidates 和 comparison output
均未创建；TEST/LOSO/Evaluation outcome 未因本任务读取。三入口 focused tests
16/16 通过。该任务没有参数更新，因此中央 CSV 不增加训练行，最近已记录的 100 个
唯一 attempts/四个 Critic V2 RUNNING 状态不变。

## Critic V2 readiness-to-comparison 合成合同验证（2026-08-20）

新增端到端 focused test，使用真实 builder/adjudicator 输出结构与三份 production
V2 configs，证明 readiness packet/adjudication 可连续通过 guided、matched-search
和 generation-comparison 边界，并验证两端 V2 guided method identity 相同。测试仅
使用合成 evidence 和临时 checkpoint，不读取真实 TEST/LOSO/Evaluation outcome，
也不创建 candidates。

readiness focused suite 15/15、合并下游与 paper 回归 33/33 通过。该验证没有参数
更新，因此中央 CSV 不增加训练行，最近已记录状态不变。

## Critic V2 LOSO aggregation-input 门（2026-08-20）

新增 V2-only LOSO aggregation protocol/input builder，替代只识别历史 V1 run 名称
的旧 builder。新路径从 21 primary + 21 matched-baseline runtime configs 读取 exact
terminal summary paths，核对 study/seed/GPU/primary pairing 与 TEST/Evaluation
boundary，生成三个固定 seed 的共享 aggregation inputs；统计仍由既有 aggregator
完成。

真实 LOSO inputs/results 未创建，任何 LOSO outcome 均未读取。新 builder、共享
aggregator 和 readiness 合并 focused suite 26/26 通过；paper evidence source 数更新
为 14，claim 数仍为 14。该任务没有参数更新，因此中央 CSV 不增加训练行，最近已
记录状态不变。

## Critic V2 paired LOSO stage runner（2026-08-20）

新增 V2-only 六 GPU runner，用未来已准备的 21 primary + 21 matched-baseline configs
构造固定 GPU0-5 queues；每 fold 严格 primary 后 baseline，每卡串行、六卡并行，只以
4096 MiB free-memory 为默认启动门。所有 42 runs 成功后才构建并聚合三个 seed，
任一失败则保留 evidence 并停止。

本次没有调用 runner，没有创建 log/run/input/result，也没有参数更新，所以中央 CSV
不增加训练行。focused tests 8/8 只验证 planning/order/refusal，不执行 GPU 训练；
最近已记录状态不变。
