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

## Critic V2 Development generation stage runner（2026-08-20）

新增 V2-only 单 GPU launcher，在真实 dual readiness 通过后才可从 GPU0-5 中选择
free memory 最大且不少于 4096 MiB 的卡，并严格按 guided XEditFlow → 六方法 matched
search → frozen Development comparison 执行。三份 runtime config 只改变冻结模板的
`device` 与 `physical_gpu_index`；已有 runtime/log/output root 时拒绝覆盖。每个 child
前的等待只使用 free memory，默认 poll 900 秒，任一失败阻止后续 stage。

本次没有调用 launcher，没有查询远端 GPU，没有创建 runtime config、log、generated
candidate 或 comparison output，也没有读取 TEST/LOSO/Evaluation outcome。focused
tests 4/4、相邻合同回归 37/37 通过且均不执行 GPU 训练。没有参数更新，因此中央
CSV 不增加训练行，最近已记录的 100 个唯一 attempts/四个 Critic V2 RUNNING 状态
不因本任务改变。

## Critic V2 post-confirmation 全链 runner（2026-08-20）

新增 V2-only post-confirmation runner 与 900 秒 conditional watcher，补齐 exact
three-seed PASS 后的 single report-only TEST → all-Development refit → 21+21 paired
LOSO → three-seed aggregation → Critic/Flow readiness → readiness-PASS-only Development
generation 顺序。runner 在任何写入前使用 production TEST gate 并拒绝 19 个 future
target 中任何已存在项；TEST/refit 只在 GPU0-5 中按不少于 4096 MiB 的最大 free
memory 选择卡。历史 V1 postselection scheduler 已在入口立即 retired。

代码推送且 A100 通过同组 103/103 回归后，conditional watcher 已启动一份：实际
脚本 PID `380389`（launch wrapper PID `380388`），poll interval 900 秒，首条日志为
`waiting_for_critic_v2_three_seed_adjudication`。启动验收时 19 个 downstream targets
全部不存在。当前只有 scheduler 等待日志新增；runner 未调用，TEST 未打开，stage
runtime/log/run、LOSO、readiness 和 candidates 均未创建，也未读取 TEST/LOSO/
Evaluation outcome。

focused tests 7/7、本机与 A100 完整相邻生产合同回归均为 103/103。watcher 不进行
参数更新，因此中央 CSV 不增加训练行；21:54 最近低频状态仍为 100 个唯一 attempts/
四个 Critic V2 RUNNING，control adjudication absent。

## Development generation bootstrap 论文统计补全（2026-08-20）

只读复核 terminal strongest-generation-baseline artifact 后，新增六行全精度 paired
source bootstrap 表，并同步 paper draft/evidence/consistency manifests。报告明确
analysis unit 为 891 个 source、seed `20260816`、10,000/10,000 defined iterations；
六个 genetic leader-advantage 95% intervals 的 lower bound 均大于 0，最近竞争者为
generate-then-rerank，点差 `0.040228719771844945`，区间
`[0.018934459870160632, 0.06168808431503615]`。解释边界仍限于 Development
independent-evaluator separation。

paper focused tests 3/3，与 frozen selection producer/input 合并回归 17/17 通过。
没有读取 Development TEST/final Evaluation、没有参数更新，因此中央 CSV 不增加
训练行，现有 Critic V2 RUNNING 状态不变。

## Independent evaluator 九 task 论文报告补全（2026-08-20）

只读复核 terminal Development evaluator summary/adjudication 后，新增九行全精度
task-region 表。九组共 18,293 records；Spearman 从
`-0.10916458562634956` 到 `0.7619576378536184`，5 组为正、4 组非正；
`RNA_HALF_LIFE_MINUTES::region=0` standardized MAE `9.220029033415157` 为最大。
paper 明确把该异质性与 narrowly passing macro gate 同时报告，不把 qualification
扩写为均匀 task reliability 或 biological validation。

focused tests 4/4，与 evaluator scorer/adjudicator 合并回归 10/10 通过。没有读取
Development TEST/final Evaluation、没有参数更新，因此中央 CSV 不增加训练行，
现有 Critic V2 RUNNING 状态不变。

## Independent evaluator global spread 论文限制（2026-08-20）

terminal summary 的 raw global prediction/target standard-deviation ratio 为
`0.0022026649600126917`。冻结 evaluator qualification protocol 没有 spread
threshold，因此没有事后改变 terminal qualification；同时，九 endpoint 尺度异质且
per-task prediction/target spread 未持久化，paper 不以该 global ratio 推断每个 task
mean-collapse，并明确不为补齐该字段重跑 terminal evaluator。

paper focused tests 5/5，与 evaluator scorer/adjudicator 合并回归 11/11 通过。没有
读取 Development TEST/final Evaluation、没有参数或 gate 更新，因此中央 CSV 不增加
训练行，现有 Critic V2 RUNNING 状态不变。

## Independent evaluator qualification checks 论文补全（2026-08-20）

新增 terminal adjudication 的 12-row exact qualification-check table，所有冻结 checks
均为 true。`candidate_rerun_authorized=true` 只支持 Development candidate rerun，
不覆盖同一裁决的 `scientific_claim_status=NOT_ESTABLISHED`。

paper focused tests 6/6，与 evaluator scorer/adjudicator 合并回归 12/12 通过。没有
读取 Development TEST/final Evaluation、没有参数更新，因此中央 CSV 不增加训练行，
现有 Critic V2 RUNNING 状态不变。

## Critic V2 post-confirmation 生产只读输入 preflight（2026-08-20）

等待 control terminal 期间，按低频监控边界只检查了 post-confirmation 唯一 runner
所需生产只读输入的文件存在性。冻结协议/config/entrypoint/templates、Development
manifest、八份 canonical records、terminal baseline/online-encoder/Flow inputs 与
Flow checkpoint 共 30 个路径全部存在，missing=0，故无需修复路径或重新部署数据。

路径枚举来自本地冻结协议和 runner 源码；远端 focused preflight 没有打开生产数据
或 outcome 文件内容、训练进度，也没有读取 Development TEST/final Evaluation outcome
或创建 future artifacts。它不进行参数更新，因此中央训练 CSV 不增加伪 attempt；
现有 Critic V2 运行状态不因本任务改变。

## Paper evidence source locator 闭合核查（2026-08-20）

paper evidence manifest 的 15 个 source locator 已完成一次文件存在性 preflight：
本地仓库或合同路径 8/8、A100 `/mnt` 路径 7/7，missing=0。检查不打开 evidence
内容、训练进度、Development TEST 或 final Evaluation outcome。

manifest 明确保留 `human_verification_required=true` 与 `submission_ready=false`，
locator PASS 不冒充人工内容验证。paper focused tests 7/7 通过。本任务没有参数更新，
中央训练 CSV 不增加伪 attempt；现有 Critic V2 运行状态不因本任务改变。

## Critic V2 control terminal NO-GO（2026-08-22）

中央 CSV 中四个 Critic V2 control attempts 已全部原位更新为 `COMPLETED`，没有重复行：

| arm | GPU | epochs | optimizer steps | selected epoch | task-macro Spearman | task-macro standardized MAE |
|---|---:|---:|---:|---:|---:|---:|
| full | 2 | 100 | 559,900 | 98 | 0.1163706632 | 2.2228258513 |
| candidate permutation | 4 | 100 | 559,900 | 12 | 0.0801854624 | 2.3963102262 |
| parameter-matched source-only | 3 | 100 | 559,900 | 1 | 0.0179762355 | 2.0924846890 |
| source+edit metadata、无 candidate global representation | 5 | 100 | 559,900 | 2 | 0.0865578266 | 2.1263093306 |

full 通过相对三个 controls 的冻结信息检查，但未超过 strongest same-information
baseline task-macro Spearman `0.1317143949`，margin `-0.0153437317`。裁决为
`CRITIC_V2_CONTROLS_DO_NOT_SUPPORT_THREE_FROZEN_SEEDS`，所以 three-seed、TEST、
refit、LOSO、readiness 与 guided generation 均未开始。19 个冻结 downstream targets
检查为 0 个存在；Development TEST 与 final Evaluation outcome 仍未读取。

中央表终态汇总为 100 个唯一 attempts：92 `COMPLETED`、3 `FAILED`、3
`INCOMPLETE_NO_TERMINAL_RECORD`、1 `STOPPED_FOR_THROUGHPUT_REPAIR`、1
`STOPPED_PRIORITY_REALLOCATION`。watcher NO-GO 空等修复提交 `990f941` 不发生参数
更新，因此不新增训练 attempt；本机与 A100 focused tests 均为 8/8。

## Critic V2 task-level failure diagnostic（2026-08-22）

只读汇总已 terminal 的五份 Development VALIDATION summaries 后，full 相对 strongest
baseline 的 Spearman 为 4/9 task wins、5/9 losses，task-macro margin
`-0.01534373173869797`；standardized MAE 为 0/9 wins、9/9 losses，macro margin
`+0.4161680105385127`。两个各 48 条的 task 合计 Spearman margin
`-0.21149352683453532`，但它们继续保留在冻结 equal-task gate 中，不作事后删除。
两个 frozen candidate-permutation tasks 的正 margin 则表明 candidate-specific signal
局部存在、但不足以形成稳定跨-task优势。

该诊断不发生参数更新，不新增或修改中央 CSV 行；Development TEST、Evaluation 和
generated candidates 均未读取。全精度 task 表与受限解释见 paper/audit artifacts，
forward route 为 Benchmark+limits negative result。

## Generation action-space geometry 终态证据包（2026-08-22）

只读汇总 terminal v2 matched-generation selection input 的七法 Development 聚合与
per-source geometry，候选数、edit-distance 和 terminal-cause 计数均为 7/7 守恒。
共享边界继续是 891 sources、每 source candidate cap 32、`SUB + STOP`；INS/DEL
不在第一阶段范围。七法 hard legality 均为 1.0，edit/candidate budget violations、
`NO_LEGAL_ACTION` 与 `NUMERICAL_FAILURE` 均为 0。

终止与候选几何存在实质方法差异：greedy/beam explicit-STOP rate 均为
`0.7048260381593715`，unguided Base Flow budget-exhaustion rate 为
`0.8702651515151515`；local search 每 source 实际返回 3--32 个候选，均值
`23.5993265993266`。Flow 的 28,512 candidate rows 中有 25,173 unique、3,339
duplicates。所有方法均为 open generated support，closed measured NDCG defined
source count 为 0，未知 outcome 不按 zero gain 或 canonical credit 处理。

新增全精度表
`docs/paper/route2_v332_generation_action_space_geometry_table_v1.csv` 与审计
`audits/route_a_v3_route2_generation_action_space_geometry_v1.json`，并同步 paper
draft/evidence/consistency manifests。per-candidate algorithmic STOP time 与六个 search
方法的 generation wall time 不在 terminal selection input 中，继续记为
`NOT_RETAINED/NOT_RECORDED`，不重跑或反推。focused suite 本机为 39 passed、
5 skipped、0 failed；A100 完整环境同组为 44/44 passed。

本任务没有参数更新，中央训练 CSV 不增加 attempt；100 个唯一 attempts 的终态仍为
92 `COMPLETED`、3 `FAILED`、3 `INCOMPLETE_NO_TERMINAL_RECORD`、1
`STOPPED_FOR_THROUGHPUT_REPAIR`、1 `STOPPED_PRIORITY_REALLOCATION`。Development
TEST、guided XEditFlow 与 new final Evaluation outcome 均未打开。

## Historical transfer 与 minimum benchmark package 裁决（2026-08-22）

按 V3.3.2 允许的 historical diagnostic 边界读取既有 GSE232572 8,068-record
zero-shot summary；该 study 已 outcome-exposed，当前角色固定为
`HISTORICALLY_OUTCOME_EXPOSED_TRANSFER_DIAGNOSTIC_NOT_FINAL_CONFIRMATION`。三 seed
model-minus-baseline task-macro Spearman 点差为 `+0.020302446832280538 /`
`+0.04853075021317742 / +0.03709351939595837`，但第一 seed 的 paired 95% CI lower
为 `-0.011728708071697243`；三 seed 的 baseline-MAE minus model-MAE 均为负。因此
`preregistered_pass=false`，只支持历史负 transfer/limits 结论，不提供 final confirmation。

新增 18-item minimum package 表与审计。13 项为 complete 或 complete-with-declared-
limits，4 项 partial，1 项 unavailable；阻止“minimum package complete”的五项为：
当前 cohort 不授权 guided/first-order comparison、新 outcome-unexposed replacement
Evaluation 不存在、其 zero-shot→adaptation 未执行、terminal timing 字段不完整、Route 2
manuscript figure builders 尚未建立。当前状态为
`MINIMUM_BENCHMARK_PACKAGE_NOT_COMPLETE`，conditional target 才是
`BENCHMARK_PLUS_TRANSFER_AND_GENERATION_LIMITS_PAPER`，不是已冻结的 submission-ready
outcome。

审计同时记录两个 2026-08-17 inventory 的陈旧字段：旧 GSE232572 `EVALUATION` role、
`LOCAL_BRANCH_NOT_PUSHED`、mRNABERT HPO running 与 historical summary 的 provisional
paper label 均由 V3.3.2 合同和 terminal evidence 覆盖，但不篡改冻结 snapshot。
本机与 A100 focused suite 均为 30/30 passed。没有读取 Development TEST/new final
Evaluation，没有运行 guided 或 E-MTAB-10902 outcome evaluation，也没有参数更新；
中央训练 CSV 不新增 attempt，100-row 终态分布不变。

## Provisional manuscript figure builders（2026-08-22）

本项为非训练 manuscript-asset task，不新增中央训练 attempt。commit `a27e04a`
新增可复现构建器与 focused tests；A100 builder test 2/2 passed，并在
`/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/figures/route2_v332_v1/` 生成两图各
PNG/PDF/SVG、alt text 与 provenance manifest。输入仅为冻结的 Development
generation/action-space、terminal Critic V2 task diagnostic 和 outcome-exposed
GSE232572 historical summary；Development TEST/new final Evaluation/guided 均未打开。

本机与 A100 paper-packet + builder 联合 focused suite 均为 14/14。MBP-17 由无
builder 的 partial 更新为 `COMPLETE_WITH_PROVISIONAL_GENERAL_FIGURES`；最低包汇总为 14 complete/
complete-with-declared-limits、3 partial、1 unavailable，四个 blocker 为 MBP-10、
MBP-13、MBP-14、MBP-15。target journal/article type/submission phase 待定，未声称
publisher compliance；总体 `submission_ready=false`、Route A claim 仍
`NOT_ESTABLISHED`。

中央 100 个唯一 attempts 的终态继续是 92 `COMPLETED`、3 `FAILED`、3
`INCOMPLETE_NO_TERMINAL_RECORD`、1 `STOPPED_FOR_THROUGHPUT_REPAIR`、1
`STOPPED_PRIORITY_REALLOCATION`。

## Prediction/Generation matched-budget baseline matrix（2026-08-22）

本项为非训练 paper-table/audit task，不新增或修改中央 CSV attempt。Goal 7 的 legal
action/STOP-budget/base-guided method figure 已由既有 system architecture figure 与
generation Figure 1 共同 terminal 覆盖，因此不重复 builder 或实验。随后新增一次性只读
terminal compute snapshot、matched-budget matrix builder/focused tests、14-row CSV 与 audit。

Prediction 侧四个 Critic V2 screen arms 在 seed/epochs/updates/parameter scale/split 上 exact
matched；strongest same-information hurdle 只有 22,120 updates，相对 Critic V2 的 559,900
updates 不是 exact compute match。Generation 侧七个 terminal 方法共享 source/action/edit/
candidate/forward caps，但 algorithm-specific training/HPO 未形成共同 numeric budget，六个
search wall times 未记录，两个 guided rows 因 Critic V2 NO-GO 保持无数值未执行。故矩阵
reporting complete，但 full contract-matched execution incomplete，headline matched rows=0。

GitHub builder commit `11d3ec0` 已推送且 A100 focused test 2/2；paper integration commit
`a4d4f64` 推送并同步后，本机与 A100 十一组联合 suite 均为 35/35。paper evidence
sources=41（28 local/contract + 13 `/mnt`）、claim markers=22、
figures/builders=6/5、MBP=14/3/1、blockers=MBP-10/13/14/15、`submission_ready=false`
不变。中央 100 attempts 仍为 92 `COMPLETED`、3 `FAILED`、3
`INCOMPLETE_NO_TERMINAL_RECORD`、1 `STOPPED_FOR_THROUGHPUT_REPAIR`、1
`STOPPED_PRIORITY_REALLOCATION`。Development TEST、new final Evaluation、generated
candidate outcome 与 guided XEditFlow 均未打开。

## Dataset qualification/development table（2026-08-22）

本项为非训练 paper-table task，不新增中央训练 attempt。新增可复现 builder、14-row
CSV、审计 JSON 与 focused tests；输入仅为冻结的 14-study inventory。表中 Development
records=`126,165`、historical outcome-exposed GSE232572=`8,068`、new final Evaluation
unexposed records=`0`、qualified canonical credit=`6,547`，qualified study credit 固定为
ordinary/A1/true-A2=`1/1/0`。GSE232572 的旧 inventory `EVALUATION` role 由 V3.3.2
覆盖为 historical diagnostic；E-MTAB outcome、sealed GSE246381、Development TEST、
new final Evaluation 与 guided 均未打开。

paper packet 仍有 22 个 claim markers，evidence sources 增至 23；本机与 A100
dataset-table + paper-packet + figure-builder 联合 focused suite 均为 17/17。中央 100 个唯一 attempts 的
终态不变：92 `COMPLETED`、3 `FAILED`、3 `INCOMPLETE_NO_TERMINAL_RECORD`、1
`STOPPED_FOR_THROUGHPUT_REPAIR`、1 `STOPPED_PRIORITY_REALLOCATION`。

## Minimum benchmark package itemwise closure（2026-08-22）

本项为非训练 paper-adjudication task，不新增中央 CSV attempt，不读取 Development TEST、
新 final Evaluation、generated candidates 或 guided outcome，也不重跑任何 terminal 方法。
对既有 18-row minimum benchmark package table 作终态语义闭合：18/18 requirements 均有
status、evidence/basis、remaining gap 与 authorized disposition，未裁决行=0，未完成行没有
被写成 PASS，因此 `itemwise_adjudication_complete=true`。

该状态不等于 package success。14 项为 COMPLETE 或 COMPLETE_WITH_DECLARED_LIMITS，3 项
PARTIAL，1 项 NOT_AVAILABLE；MBP-10/13/14/15 继续分别受 Critic V2 dependency NO-GO、
replacement external study unavailable、相应 zero-shot/adaptation 不可执行和 terminal timing
不可追溯限制。故 `minimum_package_complete=false`、`submission_ready=false`，当前 cohort 的
授权动作仍是 `NO_RERUN_NO_GUIDED_NO_PROTECTED_OUTCOME_READ`。

GitHub commit `e6607f6` 已推送并在 A100 一次 fast-forward；本机与 A100 paper-evidence
focused suite 均为 18/18。evidence sources=48（34 local/contract + 14 `/mnt`）、claims=22、
figures/builders=7/6、中央 100-row 92/3/3/1/1 与
`scientific_claim_status=NOT_ESTABLISHED` 均不变。

## Final manuscript-route outcome adjudication（2026-08-22）

本项为非训练 scientific/paper adjudication task，不新增中央 CSV attempt，不读取
Development TEST、新 final Evaluation、generated candidates 或 guided outcome。按主合同
A/B/C 的逐条触发条件冻结 final manuscript route：Outcome A 因最低包不完整、无新的
outcome-unexposed Evaluation、Critic not ready、guided 未运行及无 terminal true-A2/guided
improvement 而不合格；Outcome B 因无稳定 outcome-unexposed external prediction value、
guided Edit Flow comparison 不完整及 benchmark package 不完整而不合格。

合同的 automatic forward rule 因此将 route 冻结为
`BENCHMARK_PLUS_TRANSFER_AND_GENERATION_LIMITS_PAPER`。该冻结只决定 manuscript direction，
不等于投稿资格：`outcome_trigger_fully_satisfied=false`、
`submission_level_outcome_eligibility=false`、`minimum_package_complete=false`、
`submission_ready=false`。新增 outcome audit 同时声明下一代数据需要 source-candidate lineage、
closed dense measured pool、biological replicate + finite positive SE、frozen/balanced context，
以及 outcomes 在全套冻结前未暴露的 independent external study。

GitHub commit `e211212` 已推送并在 A100 一次 fast-forward；本机与 A100 paper-evidence
focused suite 均为 19/19。evidence sources=49（35 local/contract + 14 `/mnt`）、claims=22、
figures/builders=7/6、MBP=14/3/1、中央 100-row 92/3/3/1/1 均不变；model、guided-generation
与 biological success 仍未建立。

## Selected-outcome claim/evidence and unsupported-claim closure（2026-08-22）

本项为非训练 paper-table/audit task，不新增中央 CSV attempt，不读取 Development TEST、新
final Evaluation、generated candidates 或 guided outcome。新增可复现 builder，从 manuscript
draft 自动提取全部 22 个 `C-R2-*` marker、claim text 与邻接 evidence IDs，再绑定显式
scientific boundary；任何 marker 漏配、重复或 unknown evidence ID 均会失败。

正式 35-row × 17-column table 含 22 个
`SUPPORTED_WITH_DECLARED_BOUNDARY` 行和 13 个 `UNSUPPORTED` 行。unsupported 集合覆盖
Outcome A/B headline、biological/guided success、external generation validation、所有方法固定
32 candidates/source、INS/DEL、unknown-as-zero、GSE232572 final confirmation、package/
submission completeness、E-MTAB outcome 和 causal region/context mechanism。13 行全部
`allowed_in_selected_outcome_manuscript=false`。

Spreadsheet 验证确认 35×17、blank evidence=0、unsupported-as-allowed=0、protected 与
completion boundary fields 全部 false。GitHub commit `799e156` 已推送并在 A100 一次
fast-forward；本机/A100 两组 claim/paper focused suite 均为 22/22。evidence sources=51
（37 local/contract + 14 `/mnt`）、draft claim markers=22、figures/builders=7/6、MBP=14/3/1、
中央 100-row 92/3/3/1/1 均不变。

## Canonical conversion flow figure（2026-08-22）

本项为非训练 paper-figure task，不新增中央 attempt。新增独立 builder/focused tests；
只读取 14-row dataset qualification table 与冻结 split counts，不读取 canonical rows、
Development TEST、E-MTAB outcome、sealed GSE246381 或 new final Evaluation outcome。
A100 builder test 2/2 passed，并在既有 `/mnt/.../figures/route2_v332_v1/` 追加一张
PNG/PDF/SVG conversion-flow figure、manifest 和 alt text，未覆盖原两图。

图中精确保留 14 studies、Development 126,165、historical 8,068、new final Evaluation
0、qualified/relaxed/listwise=`6,547/88,652/30,966` 和 split
`89,580/18,293/18,292`；arrow width 明确不编码 magnitude。paper packet 仍有 22 个
claim markers，evidence sources 增至 25；本机与 A100 四组联合 focused suite 均为 19/19。
MBP-17 继续是 `COMPLETE_WITH_PROVISIONAL_GENERAL_FIGURES`，但最低包仍不完整且
`submission_ready=false`。中央 100 个 attempts 的 terminal 分布不变。

## Development/Evaluation architecture figure（2026-08-22）

本项为非训练 paper-figure task，不新增中央 attempt。新增独立 builder/focused tests；
只读取已冻结的 dataset qualification table、method-repair/readiness protocols 与 minimum
package audit，不读取 canonical rows、Development TEST、E-MTAB outcome、sealed GSE246381
或 new final Evaluation outcome。A100 builder test 2/2 passed，并在既有
`/mnt/.../figures/route2_v332_v1/` 追加一张 PNG/PDF/SVG architecture figure、manifest
和 alt text，未覆盖已有图件。

图中保留 Development `126,165` 与 split `89,580/18,293/18,292`，明确 TRAIN 用于拟合、
VALIDATION 用于选择、Critic V2 当前 NO-GO、TEST 仅在 gate 通过后单次 report-only。
GSE232572 保持 historical outcome-exposed diagnostic，E-MTAB-10902 保持 conversion-
failure-only/outcome-unread，GSE246381 保持 sealed/unread；未来 replacement Evaluation
当前 absent、unexposed records=0，且顺序固定为 full freeze → one zero-shot permanently
recorded → only then adaptation，zero-shot headline 不得覆盖。

paper packet 仍有 22 个 claim markers，evidence sources 增至 27；manuscript figures
现为 4 张、3 个 builders，MBP-17 和最低包 14/3/1 汇总不变；本机与 A100 五组联合
focused suite 均为 21/21。中央 100 个 attempts 的 terminal 分布继续是 92
`COMPLETED`、3 `FAILED`、3 `INCOMPLETE_NO_TERMINAL_RECORD`、1
`STOPPED_FOR_THROUGHPUT_REPAIR`、1 `STOPPED_PRIORITY_REALLOCATION`。

## V3.3.2 NATIVE / COMMON / ARCH_CONTROLLED 三轨结果表（2026-08-22）

本项为非训练 paper-table/audit task，不新增中央 attempt。只读提取显式冻结的 A100
Development-validation 小型 JSON 字段，形成版本化 terminal-input snapshot；未读取
Development TEST、新 final Evaluation、generated candidates 或 guided outcome。

新增 52-row 三轨表：`NATIVE_REPRODUCTION / COMMON_SOURCE_RELATIVE_TASK /
ARCH_CONTROLLED` 行数为 10/12/30，数值结果行为 0/9/26。Track A 只保留 official/native
parity、smoke、literature/task-mismatch 状态，没有原论文 native numeric reproduction，且
不能进入当前 headline。Track B 的 8 个 headline-eligible 行只允许在各自相同的 5′UTR、
3′UTR 或 nine-task scope 内横向比较。Track C 保留 aligned-A1 未 materialize、scratch/frozen
非因果匹配、generic-trunk/region-adapter 缺项和两个 guided NO-GO。

GitHub builder commit `e2a9b63` 已推送，A100 同步后 focused test 2/2；paper integration
commit `99136f6` 后，本机与 A100 八组联合 suite 均为 27/27。paper evidence sources
从 31 增至 34（local/contract 22、A100 `/mnt` 12），
claim markers=22、figures/builders=5/4、MBP=14/3/1、blockers=MBP-10/13/14/15 和
`submission_ready=false` 不变。中央 100 个 attempts 的 92/3/3/1/1 终态分布不变。

## Predictor–Legal XEditFlow–Independent Evaluator architecture figure（2026-08-22）

本项为非训练 paper-figure task，不新增中央 attempt。新增独立 builder/focused tests；
只读取冻结的 critic/Base Flow/evaluator configs、evaluator qualification 与 reward policy，
以及小型 terminal freshness/geometry/minimum-package audits。旧 generation-readiness audit
中的 205,717-parameter Base Flow 被识别为陈旧来源并排除，未覆盖 V3.3.2 的 V2 状态。

图中分离 frozen mRNABERT 113,389,056 + trainable edit-centered Delta head 9,342,914、
position/progress `SUB+STOP` Base Flow 和 distinct Siamese evaluator 509,845 actual
parameters。Base Flow 只标 `FLOW_G0_READY` engineering；critic guidance 为虚线 prospective
且 current NO-GO；evaluator 只作 Development method selection。critic self-score、
independent evaluator score 与 unavailable measured outcome 分列；无 evaluator→generator
gradient、无 generator→critic gradient、生成时 critic 不更新，generated candidates 不加
canonical credit。

A100 builder test 2/2 passed，并在 `/mnt/.../figures/route2_v332_v1/` 追加 PNG/PDF/SVG、
manifest 与 alt text。paper packet 仍有 22 个 claim markers，evidence sources 增至 29；
manuscript figures 现为 5 张、4 个 builders；本机与 A100 六组联合 focused suite 均为
23/23，最低包与 100-row terminal 分布不变。

## V3.3.2 Prediction/Generation baseline inventory matrix（2026-08-22）

本项为非训练 paper-table/audit task，不新增中央 attempt，也不读取训练进度、Development
TEST、新 final Evaluation、generated candidates 或 guided outcome。新增可复现 builder、
45-row CSV、独立 audit 和 2-test focused suite；矩阵将 Prediction 34 行分为 internal
controls 11、classical 6、neural 7、task-specific/foundation 10，并列出 Generation 11 行。

矩阵明确保留 standalone study mean 的 composite mapping、absolute-candidate component
限制、ordinal/listwise 两项 `CONFIGURED_NOT_TERMINAL_INDEPENDENT_BASELINE`、六个已执行
common-task adapters 与三个 literature/task-mismatch references。Generation 侧以当前
terminal table 覆盖旧 inventory：七个 matched methods terminal，exhaustive 仅 small-space
reference，first-order 与 frozen-critic XEditFlow 因 Critic V2 NO-GO 未运行，masked discrete
flow/diffusion 仅为 task-mismatch 文献对照。`matrix_is_result_table=false`，没有结果指标列。

本机 builder focused test 2/2；本机与 A100 七组联合 suite 均为 25/25。A100 从 GitHub
commit `589d263` 快进后 builder focused test 2/2，并在 paper integration commit
`4734eae` 后完成联合复核。paper packet evidence sources 由 29 增至 31（本地/合同
19，A100 `/mnt` 12），claim markers 仍为 22，figures/builders 仍为 5/4；MBP 14/3/1、
四个 blockers 和 `submission_ready=false` 不变。中央 100 个 attempts 的终态继续是 92
`COMPLETED`、3 `FAILED`、3 `INCOMPLETE_NO_TERMINAL_RECORD`、1
`STOPPED_FOR_THROUGHPUT_REPAIR`、1 `STOPPED_PRIORITY_REALLOCATION`。

## Frozen Development learning curves（2026-08-22）

本项为非训练 paper-figure task，不新增或修改中央 CSV attempt。新增独立 builder/focused
tests，只读取六个 selected predictor、Critic V2 四个 control arms、independent evaluator
与 Base Flow G0 的 terminal Development histories；不监控实时训练，不读取 Development
TEST/new final Evaluation，也不运行 guided XEditFlow。所有曲线保持 raw/unsmoothed。

预测器 8-epoch 曲线明确使用 pooled Validation Spearman，图例中的 final architecture
selection 数值明确使用另算的 task-macro Spearman，禁止混用或跨 panel 排名。Critic full
selected epoch 98 仍低于 strongest same-information baseline，terminal NO-GO 不变；
evaluator 只保留 Development method-selection qualification；Base Flow 30-epoch NLL 图
明确显示 epoch 1 后 validation loss 恶化，仍只属 engineering component。

GitHub commits `f28d04f`/`9659da7` 已推送，A100 builder focused test 2/2 passed；正式
PNG/PDF/SVG、manifest 与 alt text 已保存到 `/mnt/.../figures/route2_v332_v1/`，视觉和
矢量/字体/不透明审计通过。paper integration commit `f94372b` 后，evidence sources=36
（23 local/contract + 13 `/mnt`）、claim markers=22、figures/builders=6/5；本机与 A100
九组联合 focused suite 均为 29/29。MBP 14/3/1、
四个 blockers、`submission_ready=false` 与中央 100-row 的 92/3/3/1/1 终态均不变。

## A1 numeric-task / true-A2 availability and result-boundary table（2026-08-22）

本项为非训练 paper-table/audit task，不新增或修改中央 CSV attempt，也不读取训练进度、
Development TEST、新 final Evaluation、generated-candidate outcome 或 guided XEditFlow。
已有 historical zero-shot figure 已满足对应交付项；replacement Evaluation study 不存在，
所以不伪造 few-shot adaptation。新增 builder/focused tests、14-row CSV 与 audit，严格拆分
9 个 A1 numeric Development Validation rows 和 5 个 true-A2 status/boundary rows。

A1 共 18,293 records，5/9 task Spearman 为正；true-A2 侧虽有 GSE269595 的 30,966
Development-exposed listwise records、已实现 evaluator 和已配置 listwise ranker，但
qualified study credit=0、independent terminal numeric result=0 rows。open generated support
下 closed measured NDCG 对七方法均无 defined source；new independent Evaluation records=0。
缺失 true-A2 numeric values 留空，不以零增益替代，不与 A1 作跨 estimand 排名。

GitHub builder commit `363c741` 已推送且 A100 focused test 2/2；paper integration commit
`410053d` 推送并同步后，本机与 A100 十组联合 suite 均为 32/32。paper evidence
sources=38（25 local/contract + 13 `/mnt`）、claim markers=22、
figures/builders=6/5、MBP=14/3/1、blockers=MBP-10/13/14/15 与
`submission_ready=false` 不变。中央 100 个 attempts 仍为 92 `COMPLETED`、3 `FAILED`、
3 `INCOMPLETE_NO_TERMINAL_RECORD`、1 `STOPPED_FOR_THROUGHPUT_REPAIR`、1
`STOPPED_PRIORITY_REALLOCATION`。

## Generation critic / independent / measured 三层结果表（2026-08-22）

本项为非训练 paper-table/audit task，不新增或修改中央 CSV attempt。按低频监控约束只做
一次权威 terminal selection input 的聚合字段核查；没有查询训练/GPU 进度，没有展开
generated candidate payload，也没有读取 Development TEST、新 final Evaluation 或 guided
outcome。新增只读 aggregate snapshot、独立 builder、focused tests、9-row × 31-column CSV
和 audit；Spreadsheet 类型/空值检查确认所有 undefined 指标与 guided NO-GO 数值保持空白。

七个已执行方法均有独立 evaluator 与稀疏 measured-recovery 汇总；六个 critic-driven 方法
具有 891/891 source 的 guiding-critic self-score，unguided Base Flow 按设计无 critic 调用。
genetic 的 source-macro critic max uplift 与 independent-evaluator max uplift 分别为
`1.1912207428186161` 和 `1.0978248587628674`，两层均为最高；unguided Base Flow 的
candidate recovery 为 `0.20286195286195285`，高于 genetic 的 `0.05443322109988777`。
所有方法 closed measured NDCG defined-source count=0；conditional recovered NDCG 的支持数
为 11--400，不作为 closed-support 横向排名。两个 guided 方法仍为 Critic V2 NO-GO。

GitHub builder commit `adcd9d4` 与 paper integration commit `3af0ac0` 已推送；A100 从
`a4d4f64` 一次性 fast-forward 到 `3af0ac0`，本机与 A100 十二组联合 focused suite 均为
38/38。paper evidence sources=44（31 local/contract + 13 `/mnt`）、claim markers=22、
figures/builders=6/5、MBP=14/3/1、blockers=MBP-10/13/14/15 与
`submission_ready=false` 不变。中央 100 个 attempts 仍为 92 `COMPLETED`、3 `FAILED`、
3 `INCOMPLETE_NO_TERMINAL_RECORD`、1 `STOPPED_FOR_THROUGHPUT_REPAIR`、1
`STOPPED_PRIORITY_REALLOCATION`；`scientific_claim_status=NOT_ESTABLISHED`。

## Generation diversity / quality–cost / failure analysis figure（2026-08-22）

本项为非训练 paper-figure task，不新增或修改中央 CSV attempt，也不查询训练/GPU 进度。
交叉核查确认既有 Figure 1 已覆盖 evaluator/recovery、STOP/budget-exhaustion 和 candidate/
duplication，但未把 quality 与 forward-equivalent cost 放在同一坐标，也未单独呈现 Hamming
diversity；因此新增独立 builder、2 项 focused tests 和四面板 provisional figure，而不重跑
七个 terminal 方法。输入仅为冻结的 7-row action-space geometry CSV 与 audit；未读取
candidate payload、Development TEST、新 final Evaluation 或 guided outcome。

独立 evaluator point-estimate quality–cost frontier 为 random legal / unguided Base Flow /
genetic；sparse measured candidate-recovery frontier 为 random legal / unguided Base Flow。
random legal 成本最低（64.00448933782268 mean forward-equivalents/source），genetic 独立
evaluator uplift 最高（1.0978248587628674），Base Flow measured recovery 与 Hamming diversity
最高（0.20286195286195285 / 0.0765737532452552）。Base Flow duplicate fraction 为
0.11710858585858586；local-search cap shortfall fraction 为 0.2625210437710438。所有方法
hard legality=1.0，edit/candidate-budget、no-legal-action、numerical failures 均为 0；无
per-method uncertainty、完整 wall time 或 closed measured NDCG，不做 formal superiority claim。

GitHub builder commit `559952c` 与 paper integration commit `e7af043` 已推送；A100 正式
PNG/PDF/SVG、manifest 与 alt text 保存到 `/mnt/.../figures/route2_v332_v1/`。PNG 为
2160×2100、约 300 dpi、全不透明；PDF 无 image resource 且含嵌入字体，SVG 无 raster
image。该检查不等于 publisher compliance；target journal/article type/phase 仍 pending。
本机与 A100 十三组联合 focused suite 均为 41/41。paper evidence sources=46（32
local/contract + 14 `/mnt`）、claim markers=22、figures/builders=7/6、MBP=14/3/1、
blockers=MBP-10/13/14/15、`submission_ready=false` 与中央 100-row 92/3/3/1/1 均不变。

## Error / domain-shift analysis（2026-08-22）

本项为非训练 paper-table/audit task，不新增或修改中央 CSV attempt，也不查询训练/GPU
进度。只读取已经封存的九任务 Development Validation critic/evaluator 汇总、converter
元数据和 outcome-exposed GSE232572 历史 zero-shot 汇总；未读取 Development TEST、新
final Evaluation、generated candidates 或 guided outcome。

新增 builder、2 项 focused tests、12-row × 41-column CSV 与独立 audit。前 9 行按 study、
assay、region 和 aggregate context 汇总 Development task error；后 3 行单独保留 GSE232572
三个历史 seed，跨层不可用指标全部留空。Spreadsheet 类型/空值检查确认 9 个 Development
行不含 historical metrics，3 个 historical 行不含 Development critic/evaluator metrics，
12 行均为 `external_confirmation_eligible=false`。

Critic V2 相对 strongest same-information baseline 的 task-level Spearman 为 4 胜 5 负，
standardized MAE 为 0 胜 9 负；九任务平均 margin 分别为 -0.01534373173869797 和
+0.4161680105385127。task n 为 48--12,048（251 倍）。5′UTR/3′UTR 的 descriptive
region summaries 分别来自 4/5 个不同 task，并被 study、assay、context、endpoint 和 task
size 混杂；persisted terminal aggregates 不能识别 within-assay context effect。GSE232572
仍有 2/3 rank-improvement CI 下界大于 0，但三条 seed 的 baseline-MAE minus model-MAE
均为负，`preregistered_pass=false`，只作 negative historical transfer。

GitHub commit `39ce66d` 已推送并在 A100 一次 fast-forward；本机与 A100 十四组联合
focused suite 均为 44/44。paper evidence sources=48（34 local/contract + 14 `/mnt`）、
claim markers=22、figures/builders=7/6、MBP=14/3/1、blockers=MBP-10/13/14/15、
`submission_ready=false` 与 `scientific_claim_status=NOT_ESTABLISHED` 不变。中央 100 个
attempts 仍为 92 `COMPLETED`、3 `FAILED`、3 `INCOMPLETE_NO_TERMINAL_RECORD`、1
`STOPPED_FOR_THROUGHPUT_REPAIR`、1 `STOPPED_PRIORITY_REALLOCATION`。

## V3.3.2 Data / rights / exposure limitations closure（2026-08-22）

按 Goal 7 顺序完成 data/rights/exposure limitation 交付。本项为非训练 paper-table/audit
task，不新增或修改中央 CSV attempt，不查询 GPU/训练进度，不读取 Development TEST、new
final Evaluation、sealed payload、generated-candidate outcome 或 guided outcome。新增可复现
builder、2 项独立 focused tests、14-row × 22-column CSV 与 audit，并将逐研究边界写入
manuscript、evidence/consistency manifests 和 MBP-18。

当前转换器或 preflight 记录中有 1 个 `true`、8 个 `false` 与 5 个无 Boolean 的公开再分发
声明；唯一 `true` 为 GSE217518 的 converter output policy，不是经人工验证的数据许可证。
当前 14-study inventory 没有逐研究绑定、accountable-human-verified license registry；所以
14/14 仍为 `HUMAN_REVIEW_PENDING`，public study-payload release authorization=0/14。当前
publication boundary 仅允许 aggregate results 与 source locators，不能声称 open-data package。

Spreadsheet artifact import/render 分三段覆盖全部 22 列，15 行含表头，无错列、公式错误或
关键边界丢失；GSE217518 的 declared `true` 与 `public_release_authorized=false` 同行保留。
GitHub core commit `fe3fb6b` 已推送，A100 自 `799e156` 一次 fast-forward；本机和 A100
全部 V3.3.2 paper/table/figure suite 均为 51/51。paper evidence sources=53（39
local/contract、14 `/mnt`）、claims=22、figures/builders=7/6、MBP=14/3/1、
blockers=MBP-10/13/14/15、`submission_ready=false` 不变；中央 100-row 仍为 92
`COMPLETED`、3 `FAILED`、3 `INCOMPLETE_NO_TERMINAL_RECORD`、1
`STOPPED_FOR_THROUGHPUT_REPAIR`、1 `STOPPED_PRIORITY_REALLOCATION`。

## V3.3.2 Methods section completion（2026-08-22）

按 Goal 7 顺序完成 Methods 内部证据稿闭包。本项不训练、不轮询 GPU/训练进度，不读取
Development TEST、new final Evaluation、sealed payload、generated-candidate outcome 或 guided
outcome。Methods 现有 14 个固定 subsection，集中报告 two-track design、数据角色与 estimand、
baseline/三轨/matched-budget、evaluator、七种 generation methods、统计 analysis unit 与
missingness、Critic V2 controls/training policy，以及 TEST/refit/LOSO/guidance 的条件顺序。

修正真实陈旧事实：provisional figure assembly 从“5 builders/6 figures”更新为终态 6/7，并
补入 quality-cost/diversity/failure figure。独立 evaluator config 的 frozen expected parameter
count 为 509,905，terminal actual 为 509,845，相差 -60；正文报告 terminal actual，历史配置
不回写。Methods completion audit 保持 human evidence/literature verification、authorship、funding、
ethics applicability、Data/Code Availability、venue rules、minimum package 与 submission 全部未完成。

claim/evidence builder 新增显式 `--overwrite` 并重建 35-row 表；默认仍拒绝隐式覆盖，22 supported
markers、13 unsupported claims 和所有 protected fields 不变。GitHub core commit `7ae4e57` 已
推送，A100 自 `fe3fb6b` 一次 fast-forward；本机/A100 全部 V3.3.2 suite 均为 54/54。
evidence sources=54（40 local/contract、14 `/mnt`）、claims=22、figures/builders=7/6、
MBP=14/3/1、blockers=MBP-10/13/14/15、`minimum_package_complete=false`、
`submission_ready=false` 与中央 100-row 92/3/3/1/1 终态不变。

## V3.3.2 Results section completion（2026-08-22）

按 Goal 7 顺序完成 Results 内部证据稿闭包。本项不训练、不轮询 GPU/训练进度，不读取
Development TEST、new final Evaluation、sealed payload、generated-candidate outcome 或 guided
outcome。Results 的 15 个 subsection 已覆盖 baseline/三轨/A1-true-A2/matched budget、独立
evaluator、七方法 generation、action geometry、quality-cost、error/domain shift、historical
transfer、learning curves、Critic V1/V2 NO-GO、minimum package 与 final manuscript route。

没有新增或重解释终态数值。completion audit 固定：45-row baseline inventory、52-row three-track、
9 个 A1 numeric tasks、0 个 terminal true-A2 numeric rows、0 个 fully contract-matched headline rows、
Critic V1 仅 1/3 seed margin positive、Critic V2 margin -0.015343731738697977、historical
`preregistered_pass=false`、new unexposed Evaluation records=0。Results section complete 不建立
model、biological、external-transfer 或 guided success，也不改变 14/3/1 package 和四个 blockers。

GitHub core commit `86e63bf` 已推送，A100 自 `7ae4e57` 一次 fast-forward；本机/A100 全部
V3.3.2 suite 均为 57/57。evidence sources=55（41 local/contract、14 `/mnt`）、claims=22、
figures/builders=7/6、MBP=14/3/1、blockers=MBP-10/13/14/15、
`minimum_package_complete=false`、`outcome_trigger_fully_satisfied=false`、
`submission_ready=false` 与中央 100-row 92/3/3/1/1 终态不变。

## V3.3.2 Discussion section completion（2026-08-22）

按 Goal 7 顺序完成 Discussion 内部证据稿闭包。本项不训练、不轮询 GPU/训练进度，不读取
Development TEST、new final Evaluation、sealed payload、generated-candidate outcome 或 guided
outcome。Discussion 从单一 next-data 草稿扩展为 5 个 subsection，并闭合为
`COMPLETE_INTERNAL_HUMAN_VERIFICATION_PENDING`。

五节严格分开：（1）Benchmark+limits 的证据贡献与未达到的 outcome/package 边界；（2）局部
candidate-specific rank signal 与 strongest same-information baseline 门槛；（3）genetic 的独立
evaluator point lead 与 unguided Base Flow 的 sparse measured-recovery point lead；（4）task/study/
assay/region/context 异质性和 outcome-exposed GSE232572 的非因果、非 final-confirmation 边界；
（5）下一批 confirmatory cohort 的 source/candidate、closed measured pool、replicate/SE、context
和 outcome-unexposed Evaluation 要求。没有新增 claim marker，也没有重解释 terminal 数值。

Discussion completion audit 保持 causal mechanism、model/biological/external-transfer/guided success、
minimum package、outcome trigger 与 submission readiness 全部 false。GitHub core commit `e6e807a`
已推送，A100 自 `86e63bf` 一次 fast-forward；本机与 A100 精确 V3.3.2 suite 均为 60/60。
A100 首次测试选择命令因远端无 `rg` 而误收集旧全测试树，该次运行无效且未作回归结论；改用
可用的精确文件选择后通过 60/60。evidence sources=56（42 local/contract、14 `/mnt`），
claims=22、figures/builders=7/6、MBP=14/3/1、blockers=MBP-10/13/14/15，中央 100-row
92/3/3/1/1、`minimum_package_complete=false`、`outcome_trigger_fully_satisfied=false` 与
`submission_ready=false` 均不变。

## V3.3.2 Data Availability section completion（2026-08-22）

按 Goal 7 顺序完成 Data Availability 内部 statement。本项不训练、不轮询 GPU/训练进度，不读取
Development TEST、new final Evaluation、sealed payload、generated-candidate outcome 或 guided
outcome。章节将第三方 source accession/locator、小型 version-controlled aggregate evidence 和
`/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/` 下的大型非公开产物明确分层。

当前清单中 accountable-human-verified study-bound license=0，14/14 license rows 仍待人工审查，
public study-payload release authorization=0/14。正文不把 locator 写成当前 access/reuse authority，
不把工作分支写成永久 archive/open-data release，也不声明 study payload、model weights 或 generated
candidates 已公开；没有 availability-on-request promise、persistent repository identifier 或未来开放
承诺。Data Availability section complete 只表示内部诚实陈述闭合，rights/content review、public
release readiness 与 submission readiness 仍未完成。

GitHub core commit `2faff4a` 已推送，A100 自 `e6e807a` 一次 fast-forward；本机/A100 全部
V3.3.2 suite 均为 63/63。evidence sources=57（43 local/contract、14 `/mnt`），claims=22、
figures/builders=7/6、MBP=14/3/1、blockers=MBP-10/13/14/15，中央 100-row 92/3/3/1/1、
`minimum_package_complete=false`、`outcome_trigger_fully_satisfied=false` 与
`submission_ready=false` 均不变。

## V3.3.2 legacy tracked-payload formal-release disposition audit（2026-08-22）

按 internal GitHub RC blocker 顺序完成 5 个 legacy tracked payload 的只读 release-disposition
核查。本项不训练、不轮询 GPU/训练进度，不读取 payload 内容、Development TEST、new final
Evaluation、sealed GSE246381、E-MTAB-10902 outcome、generated-candidate outcome 或 guided
outcome；中央 attempt 表没有新增行，仍为 100 total / 92 completed / 3 failed /
3 incomplete / 1 stopped-throughput / 1 stopped-priority。

路径/tracking/size/text-reference 核查确认：`data_registry/excel_inventory.parquet` 为 46,498 bytes；
4 个 superseded B0 JSONL 合计 34,739,577 bytes；5 文件总计 34,786,075 bytes。它们全部仍 tracked，
与 V3.3.2 formal Git payload boundary 冲突。Parquet 有历史 producer
`scripts/data/import_excel_inventory.py`，没有当前 Route 2 consumer；四个 B0 JSONL 仍被 4 个可调用
legacy entrypoints 直接读取：`audit_split_manifests.py`、`eval_tracks.py`、`leakage_audit.py` 和
`fm0_exposure_audit.py`。旧 v3.1 合同要求 old B0 为 `SUPERSEDED_NOT_LOADABLE`，但当前没有找到
active-loader negative-test evidence，因此先前仅写“禁止 active load”的说明不足以证明实现闭合。

推荐顺序固定为：在用户明确授权后，先为 4 个 legacy readers 增加 fail-closed guards/negative tests，
保持 JSONL 不修改；再将 5 个 payload 保存在
`/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/legacy_repository_payloads/` 并把 Parquet producer 默认
输出移出 Git tree；随后才停止 current HEAD tracking 并加入窄范围 ignore；最后重新裁决 RC。共享 Git
history rewrite 不在本任务建议内，若要执行必须另行明确授权。当前没有 copy/delete/move、history
rewrite、reader behavior change、formal tag 或 GitHub Release。

Code Availability 已修正为：Route 2 runtime artifacts 位于 `/mnt`，但当前 working tree 仍含 5 个
legacy data payload，formal-release payload boundary 不合规。新增 disposition audit/memo 与 4 项
focused tests；legacy/Code Availability/RC/evidence focused tests=31/31。本机/A100 精确 V3.3.2 suite
均为 85/85。GitHub core commit `5bd4424` 已推送；A100 自 `b93ba20` 一次 fast-forward 到该 commit。
evidence=65（51 local/contract、14 `/mnt`），claims=22、figures/builders=7/6、MBP=14/3/1、
blockers=MBP-10/13/14/15、formal release/tag=false、payload migration authorized=false、
`minimum_package_complete=false`、`outcome_trigger_fully_satisfied=false` 与
`submission_ready=false` 均不变。

## V3.3.2 Code Availability section completion（2026-08-22）

按 Goal 7 顺序完成 Code Availability 内部 statement。本项不训练、不轮询 GPU/训练进度，不读取
Development TEST、new final Evaluation、sealed payload、generated-candidate outcome 或 guided
outcome。正文绑定 GitHub repository locator 和工作分支，但不把 locator/working branch 写成
unauthenticated public access、immutable release 或 permanent archive。

仓库存在 `pyproject.toml`、`requirements-lock.txt`、`environment.yml` 与 `Dockerfile`，记录
Python 3.10 / PyTorch 2.5.1 环境；但 Route 2 V3.3.2 全证据流程尚未从这些描述执行独立 clean-
environment reproduction 或 accountable-human verification。README authority header 仍早于当前
Route A V3.3.2 合同；当前 HEAD 无 Route A release tag、persistent archive ID 或 archived container。
package metadata 为 `Proprietary`，无 tracked standalone `LICENSE`，因此 repository access 不写成
reuse grant。`/mnt/.../route2/` 大型 artifacts 不属于 Git code release；无 code-on-request promise。

GitHub core commit `ceeae40` 已推送，A100 自 `2faff4a` 一次 fast-forward；本机/A100 全部
V3.3.2 suite 均为 66/66。evidence sources=58（44 local/contract、14 `/mnt`），claims=22、
figures/builders=7/6、MBP=14/3/1、blockers=MBP-10/13/14/15，中央 100-row 92/3/3/1/1、
`minimum_package_complete=false`、`outcome_trigger_fully_satisfied=false`、public code release 与
`submission_ready=false` 均不变。

## V3.3.2 internal GitHub branch release candidate（2026-08-22）

按当前 Git/论文顺序形成内部 GitHub 分支候选。本项不训练、不轮询 GPU/训练进度，不读取
Development TEST、new final Evaluation、sealed payload、generated-candidate outcome 或 guided
outcome；合同只要求 logical-task focused tests/commit/push，并未在 package incomplete 时授权 formal
GitHub Release/tag，因此状态固定为
`INTERNAL_GITHUB_BRANCH_CANDIDATE_ASSEMBLED_FORMAL_RELEASE_NOT_AUTHORIZED`。

README 顶部新增 Route A V3.3.2 branch notice，明确后续 repository-wide v2 内容只作背景；论文
标题更新为 Benchmark+limits evidence manuscript，Data/Code Availability 的 internal review-pending
状态进入 header。RC audit 绑定 Methods 14、Results 15、Discussion 5、claims 22 supported markers /
13 unsupported rows、figures/builders 7/6、evidence 59 与 MBP 14/3/1 四 blockers；formal release、
tag、archive、public access、submission authorization 均 false。

tracked-file 审计发现既有 `data_registry/excel_inventory.parquet`（46,498 bytes）和 4 个 legacy B0
JSONL（合计 34,739,577 bytes）。它们违反当前 formal Git release boundary，但又被历史脚本/审计
引用，旧 v3.1 preservation rule 要求四个 B0 files 原样保留且禁止 active load；本项不打开内容、
不删除、不迁移，作为 formal-release blocker 登记。GitHub core commit `52a41cb` 已推送，A100 自
`ceeae40` 一次 fast-forward；本机/A100 V3.3.2 suite 均为 69/69。中央 100-row 92/3/3/1/1、
MBP blockers=MBP-10/13/14/15、`minimum_package_complete=false`、
`outcome_trigger_fully_satisfied=false` 与 `submission_ready=false` 均不变。

## V3.3.2 official provider rights evidence and FAIR gap closure（2026-08-22）

按 Data Availability 后续顺序完成 14-study 官方 provider policy/accession evidence 表。本项不训练、
不轮询 GPU/训练进度，不读取 Development TEST、new final Evaluation、sealed GSE246381、
E-MTAB-10902 outcome、generated-candidate outcome 或 guided outcome；中央 attempt 表没有新增行，
仍为 100 total / 92 completed / 3 failed / 3 incomplete / 1 stopped-throughput /
1 stopped-priority。

outcome-free 官方字段核查确认 12 个 GEO accession 均由 NCBI 官方 accession 服务返回，1 个 ENCODE
accession 的官方 landing route 返回 HTTP 200，1 个迁移后的 E-MTAB accession 由 BioStudies API
返回。NCBI 一般政策支持访问、使用与分发，但明确保留 submitter IP 例外；ENCODE 一般政策支持下载、
分析与发表，但未逐项裁决本项目重打包；BioStudies 的 new-dataset CC0 范围没有追溯套用于迁移后的
E-MTAB record，且所选 record-specific rights 字段中 license/release field 均为 0。因此 14/14
analysis/publication/citation routes 可记录，但 study-specific license=0/14、project payload
redistribution authorization=0/14、accountable human review pending=14/14。

新增 14×36 CSV、冻结的 provider-source snapshot、可重复 builder、audit 和 4 项 focused tests；
Data Availability 扩展为 5 个证据绑定段落。FAIR 当前只支持 findable=14、metadata-accessible=14，
interoperable-metadata-assessed=0、reusable-license-complete=0；target journal 仍未选定，Springer
Nature policy 只作 baseline。工作簿技能要求的 artifact-operation marker 在当前 workspace 缺失，
因此没有绕过限制生成 XLSX，也没有声称完成视觉工作簿审查；改用项目原生 CSV schema、全列文本、
布尔/公式样式值与可重复生成测试验证，结果为 14 rows × 36 columns、14 unique studies、0 formula-
like cells。

GitHub core commit `2adb8c3` 已推送；A100 工作树自 `52a41cb` 一次 fast-forward 到 `2adb8c3`。
本机/A100 精确 V3.3.2 suite 均为 73/73，focused integration suite 为 31/31。evidence=62
（48 local/contract、14 `/mnt`），claims=22、figures/builders=7/6、MBP=14/3/1、
blockers=MBP-10/13/14/15、formal GitHub release/tag=false、`minimum_package_complete=false`、
`outcome_trigger_fully_satisfied=false` 与 `submission_ready=false` 均不变。

## V3.3.2 accountable-human study-rights review packet preparation（2026-08-22）

按 provider-rights evidence 后续顺序完成 14-study accountable-human review packet 的机器侧准备。
本项不训练、不轮询 GPU/训练进度，不读取 Development TEST、new final Evaluation、sealed
GSE246381、E-MTAB-10902 outcome、generated-candidate outcome 或 guided outcome；中央 attempt 表
没有新增行，仍为 100 total / 92 completed / 3 failed / 3 incomplete /
1 stopped-throughput / 1 stopped-priority。

packet 为 14 rows × 42 columns：17 个冻结 machine-evidence fields、20 个人工判断/签署 fields 和
5 个 protected-outcome fields。当前 14 rows 全部 `PENDING`，`COMPLETED=0`、`HOLD=0`、
accountable signoff=0、target-journal policy checked=0、exact-file redistribution review authorization=0。
人工姓名、角色、机构、日期、study-specific rights source、license/terms、analysis/publication decision、
redistribution decision、exact-file scope、target journal、Data Availability approval 和 signoff 均保持空白；
agent 没有代替 accountable human 填写或裁决。

同一 builder 支持人工填表后的 completeness audit：机器证据不可改写；`COMPLETED` 必须具备身份、日期、
rights source、non-outcome metadata/content scope、citation、use/redistribution、target-journal policy、
Data Availability wording 和 signoff；exact-file authorization 必须列出精确文件范围；`PENDING` 或
`HOLD` 不能授权 exact files。即使未来 14 rows 全部完成，review completion 也不自动成为 project
release authorization，stable repository/version、code license、legacy tracked-payload policy 和最终
release decision 仍是独立 gate。

spreadsheets skill 的 artifact-operation marker 在 workspace 中仍缺失并返回 `MODULE_NOT_FOUND`；
没有猜测路径、安装依赖、调用替代 workbook library、生成 XLSX 或声称视觉验证。bundled Python
只读 CSV 核查通过：14×42、14 unique studies、machine/human/protected=17/20/5、0 formula-like
cells。GitHub core commit `b93ba20` 已推送；A100 自 `2adb8c3` 一次 fast-forward 到该 commit。
validator focused tests=8/8，本机/A100 精确 V3.3.2 suite 均为 81/81。evidence=64
（50 local/contract、14 `/mnt`），claims=22、figures/builders=7/6、MBP=14/3/1、
blockers=MBP-10/13/14/15、formal release/tag=false、human review complete=false、
`minimum_package_complete=false`、`outcome_trigger_fully_satisfied=false` 与
`submission_ready=false` 均不变。

## V3.3.2 legacy B0 active-loader fail-close（2026-08-22）

按旧 v3.1 preservation rule 与当前 formal-release disposition 顺序，完成 4 个 legacy B0 direct
reader 的窄范围 fail-close。本项不训练、不轮询 GPU/训练进度，不读取 Development TEST、new final
Evaluation、sealed GSE246381、E-MTAB-10902 outcome、generated-candidate outcome 或 guided output；
中央 attempt 表没有新增行，仍为 100 total / 92 completed / 3 failed / 3 incomplete /
1 stopped-throughput / 1 stopped-priority。

新增共享 `legacy_split_guard.py`；当 4 个入口请求 repository-root `data/b0_splits` 时，在读取
canonical records、manifest 或 config 前统一抛出 `SUPERSEDED_NOT_LOADABLE`。7 项 focused negative
tests 覆盖 shared guard、4 个 CLI 入口、guard-before-load ordering 和 4 个 JSONL size 不变；guard
只拒绝已声明 superseded 的 repository root，不改变其他显式 split path 的行为。旧 JSONL/Parquet
内容均未打开，5 个 payload 均未 copy/delete/move，Git history 未改写，也未创建 formal tag/Release。

disposition audit/memo、Code Availability、evidence manifest 与 internal RC 已同步：4/4 readers
guarded、0 unguarded、negative-loader evidence=true。formal-release blocker 已从“reader 未 fail-close
与 5 文件 tracked”收窄为“5 文件仍 tracked”；迁移/停止 tracking 仍需明确用户授权。evidence 仍为
65（51 local/contract、14 `/mnt`），claims=22、MBP=14/3/1、blockers=MBP-10/13/14/15。

guard/disposition/Code Availability/RC focused tests=17/17；本机/A100 精确 V3.3.2 suite 均为
92/92。GitHub core commit `794df0d` 已推送，A100 自 `5bd4424` 一次 fast-forward 到该 commit。
payload migration authorized=false、formal release/tag=false、`minimum_package_complete=false`、
`outcome_trigger_fully_satisfied=false` 与 `submission_ready=false` 均不变。

## V3.3.2 Excel inventory generated-output boundary repair（2026-08-22）

在 legacy payload migration 仍未授权时，先完成不接触 payload 的 producer-side repair。本项不训练、
不轮询 GPU/训练进度，不读取 Development TEST、new final Evaluation、sealed GSE246381、
E-MTAB-10902 outcome、generated-candidate outcome 或 guided output；中央 attempt 表没有新增行，仍为
100 total / 92 completed / 3 failed / 3 incomplete / 1 stopped-throughput / 1 stopped-priority。

只读核查确认 `scripts/data/import_excel_inventory.py` 的可达 CLI 默认会把生成型 Parquet 写到 tracked
`data_registry/excel_inventory.parquet`，且显式 `--parquet` 也不会反映到 audit 的 output 字段。默认现已
改为 `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/data_registry/excel_inventory.parquet`；小型
`docs/data/excel_inventory_audit.md` 仍留在 Git，audit renderer 改为记录实际选定的 Parquet path。
历史 audit 增加明确说明，保留旧 output 作为历史运行事实，不伪称现有文件已迁移。

本项没有运行真实 importer，没有读取真实 Excel 或现有 Parquet，没有创建 `/mnt` output，也没有对
5 个 tracked payload 执行 copy/delete/move。disposition/memo、Code Availability、manuscript、evidence
manifest 与 internal RC 已同步；剩余 ordered migration actions 从 4 项降为 3 项，但 formal-release
blocker 仍是 5 个文件 tracked，迁移/停止 tracking 仍需明确用户授权。evidence 仍为 65
（51 local/contract、14 `/mnt`），claims=22、MBP=14/3/1、blockers=MBP-10/13/14/15。

producer-boundary/legacy/Code Availability/RC focused tests=13/13；本机精确 V3.3.2 suite=95/95。
本机 importer E2E 因未安装可选 `pyarrow/fastparquet` 未执行完成，未安装或绕过依赖；A100 项目环境
一次完成精确 V3.3.2 95 项与 importer 8 项，合计 103/103。GitHub core commit `dc2ed02` 已推送，
A100 自 `794df0d` 一次 fast-forward 到该 commit。formal release/tag=false、
`minimum_package_complete=false`、`outcome_trigger_fully_satisfied=false` 与
`submission_ready=false` 均不变。

## V3.3.2 authorized legacy payload current-HEAD migration（2026-08-22）

用户明确授权将 5 个 legacy payload 先保存到 Route 2 `/mnt` root，再停止 current-HEAD tracking 并
加入窄 ignore；同时禁止 shared-history rewrite 和 formal tag/Release。本项不训练、不轮询 GPU/训练
进度，不读取 Development TEST、new final Evaluation、sealed GSE246381、E-MTAB-10902 outcome、
generated-candidate outcome 或 guided output；中央 attempt 表没有新增行，仍为 100 total /
92 completed / 3 failed / 3 incomplete / 1 stopped-throughput / 1 stopped-priority。

A100 preflight 确认五个源文件全部 tracked、大小与审计一致，目标目录不存在同名冲突。随后以无覆盖
copy 保存到 `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/legacy_repository_payloads/`；逐项验证
source/destination byte size 一致，并写入 `PROVENANCE.md`。五文件合计 34,786,075 bytes，其中四个
B0 JSONL 合计 34,739,577 bytes；没有打开 payload 内容，没有生成项目 checksum。

current HEAD 已停止 tracking 这五个精确路径，`.gitignore` 只增加五条精确规则，不忽略整个
`data_registry/` 或 `data/b0_splits/`。四个 legacy reader 继续在任何 input read 前抛出
`SUPERSEDED_NOT_LOADABLE`；Excel inventory future default 继续指向 `/mnt/.../route2/data_registry/`。
shared Git history 未改写，formal tag/Release 未创建，public payload redistribution 未授权。

disposition/memo、Code Availability、manuscript、consistency/evidence manifests 与 internal RC 已
重裁：current-HEAD tracked payload=0、migrated copies=5、payload-boundary compliant=true、legacy
payload policy resolved=true。新增 `/mnt` provenance evidence，evidence 65→66（local/contract 仍 51，
`/mnt` 14→15）。payload blocker 从 RC 移除，但 MBP=14/3/1、MBP-10/13/14/15、14-study human
rights/signoff、clean-environment、immutable archive 和 manuscript metadata/disclosure blockers 不变；
formal release/tag 与 `submission_ready` 仍为 false。

migration/guard/Code Availability/RC/evidence focused tests=42/42；本机精确 V3.3.2 suite=96/96；
A100 fresh current HEAD 完成 V3.3.2 96 项与 importer 8 项，合计 104/104。GitHub core commit
`b6fbdce` 已推送，A100 自 `1d899dd` 一次 fast-forward 到该 commit。

## 70. XEditCritic V3 + XEditSetFlow V3 prospective freeze and projection boundary（2026-08-23）

用户在模型性能讨论后冻结新的联合方法修复：Critic 三 seed 每个 task-macro Spearman 至少 `0.25`、
中位数至少 `0.30`、每 seed matched-baseline margin 至少 `0.07`、中位 margin 至少 `0.10`，并选择
SetFlow + soft-value SMC 路线。新增 machine/human protocol 固定 Critic seeds
`20260830/20260831/20260901/20260902`、SetFlow seeds `20260903/20260904/20260905/20260906`、
容量 arms、训练策略、十八点 guidance grid、320 forward-equivalent/source ceiling 和 fail-terminal 规则；
Critic V2、Base Flow V2 与 matched-generation V2 terminal 结果均不重跑。

旧 predictor loader 的严格边界问题已如实记录：历史实现先完整解析 canonical row，再按 split 丢弃
TEST；没有证据表明 TEST 进入 loss/checkpoint/metrics，但 legacy runs 不支持“protected outcome 字段从未
解析”的最强措辞。新 `DevelopmentProjectionV3` 在完整 JSON decode 前只抽取 record id 并查询冻结
manifest；只对 TRAIN/VALIDATION 行解码并写 label-bearing projection，TEST 行保持未解析。一般 TEST
projection 不存在，未来只能由 three-seed PASS 后的一次性原子 adjudicator 实现。

outcome-free endpoint registry 覆盖 Development 的八个 endpoint，并把 GSE149487 的 translation-
efficiency ratio 与 RNA-abundance ratio 映射为不同 quantity/measurement semantics。projection schema、
builder、CLI 和 loader 已实现；合成 protected TEST 行故意包含无法完整解析的 JSON payload，focused
tests 仍能完成 TRAIN/VALIDATION projection，证明 TEST 行没有发生完整 decode。

实现 commit `1659633` 推送并在 A100 fast-forward 后，真实 projection 只 materialize 一次：TRAIN
89,580 行 / 139,955,582 bytes，VALIDATION 18,293 行 / 27,545,060 bytes，TEST withheld 18,292；
canonical full-decode counts 为 TRAIN=89,580、VALIDATION=18,293、TEST=0。八个 endpoint 的投影总数
107,873，GSE149487 两个 descriptors 各 48 条且语义分离。A100 focused tests=11/11、既有精确
V3.3.2 suite=96/96；本项不训练、不新增中央 attempt，中央表仍为 100 total / 92 completed /
3 failed / 3 incomplete / 1 stopped-throughput / 1 stopped-priority。Development TEST 和新的 final
Evaluation outcome 均未读取，model/generation/biological success 与 submission-ready 均为 false。

## XEditCritic V3 edit-site token feature preflight（2026-08-23）

本项是新训练前的表示/缓存实现，不创建训练 attempt；中央计数保持 100 total / 92 completed /
3 failed / 3 incomplete / 1 stopped-throughput / 1 stopped-priority。新 builder 只接受已授权的
`DevelopmentProjectionV3` TRAIN/VALIDATION，不能 fallback 到 canonical outcome JSONL。

真实 projection 的 record/edit/unique-sequence/unique-sequence-position 几何为
107,873 / 346,862 / 43,730 / 76,159，maximum edits/record=38；cache 使用 float16 shared
sequence-position tensors 与 record ragged offsets，不截断 edit 数且不持久化 raw sequence。A100 frozen
mRNABERT real-model smoke 的 3/3 requested positions、768-d global/local tensors 全部 finite，参数数为
113,389,056。完整 cache 尚未 materialize，Critic screen 尚未启动；Development TEST、新 final
Evaluation outcome、model success、generation success 与 submission-ready 状态均未改变。

## XEditCritic V3 architecture/LoRA preflight（2026-08-23）

本项仍是训练前实现，不创建新 attempt，中央计数保持 100 / 92 / 3 / 3 / 1 / 1。C0/C1/C2
trainable parameters 精确为 486,784 / 1,798,528 / 29,489,049；固定 last-four rank-16 LoRA 增加
983,040，C3 总数为 30,472,089。该实际值低于协议的 32–36M 估计，但完全对应已冻结的真实
mRNABERT combined-QKV/gated-FFN geometry；没有结果后扩 rank 或解冻额外 block。

主模型 swap antisymmetry、identity zero、unknown-study scale=1、study-only multiplicative calibration、
四 arm finite output 和三项 candidate-information control 的 parameter equality 已通过本机/A100 focused
tests 10/10；精确 V3.3.2 为 96/96。Critic screen 仍未启动，model success 和 submission-ready 不变。

## XEditCritic V3 cache-arm runner preflight（2026-08-23）

本项实现 C0/C1/C2 与三项同几何 control 的正式 runner，但尚未参数更新，中央 attempt 计数仍为
100 / 92 / 3 / 3 / 1 / 1。真实 TRAIN sampler 为 89,580 draws 和 2,802 batches/pass，8 passes
预计 22,416 updates；batch task-homogeneous、task allocation 按 sqrt size 且 record repeat cap=4。
第 8 pass 只在同 task、不同 source group 间构造 ranking pair，final pass checkpoint 固定不再选峰值。

exact source/task complete candidate-bundle permutation 覆盖 29,271 recipients，29,259 candidates
实际改变、适用 task=6。两个无 TRAIN-support 的 Validation endpoint 只使用 TRAIN-region/global target
scale fallback。A100 C0 batch-32 forward finite；focused/V3.3.2 本机与 A100 分别为 11/11、96/96。
C3 online-LoRA runner 未完成，screen 未启动，protected outcome/model-success 状态不变。

## XEditCritic V3 C3 runner/screen gate preflight（2026-08-23）

C3 online last-four-LoRA runner 与 12-artifact strict screen gate 已实现，但没有 optimizer step 或新
attempt，中央计数保持 100 / 92 / 3 / 3 / 1 / 1。真实 A100 short-sequence forward/backward 仅验证
finite output、identity zero、983,040 LoRA parameters 的 gradient 和 0 non-LoRA encoder gradients；
不构成模型性能证据。

C3 使用 effective batch32 / physical microbatch1，screen gate 只有在 C2/C3 full 各自胜过三项同几何
control、permutation、C0 及全部绝对门槛时才允许选择；PASS 不开放 TEST。focused/V3.3.2 本机与
A100 为 19/19、96/96。cache-online 数值一致性等待完整 cache materialization 后执行；screen 未启动。

## XEditCritic V3 cache terminal 与 XEditSetFlow V3 implementation preflight（2026-08-23）

本项没有 parameter update，因此中央 training attempt 不新增，保持 100 / 92 / 3 / 3 / 1 / 1。
Critic cache 已在 A100 GPU2 terminal materialize 并完整载入验证：107,873 records、346,862 edits、
43,730 sequences、76,159 sequence-position，raw sequence/TEST/Evaluation record 均为 0；正式
cache-online 三几何数值对齐在 current HEAD 同步后执行。

SetFlow V3 的 F1/F2/F3、set-marginal loss、balanced sampler、source-token cache、common-NLL trainer、
F0 epoch-1 read-only replay、unguided recovery/G0 validator 与 strict screen gate 已实现。F2/F3 精确
冻结 Development vocab 下 trainable parameters 为 16,178,790 / 42,196,934；合法 cohort 为 TRAIN 68,294、
VALIDATION 15,924。
source cache 与 formal training 尚未启动，Base Flow V2 terminal 不重跑；本机 focused/V3.3.2 为
36/36、96/96。Development TEST/Evaluation read、model/generation success 与 submission-ready 均为 false。

cache-online v1 raw comparison 是零参数工程验证，不新增 training attempt。它按冻结 0.02/0.005
max/mean gate 终态 FAIL（0.0678864/0.00028031）并保留；阈值未改。C3 runner 随后改为 cache-anchored
LoRA delta，且 frozen encoder dropout 保持关闭；修复不改变 LoRA rank、可训练 block、head、seed、
训练预算或科学 gate。v2 同阈值 A100 revalidation 尚未执行，Critic screen 仍未启动。

repair commit `f1b6131` 的 A100 v2 同阈值 revalidation PASS，三几何 max/mean difference=0/0；真实
anchored backward smoke 仍有 983,040 LoRA gradients、0 frozen-encoder gradients，且无 optimizer step。
因此本项仍不新增 training attempt。修复 focused/V3.3.2 A100=15/15、96/96，SetFlow 实现全 focused/
V3.3.2 A100=36/36、96/96；formal screen 尚未启动，中央 100-row 计数不变。

Critic V3 C0–C3 已接入中央 ledger：RUNNING 必须先于首个 optimizer step，terminal success/failure
原子 upsert 同一 attempt id；cache arm 无参数变化、或 C3 head/LoRA 任一未变化均硬失败。focused/
V3.3.2 本机=13/13、96/96。该实现本身不新增 attempt；只有正式 screen arm 启动才将中央表从100行增加。

正式 C0/C1 screen seed20260830 已分别在 A100 GPU0/GPU5 启动，使用 commit `22317ed`；启动前两个 exact
output directories 均不存在。一次 elapsed=64s 的状态读取早于合同约定约 5 分钟，属于监控节奏错误；
当时两进程均为 `Rl` 且两卡已有 CUDA 显存占用，但日志尚无 pass event，远端缺少 `rg` 导致 ledger 查询
没有执行。没有立即补查，也没有把该快照当作正式 alive check；下一次合并检查延后到 launch elapsed≥30m。

等待期间新增的 Critic/SetFlow confirmation、atomic TEST、closed benchmark、soft-value/SMC 和 readiness
代码均未执行参数更新，因此不新增 training attempt。其 A100 sync/test 将等待 C0/C1 terminal，避免正在
运行的 completion upsert 读取到不同于实际训练的 Git HEAD。Development TEST、new final Evaluation、
guidance、model/generation success 与 submission-ready 状态均未改变。

到期后的单次合并检查显示 C0/C1 均仍存活，launch elapsed 均为 1,913 秒；SetFlow source-token cache
仍存活，elapsed=2,332 秒，尚无 terminal summary。该检查没有读取 pass-level curve 或完整日志；下一次
检查再次至少间隔 30 分钟。A100 工作树继续固定在训练启动 commit `22317ed`，后续 GitHub commit
不在 active attempt terminal 前同步到 A100。

## XEditFlow V3 value distillation、SMC runtime 与 closed evaluator preflight（2026-08-23）

新增 readiness-gated value target assembler：只接受 TRAIN state，每 state 精确 8 个 frozen SetFlow rollout，
每 rollout 精确接收 Critic seeds 20260831/20260901/20260902 的三个 study-neutral calibrated prediction；
independent evaluator、Development TEST 与 new Evaluation 均不得进入 target。由于原协议未指定 value
optimizer 的剩余细节，已在任何 outcome 结果前前瞻冻结为 8 个固定 passes、BF16/AdamW、batch32、
Huber-delta1 与 final-pass checkpoint，不做 epoch 结果重选。

新增实际 CUDA/BF16 scalar-value trainer、32-particle batched SMC provider/runner 和 exact closed evaluator。
SMC 以 base transition 为 proposal、`exp{beta × [V(child)-V(state)]}` 为逐步 importance weight，因此加权
粒子对应冻结的 potential-guided rate，且没有 free action-ratio head。ESS<16 时 stratified resampling；
首轮不足 32 个唯一候选时使用新 decoder seed stream 追加 32-particle round，直到 cap 或 compute 余量不足。
每 source 320 ceiling 中固定预留三次终态 critic member forward；三名 critic 必须分别计费。primary 与
replay 使用相同 seed stream，replay 计算不伪装成独立 training repeat。

closed evaluator 只允许 Development Validation measured-neighborhood，对每个最多 5-edit candidate 的所有
排列精确求和，并在同一 source 内缓存已评分 state；source-level undefined NDCG 不填 0。上述入口均由
`CRITIC_READY_FOR_GUIDANCE` + SetFlow three-seed confirmation 双 gate 阻塞，当前没有运行 value target、
value training、SMC、closed outcome benchmark 或任何 guidance grid combination。

合并本机 post-screen focused cohort=89/89，精确 V3.3.2=96/96，新增 Python compile 与 `git diff --check`
通过。A100 focused/V3.3.2 仍等待 active C0/C1 terminal 后统一同步；中央 attempt 只包含已启动的 C0/C1，
本项实现没有新增 optimizer attempt。Development TEST/new final Evaluation read、critic/generator success、
guidance authorization、model claim 与 submission-ready 均保持 false。

guidance screen config preparer 进一步固定为 6 个 `κ×τ` value jobs 与精确 18 个 `κ×τ×βmax` generation/
closed jobs，且只允许 base-flow seed20260904；adjudicator 只接受完整 18-grid 并按冻结的五级顺序选一次。
focused=4/4、V3.3.2=96/96；readiness 未满足时 config preparation 本身硬失败，当前未 materialize job manifest。

Critic post-TEST runtime 已补齐但仍未获运行授权：frozen TEST PASS 后只为 selected C2/C3 生成三个
all-Development refit，pass count 固定为三 confirmation selected-pass 的整数中位数；三个 refit 完成后
才生成 3 seeds × 7 held-out studies × selected/C0 的 42 个 paired LOSO jobs。REFIT 使用全部 107,873
TRAIN+VALIDATION records 且不做 epoch 重选；LOSO held-out study 统一映射到 unknown study scale=1，
不允许 study intercept。七 study inventory 含 dense-study stress test GSE269595，缺 fold/seed/arm 均不能 adjudicate。

cache C0/C2 与 online-LoRA C3 trainer 均已支持 REFIT/LOSO partition；post-TEST authorization 只读取 atomic
frozen TEST 的 PASS gate，不重新读取 TEST outcome。focused=26/26、V3.3.2=96/96、diff-check PASS。
当前 atomic frozen TEST 尚未授权，因此未生成 refit/LOSO configs、未启动 3+42 个 jobs，中央 attempt 不变。

最终 Critic readiness composer 只在 three-seed、atomic frozen TEST、三 refit、LOSO 四个 predecessor 全部
PASS 时输出 `CRITIC_READY_FOR_GUIDANCE`；LOSO NO-GO 等任一失败保持 blocked。focused=12/12、
V3.3.2=96/96。该 composer 不读取 record/outcome，且始终保持 new final Evaluation closed。

最终 XEditFlow adjudicator 现要求精确三个 base-flow seeds、每 seed 六个冻结 method roles、source-paired
NDCG/evaluator bootstrap、matched compute 与 protected-read evidence，随后调用严格 three-seed gate。
只有完整 PASS 才授权一次 replacement Evaluation；即使 PASS，`submission_ready` 仍保持 false，必须等待
external measured outcome 复现。focused=4/4、V3.3.2=96/96；当前没有 final comparison artifacts，未运行。

## XEditFlow V3 closed-score / strongest baseline adapter preflight（2026-08-23）

本项只补充 benchmark adapter，不构成新的 optimizer attempt，因此中央训练表不新增行。common closed
score table 现在必须精确覆盖 Development Validation measured candidates，并使用 sourcewise shifted-exp
正权重保持冻结分数的 source 内排序；undefined source 不填零。历史 frozen strongest baseline 继续绑定
`genetic` 与 320 forward-equivalents/source，仅映射已经 terminal 的 open-generation/G0 事实，不复用旧
open NDCG 作为 closed NDCG，也不重新选择 baseline。新的 closed result 只有在 readiness 后由 frozen
genetic guiding checkpoint 对 common measured candidates 评分才能产生，当前未执行。

定向测试=12/12、完整 XEditFlow V3 focused=71/71、本机精确 V3.3.2=96/96、compile/diff-check PASS；
A100 tests/sync 等 active screen jobs terminal 后执行。Development TEST/Evaluation outcomes 未读取，
guidance 与 replacement Evaluation 均未授权，submission-ready=false。

## XEditFlow V3 common closed producer preflight（2026-08-23）

本项不训练，中央 optimizer attempt 表不新增行。三个 frozen base-flow seed 现在各自具备六方法 closed
job inventory：full/unguided 使用各自 transition distribution 的 exact all-permutation terminal probability；
first-order 使用 source-anchored additive potential；simple-rate/rerank 使用 frozen 三 Critic ensemble terminal
reward；strongest baseline 使用只读 genetic guiding checkpoint score。所有方法严格共享 Development
Validation measured candidate cohort，undefined source 不填零，旧 open NDCG 不复用。first-order/simple-rate
不做每个 path state × 全合法 child 的额外 Critic normalization，避免 closed evaluation 在 320/source 的生成
预算之外引入爆炸计算；它们的 open generation 仍保持各自真实采样分布，并未被替换成 rerank。

完整 XEditFlow V3 focused=83/83、本机精确 V3.3.2=96/96、compile/diff-check PASS。双 readiness gate 未
通过，故本项只完成实现与 preflight，没有执行 closed outcome benchmark、没有访问 Development TEST 或
new Evaluation；guidance、replacement Evaluation 和 submission-ready 仍为 false。

## XEditFlow V3 final execution-chain preflight（2026-08-23）

本项不新增 optimizer attempt。三个 frozen base-flow seed 的 full/control/closed/open/independent-evaluator/
bootstrap/final-evidence job chain 已完整 materialize；每 seed 精确六方法，最终只允许 3×6 manifest。修复了
两个正式执行前的可达错误：strongest selection pool 现使用真实冻结值
`DEVELOPMENT_MEASURED_NEIGHBORHOOD`；full SMC headline compute 现把三名 terminal Critic member
forwards 加入 generation subtotal，若总数超过 320/source 即失败。independent evaluator 也必须与三个
Critic refit checkpoint 全部分离，并与 frozen genetic strongest artifact 中的 evaluator 路径一致。

XEditFlow V3 focused=83/83、independent-evaluator focused=4/4、本机精确 V3.3.2=96/96、compile/
diff-check PASS。当前所有 final jobs 仍被 readiness 双 gate 阻塞，未执行 final comparison，未访问
Development TEST/new Evaluation；replacement Evaluation authorization=false，submission-ready=false。

## 05:01 scheduled screen status（2026-08-23）

中央 active attempts 未终态：C1/F1/F2 分别已运行 2:44:38、1:14:29、0:59:32，进程均存活且没有
terminal/error marker，所以不更改 ledger status。GPU0–5 free memory 为 2,569/7,983/3,175/5,471/
4,267/6,743 MiB；结合当前正式 arms 的实际设备占用，没有足够余量启动 F3/C2/C3。没有降容量、没有
CPU fallback、没有新增或重复 attempt。下一次检查至少 30 分钟后；Development TEST/Evaluation 状态不变。

## XEditCritic V3 C0 terminal 与 SetFlow source-cache terminal（2026-08-23）

第二次合同节奏检查时，C0 已 terminal，C1 仍运行。C0 使用 seed20260830、8 passes、22,416 updates，
parameter_changed=true；Development Validation task-macro Spearman=`0.1108180590`、positive tasks=8/9、
task-macro standardized MAE=`1.9924297611`、prediction std=`52.32135326`。C0 是 matched same-information
baseline，不单独触发 selectable gate；该结果原样保留，不重训。Development TEST/new Evaluation read=0，
central ledger 行为 `COMPLETED`。C1 在 elapsed=4,415 秒仍为 `RUNNING`，无 terminal artifact，下一次至少
30 分钟后检查。

SetFlow source-token cache 已 terminal 并完成一次全量 loader validation：artifact=4,332,870,924 bytes，
eligible records=84,218、unique sources=19,303、source tokens=2,817,781、tensor shape=`[2817781,768]`、
float16；raw sequence payload=false、TEST/Evaluation records=0。构建任务不更新参数，不新增 training attempt。

GPU0–5 当时 free memory 仅约 2.6/9.9/3.2/7.9/4.3/7.0GB，主要由 tokenizer benchmark、
ToeholdDesignBench 与其他用户任务占用；没有本项目可安全终止的遗留进程。C2/C3 因显存不足未启动，
没有写虚假 RUNNING ledger，也没有改变 batch/容量或 CPU fallback。中央表现为原 100 rows + C0/C1 两行：
总计102，completed93，另有 C1 running1；旧 3 failed/3 incomplete/1 stopped-throughput/1 stopped-priority
保持。A100 HEAD 仍为 `22317ed`，未同步后续代码。

## XEdit V3 strict gate artifact-identity preflight（2026-08-23）

本项是 outcome-free gate 审计与实现修复，不是新的 optimizer attempt，因此中央 CSV 不新增行，C0/C1/F1/F2
既有状态也不因本项改变。Critic screen/confirmation 现要求 exact run、arm、control、完整 candidate-bundle
permutation、parameter matching、split/pass/update 和 CUDA/BF16 provenance；SetFlow screen/confirmation 现
要求 exact F0 frozen reference 与 F1/F2/F3 arm/role/split/budget/unguided provenance。最终 XEditFlow closed
assembly 现要求六方法共享完全相同的 measured/defined source support，macro NDCG 与 defined-source mean
一致，independent-evaluator headline margin 与 paired-source mean 一致；undefined source 继续排除而不填零。

所有冻结数值 threshold、seed、baseline 和 selection order 不变。新增与相邻合并 XEdit V3 focused tests
=183/183，精确 V3.3.2=96/96，compile/diff-check PASS；A100 tests/current-HEAD sync 继续等待 launch head
`22317ed` 的 active C1/F1/F2 terminal。本项没有重复 C0、Critic V2、Base Flow V2 或其他 terminal 实验，
Development TEST/new Evaluation read=0，guidance/replacement Evaluation/submission-ready 仍为 false。

## XEditFlow V3 closed frozen search baseline coverage（2026-08-23）

本项不新增 optimizer attempt。final benchmark config 现为 `random_legal/greedy/beam/genetic/local_search`
五个历史 frozen search baseline 补齐 common closed metric jobs；五者的原 runner 使用同一 matched guiding
checkpoint 排序 terminal candidates，因此只复用一次 strongest score table，避免重复五次相同 scorer
forward。每个 metric config 显式记录其报告 method 与 score-table method；closed evaluator 对该映射硬校验。

历史 open-support 结果保持原样，未被重命名为 closed；五方法在 common fixed candidate set 上预期同排名，
open support 的算法差异继续由 recovery/diversity/cost 表达。合并 focused=184/184、V3.3.2=96/96、
compile/diff-check PASS；readiness 前未执行 metric，Development TEST/new Evaluation read=0。A100 sync/test
仍等待 launch-head `22317ed` active screen jobs terminal。

## 05:34 scheduled screen status（2026-08-23）

C1/F1/F2 仍为既有三个 RUNNING attempts，elapsed=3:17:35/1:47:26/1:32:29；没有 terminal summary 或
error marker，因此中央 CSV 不改写状态、不新增行。GPU0–5 free memory=2,591/7,043/3,175/5,193/
4,267/7,701 MiB，仍不足以安全启动 C2/C3/F3；未降低容量或使用 CPU fallback。A100 HEAD 继续为
`22317ed`，下一次检查不早于 06:04:56。protected outcome 与 downstream authorization 状态不变。

## XEdit V3 prospective Methods addendum（2026-08-23）

本项只新增 outcome-free 论文 Methods addendum 与 claim-boundary tests，不构成 optimizer attempt，中央 CSV
不新增行。文档覆盖 Critic/SetFlow/soft-value/SMC/closed-open benchmark 和真实参数量，所有性能、泛化与
投稿结论均标记为仍需 terminal evidence；当前 V2 Results 未被改写。focused=3/3、V3.3.2=96/96、
diff-check PASS，Development TEST/new Evaluation read=0；A100 sync/test 延后到现有 screen jobs terminal。

## XEdit V3 prospective Experiments protocol（2026-08-23）

本项是等待 formal jobs 时的论文/benchmark 报告冻结，不构成 optimizer attempt，中央 CSV 不新增行。
新增协议把 Critic、unguided SetFlow、guided SMC、matched compute 和 future external confirmation 映射到
九张预定表、四幅图及明确统计单位；closed/open/evaluator/self-score/compute 不合并，required negative
task/fold rows 不选择性删除。focused=4/4、V3.3.2=96/96、JSON/diff-check PASS，protected outcome read=0；
A100 HEAD 仍保持 `22317ed`，current-HEAD sync/test 等 active launch-head jobs terminal。

## XEditCritic V3 gate parameter-identity repair（2026-08-23）

本项修复 screen/confirmation terminal artifact identity gate，不构成 optimizer attempt，中央 CSV 不新增行。
gate 现核对四 arm 精确 trainable capacity、C0–C2 parameter change、C3 head+LoRA change 以及 C3 固定
effective/physical microbatch；避免 wrong-capacity 或无学习更新的 summary 进入 adjudication。训练参数、数值
threshold、seed、baseline 与 task 均不改变。focused=21/21、V3.3.2=96/96、compile/diff-check PASS，
Development TEST/new Evaluation read=0；A100 sync/test 继续等待 launch-head jobs terminal。

## XEditSetFlow V3 gate artifact-identity repair（2026-08-23）

本项只强化 SetFlow screen/confirmation 对既有 terminal fields 的验证，不新增 optimizer attempt 或中央 CSV
行。gate 现要求 exact F2/F3 capacity、891-source/28,512-trajectory unguided cohort、method/batch identity、
small-graph exactness、零 validation update/critic/evaluator/protected read，以及无 canonical/biological overclaim。
数值 gate、seed、训练和 F0/V2 terminal 历史不变。focused=28/28、V3.3.2=96/96、compile/diff-check PASS；
A100 current-HEAD sync/test 等 `22317ed` active jobs terminal。

## 06:07 scheduled screen status（2026-08-23）

C1/F1/F2 仍为既有三个 RUNNING attempts，elapsed=3:50:17/2:20:08/2:05:10；均存活且没有 terminal、
failure 或 error marker，因此中央 CSV 不改状态、不新增行。GPU0–5 free memory=2,569/9,095/3,153/5,209/
4,267/5,919 MiB，仍不足以启动 C2/C3/F3；GPU6/7 未获授权，未使用。下一次检查不早于 06:37:38，
A100 HEAD=`22317ed`，protected outcome 和 downstream authorization 均不改变。

## XEditFlow V3 equal-wall-time / resource evidence repair（2026-08-23）

本项不构成 optimizer attempt，中央训练 CSV 不新增行，也不更改 C1/F1/F2 的 RUNNING 状态。final
matched-compute chain 原先只保证 320 forward-equivalent ceiling；full/control runner 的 aggregate wall time
口径不同，历史 strongest genetic artifact 又没有 per-source generation time，因此无法真实完成冻结协议要求的
same-cohort wall time、peak VRAM 与 equal-wall-time sensitivity。该缺口现于任何 final comparison 执行前修复。

full SMC、四个 matched controls 与 frozen search runner 现逐 source 执行 CUDA synchronize，并记录统一的
`A100_END_TO_END_GENERATION_INCLUDING_REPLAY_AND_REQUIRED_SELECTION_SCORING` 时间、该 scope 的 peak VRAM
和设备名称。first-order/simple-rate/unguided 的 post-hoc terminal critic diagnostic 不计入 selection wall time；
generate-then-rerank 的 terminal critic 决定排序，因此计入。旧 `peak_vram_mb` search 字段保留，避免破坏已有
generation evaluator 接口。

新增 equal-wall builder 要求精确六方法、冻结 891-source 顺序、A100、每 source 正计时、完全一致的 closed
support 和 undefined-not-zero-fill；以六方法 full-cohort 最短总时间作为预算，只比较各方法都完整完成的共同
source prefix，不把 partial source 计为完成。该 sensitivity 是报告性分析，不新增 performance threshold，
但缺失或伪造资源证据会阻止 final seed assembly/adjudication。

历史 strongest genetic 不追溯填充时间，也不改写 terminal 结果；final manifest 只新增一次 timing-only V3
benchmark job，固定原 guiding checkpoint、256 critic forwards/source、population32、seed20260816 和其余
既有 search hyperparameters，明确 `timing_only_no_baseline_reselection=true`。该 job 尚未运行，并继续受
readiness 与 GPU0–5 约束。

正确 Python 3.13 环境下扩展 XEdit focused cohort=212/212、精确 V3.3.2 cohort=96/96；本批定向
equal-wall/final-chain tests=50/50，Python compile 与 diff-check PASS。默认 macOS Python 3.9 对既有
`zip(..., strict=True)` 的两项失败被确认是解释器不满足项目 Python≥3.10 要求，不据此改写正式实现。
Development TEST/new Evaluation read=0，guidance/replacement Evaluation/submission-ready 状态不变；A100
current-HEAD sync/test 仍等待 launch-head `22317ed` 的 active jobs terminal。

## 06:40 scheduled screen status（2026-08-23）

C1/F1/F2 仍为同三个 RUNNING attempts，elapsed=4:22:47/2:52:38/2:37:40；进程均存活，terminal
summary、failure artifact 与 error marker 全部不存在，因此中央 CSV 不改状态、不新增行，也没有读取活跃
curve 或 performance metric。GPU0–5 free memory=2,569/6,581/3,177/4,489/4,307/7,525 MiB，
utilization 均为 100%，仍不足以启动 C2/C3/F3；GPU6/7 未获授权且未使用。

C1 已超过 4 小时，后续按合同改为 60 分钟间隔，下一次 C1 检查不早于 07:40:07；F1/F2 仍预计少于
4 小时，下一次只检查 F1/F2，不早于 07:10:07。未降容量、未 CPU fallback，A100 HEAD 继续固定
`22317ed`；Development TEST/new Evaluation 与 downstream authorization 状态不变。

## XEditCritic V3 C3 ranking singleton-microbatch preflight（2026-08-23）

本项为 C3 尚未启动前的运行时一致性修复，不新增中央 optimizer attempt，也不改变 C1/F1/F2 的 RUNNING
状态。此前 C3 Huber 路径使用 physical microbatch1，但 final-pass ranking pair 会以 batch2 进入在线
mRNABERT，与声明 provenance 不符且形成 late-pass OOM 风险；现左右 pair member 分别 batch1 前向，再按
原目标计算同一 logistic difference。pair、loss weight、effective batch32、updates、seed、容量与 gate 均不变。

定向/相邻=20/20、Critic V3 focused=67/67、本地精确 V3.3.2=96/96、compile/diff-check PASS；A100 tests/
current-HEAD sync 在 `22317ed` active jobs terminal 后执行。Development TEST/new Evaluation read=0，C3
formal attempt 仍为未启动。

## XEditFlow V3 physical-forward accounting preflight（2026-08-23）

本项是 formal guidance 前的 matched-compute 计费修复，不构成 optimizer attempt，中央 CSV 不新增行，也
不修改 C1/F1/F2 的 RUNNING 状态。原实现把 critic scorer 的内部 microbatch 忽略为每成员一次调用，并漏计
deterministic replay 实际执行的 forward；32 candidates、microbatch4、三成员的 terminal reservation 因此从
错误的 3 修正为 `3 × ceil(32/4) = 24`。

所有 base/value/critic 现在按实际物理 batch 计费，primary/replay 合并后再检查 320 ceiling，cache hit 保持
零新增 forward，guidance adjudicator 不再重复加 reservation。SMC remaining-compute 字段同步使用 24。该
修复不改变模型、候选、seed、grid、gate 或 terminal evidence；compile PASS、相关定向=45/45、XEdit
focused=209/209、本机精确 V3.3.2=96/96、JSON/diff-check PASS。A100 current-HEAD 验证等待 launch-head
`22317ed` active jobs terminal。Development TEST/new Evaluation read=0。

## 07:12 F-only scheduled screen status（2026-08-23）

F1/F2 仍为同两个 RUNNING attempts，elapsed=3:25:07/3:10:09；均存活，没有 terminal/failure/error marker，
未读取 curve 或 performance metric，因此中央 CSV 不改状态、不新增行。本次未检查 C1。GPU0–5 free
memory=2,569/8,853/3,175/4,595/4,285/8,223 MiB，仍不足以启动 F3/C2/C3。下一次 F-only 检查不早于
07:42:37，C1 仍不早于 07:40:07；A100 HEAD=`22317ed`，protected outcome 与 downstream authorization
不变。

## XEditSetFlow V3 F1 screen training terminal（2026-08-23）

F1 seed20260903 已 terminal `COMPLETED`，中央 CSV 第104行已更新；3 passes early-stop、selected pass1、
12,807 updates、best common Validation set-NLL=5.47242674446921。相对冻结 F0 5.397907635224613 的
relative improvement=-0.013805184208472678，F1 的 10% NLL diagnostic check明确失败。F1不可 final，
不重训、不改变 F2/F3 gate。

唯一一次 F1 unguided validation 已在 GPU1 启动，Python PID3153416；首次5分钟检查仍运行且无 terminal/
failure/error，下一次不早于08:20:27。F2/C1仍RUNNING，F3/C2/C3未启动。Development TEST/new Evaluation
read=0，A100 HEAD保持`22317ed`。

## 08:15/08:20 scheduled screen status（2026-08-23）

F2仍为RUNNING，elapsed=4:12:57，无terminal/failure/error；已切换60分钟节奏，下次不早于09:15:25。
F1 unguided validation亦RUNNING，elapsed=36:49，无terminal/failure/error，下次不早于08:50:44。两次观察
均未读取active metric；C1未触碰，仍不早于08:42:39。GPU0–5无足够安全显存，F3/C2/C3未启动，中央
CSV和protected outcome状态不变，A100 HEAD=`22317ed`。

## XEditSetFlow V3 F1 full diagnostic terminal（2026-08-23）

F1 validation terminal `FLOW_G0_READY`：recovery=0.26917321361765806、top-k=0.21314548765446636，达到两项
阈值；unique=0.4192269921436588，未达到0.90；legality=1且全部failure counter=0。training NLL相对F0
恶化1.3805%，所以F1 objective-only diagnostic完整失败、不可final且不重训。它支持“set-marginal改善
measured recovery但旧小trunk严重mode concentration”的方法诊断，不支持模型优势claim。

C1在08:45:08仍RUNNING、无terminal/failure/error，下一次不早于09:45:08；F2下一次不早于09:15:25。
protected outcome read=0，A100 HEAD保持`22317ed`。

## XEditSetFlow V3 F2 training terminal / F3 launch / capacity correction（2026-08-23）

F2 seed20260903 training terminal `COMPLETED`：4 passes early-stop、selected pass2、17,076 updates、best common
set-NLL=2.0680908163671576，相对F0改善61.687%，训练侧NLL check通过；validation仍待terminal，不能提前判
screen PASS。F2 terminal不重训。

formal frozen vocab的实际capacity为F2=16,178,790、F3=42,196,934；旧gate/preflight两者各高估224，已在
screen adjudication前纠正。模型/数据/seed/threshold均未变。F3 training已在GPU3启动，F2 unguided
validation已在GPU1启动；首次检查遵守5分钟。定向10/10、SetFlow focused28/28、V3.3.2 96/96，A100 HEAD
仍为`22317ed`，protected outcome read=0。

## F3 training / F2 validation initial health（2026-08-23）

F3中央CSV第106行为RUNNING，实际Python PID3408897、elapsed537秒、GPU3 free3,861MiB；F2 validation
PID3408905、elapsed538秒、GPU1 free8,213MiB。两者无terminal/failure/error，下次不早于09:58:36。
capacity/batch/seed未变，active metric未读，C1仍不早于09:45:08，A100 HEAD=`22317ed`。

## XEditCritic V3 C1 terminal / C2 full launch（2026-08-23）

C1 terminal `COMPLETED`：Spearman=0.1386460633119141、相对C0 margin=0.027828004309577672、MAE=
1.9004665150593998、8/9 tasks正。C1弱改善但未达screen阈值，不可final、不重训。Critic terminal summary
文件约定已纠正为`run_summary.json`，此前错误existence check未改变结果。

C2 full已在GPU5启动，PID3481436、seed20260830、control NONE；中央screen inventory仍缺C2 controls与
C3 full/controls，不允许adjudicate。TEST/Evaluation read=0，A100 HEAD保持`22317ed`。

## C2/F3/F2-validation scheduled health / C2 source-only launch（2026-08-23）

C2 full在09:55:15启动后首次检查健康，中央CSV第107行为RUNNING；下次不早于10:25:15。F3 training与
F2 validation在10:01:53仍健康、无terminal/failure artifact，下次均不早于10:31:53；没有读取active
metric。

GPU0释放33,368MiB后启动C2 `SOURCE_ONLY` control（PID3577060，seed20260830），启动前确认同名中央行、
terminal与failure artifact均不存在；runner将创建唯一正式attempt，首次health检查等待至少5分钟。C3受
current-HEAD singleton-ranking修复约束仍未启动。protected outcome read=0，A100 HEAD=`22317ed`。

10:09:27首次检查确认实际Python PID3577062（3577060为launcher）、中央CSV第108行为RUNNING，
device=`cuda:0`、precision=BF16；无terminal/failure/error，下一次不早于10:39:27。

## XEdit V3 interim Development evidence note（2026-08-23）

新增`docs/paper/route2_xedit_v3_interim_results_evidence_v1.md`，将C0/C1与F0/F1 terminal diagnostics及F2
training-only事实从前瞻protocol中分离记录。没有新参数更新或metric read，本项不新增中央attempt；screen、
guidance、TEST/Evaluation授权状态均不变。

## XEditSetFlow V3 F2 terminal arm gate / F3 health（2026-08-23）

F2中央CSV第105行保持`COMPLETED`，训练NLL相对F0改善61.687%；唯一冻结validation已terminal：recovery=
0.2924616535727647、top-k=0.168278220268518、unique=0.6793630751964085，G0 correctness全部通过。
F2因unique<0.90单臂gate失败，terminal、不重训、不进入confirmation。

F3中央CSV第106行仍RUNNING，PID3408897、elapsed4,385秒、无terminal/failure，下次不早于11:02:43。
F3 validation未开始，总screen不adjudicate；protected outcome read=0，A100 HEAD=`22317ed`。

C2 full在10:29:35仍RUNNING（中央CSV第107行、PID3481436、elapsed2,454秒），无terminal/failure，下一次
不早于10:59:35；未因GPU5瞬时低utilization叠加任务。

## F2 terminal diversity-by-budget diagnostic（2026-08-23）

已terminal F2的B1/B3/B5 mean unique分别为0.4832461977/0.7134046053/0.8066165123，达到0.90的source
比例为0/0.2203947368/0.3580246914；失败贯穿三个预算而非仅B1。本项只读终态artifact、不训练、不新增
中央attempt；F2/F3与protected outcome状态不变。

## 10:59–11:09 scheduled screen health（2026-08-23）

C2 full/F3 training/C2 source-only分别在10:59:48/11:03:35/11:09:59保持RUNNING，elapsed分别为
4,267/6,236/4,033秒，均无terminal/failure；下一窗口分别为11:29:48/11:33:35/11:39:59。未读active
metric、未新增attempt或叠加GPU任务，protected outcome read=0，A100 HEAD=`22317ed`。

## 11:30–11:40 scheduled screen health（2026-08-23）

C2 full/F3 training/C2 source-only分别在11:30:38/11:34:12/11:40:26保持RUNNING，elapsed分别为
6,117/8,073/5,860秒，均无terminal/failure；下一窗口分别为12:00:38/12:04:12/12:10:26。自动续轮未
触发早查，未读active metric、未新增attempt或叠加GPU任务；protected outcome read=0，A100 HEAD=`22317ed`。

## 12:00–12:11 scheduled screen health（2026-08-23）

C2 full/F3 training/C2 source-only分别在12:00:54/12:04:56/12:11:01保持RUNNING，elapsed分别为
7,933/9,917/7,695秒，均无terminal/failure；下一窗口分别为12:30:54/12:34:56/12:41:01。未读active
metric、未新增attempt或叠加GPU任务；protected outcome read=0，A100 HEAD=`22317ed`。

## 13:01–13:13 scheduled screen health（2026-08-23）

C2 full/F3 training/C2 source-only分别在13:01:35/13:05:49/13:13:04保持RUNNING，elapsed分别为
11,574/13,570/11,418秒，均无terminal/failure；下一窗口分别为13:31:35/13:35:49/13:43:04。
F3下次若仍运行将切换60分钟节奏；未读active metric、未新增attempt或叠加GPU任务，protected outcome
read=0，A100 HEAD=`22317ed`。

## SetFlow screen launch-HEAD stage identity repair（2026-08-23）

current-HEAD screen gate现可对`22317ed`生成的精确terminal v3 screen summary在缺少后加`run_stage`字段时，
由schema+seed20260903+terminal identity唯一解释为SCREEN；其他缺失/错误stage仍硬失败，confirmation仍要求
显式CONFIRMATION。terminal artifacts与所有metric/gate/selection不改。本项不新增中央optimizer attempt；
focused=30/30、精确V3.3.2=96/96，A100 current-HEAD测试等待旧HEAD jobs terminal，protected read=0。

## F3 validation terminal / XEDITSETFLOW_V3_SCREEN_NO_GO（2026-08-23）

F3唯一unguided validation已terminal：recovery=0.19397680508791618、top-k=0.10487128067094396、
unique=0.6374508978675645；训练NLL改善62.014%，G0 correctness全部通过，但三项生成门槛均失败。
F2/F3两个selectable arms均terminal fail，SetFlow V3冻结为`XEDITSETFLOW_V3_SCREEN_NO_GO`；selected arm为空，
confirmation/retraining/额外seed/降阈值均不授权。validation不新增optimizer attempt，protected read=0；正式
current-HEAD gate artifact与A100 tests等待旧HEAD C2 jobs terminal后materialize。

## C2 EDIT_METADATA_ONLY control launch（2026-08-23）

F3释放GPU3后启动预注册C2 `EDIT_METADATA_ONLY`，seed20260830、旧launch HEAD=`22317ed`；中央CSV第109行
为RUNNING，PID889042。5分钟初检确认CUDA/GPU3、无terminal/failure；stderr仅PyTorch性能warning，下一次
不早于18:32:31。该项是screen required control，不重复terminal实验，不改变SetFlow NO-GO或protected read。

## Critic screen launch-HEAD training identity repair（2026-08-23）

current gate现仅对`22317ed`写出的精确terminal `screen_run.v1`在同时缺少后加selected-pass/scope字段时，
由seed20260830、8 passes、22,416 updates与fixed-final-pass policy派生pass8/frozen scope；其他缺失或错误
identity仍硬失败，terminal artifact/metric/threshold不变。本项不新增中央attempt；Critic focused=70/70、
精确V3.3.2=96/96，A100 current-HEAD测试pending，protected read=0。

## C2 full/source-only terminal / remaining controls running（2026-08-23）

C2 full中央CSV第107行COMPLETED：Spearman0.1042656112、C0 margin-0.0065524478、MAE1.9705208102、
7/9 tasks正，primary screen criteria已失败。source-only第108行COMPLETED：Spearman-0.0013102930，full胜该
control但仍不eligible。第109行edit-metadata保持RUNNING；第110/111行新增预注册no-candidate/permutation，
5分钟初检均健康。controls继续到terminal，不因C2 full失败而删减；protected read=0，C3仍待current-HEAD sync。

## 18:55–19:03 C2 remaining-control health（2026-08-23）

`NO_CANDIDATE_SEQUENCE`、candidate-bundle permutation、`EDIT_METADATA_ONLY`分别在18:55:33、18:59:57、
19:03:31保持RUNNING，elapsed为2,492/2,159/3,972秒，均无terminal/failure；后两项中央CSV第111/109行
仍为RUNNING。下一窗口分别为19:25:33/19:29:57/19:33:31。没有新optimizer attempt或主动停止；无代码
变化，focused/V3.3.2 cohort不重复，最近为70/70与96/96；A100 current-HEAD tests pending，protected read=0。

## F2/F3 terminal budget/domain recovery diagnostic（2026-08-23）

只读终态outcome-free aggregation显示，F3相对F2在B1/B3/B5的recovery差值为-0.0393/-0.2094/-0.0424，
top-k差值为-0.0244/-0.1399/-0.0234，unique差值为-0.0245/-0.0574/-0.0416；更大F3容量没有在任何
budget修复生成指标。F3四个domain unique均低于0.90，三项minority-domain recovery为0–0.0405；总体受
652/891-source GSE114002组成影响。该项不新增optimizer attempt、不改gate、不读protected outcome、不重复test。

## 19:25–19:33 C2 remaining-control health（2026-08-23）

C2 no-candidate/permutation/edit-metadata分别在19:25:57/19:30:13/19:33:56保持RUNNING，elapsed为
4,316/3,975/5,797秒，中央CSV第110/111/109行均仍RUNNING且无terminal/failure。下一窗口为
19:55:57/20:00:13/20:03:56。未新增attempt或停止control；无代码变化，不重复测试，protected read=0。

## 19:56–20:04 C2 remaining-control health（2026-08-23）

no-candidate/permutation/edit-metadata分别在19:56:19/20:00:32/20:04:49保持RUNNING，elapsed为
6,138/5,794/7,651秒，中央CSV第110/111/109行均RUNNING，无terminal/failure。下一窗口为
20:26:19/20:30:32/20:34:49；未新增attempt、未停control、不重复测试，protected read=0。

## 20:26–20:36 C2 remaining-control health（2026-08-23）

no-candidate/permutation/edit-metadata分别在20:26:38/20:31:03/20:36:21保持RUNNING，elapsed为
7,957/7,625/9,542秒，中央CSV第110/111/109行均RUNNING，无terminal/failure。下一窗口为
20:56:38/21:01:03/21:06:21；低util不作干预门槛，未新增attempt或重复测试，protected read=0。

## 20:56–21:07 C2 remaining-control health（2026-08-23）

no-candidate/permutation/edit-metadata分别在20:56:59/21:02:13/21:07:34保持RUNNING，elapsed为
9,778/9,495/11,415秒，中央CSV第110/111/109行均RUNNING，无terminal/failure。下一窗口为
21:26:59/21:32:13/21:37:34；未新增attempt、未停control、不重复测试，protected read=0。

## 21:27–21:38 C2 remaining-control health（2026-08-23）

no-candidate/permutation/edit-metadata分别在21:27:29/21:33:00/21:38:51保持RUNNING，elapsed为
11,609/11,342/13,293秒，中央CSV第110/111/109行均RUNNING，无terminal/failure。下一窗口为
21:57:29/22:03:00/22:08:51；下一观测超过四小时后各自切换60分钟。未新增attempt，protected read=0。

## 21:58–22:09 C2 remaining-control health（2026-08-23）

no-candidate/permutation/edit-metadata分别在21:58:08/22:02:39/22:09:02保持RUNNING，elapsed为
13,448/13,121/15,103秒，中央CSV第110/111/109行均RUNNING且无terminal/failure。下一窗口为
22:28:08/22:32:39/23:09:02；edit-metadata已超过四小时并切换60分钟。permutation因时钟偏移估计不足
早于预定边界21秒，本轮未立即重查且后续改按实际观测时间起算。未新增attempt，未读active metric，
protected read=0。

## 22:28–22:32 C2 controls enter long-interval monitoring（2026-08-23）

no-candidate/permutation分别在22:28:23/22:32:40保持RUNNING，elapsed为15,263/14,923秒，中央CSV
第110/111行仍RUNNING且无terminal/failure；二者均超过四小时，后续切换60分钟，下一窗口为
23:28:23/23:32:40。edit-metadata本轮未提前重查，下一窗口仍为23:09:02。三个control现均进入
60分钟节奏；未新增attempt、未读active metric，protected read=0。

## Route 2 heartbeat long-run interval alignment（2026-08-23）

既有`route2` heartbeat从130分钟原位调整为60分钟，以匹配当前三个超过四小时control的合同监控节奏；
保持ACTIVE、failed-only与当前thread target，未创建重复automation。此运营调整不新增optimizer attempt、
不查询远端训练、不改变A100 HEAD，protected read=0。

## 23:09 C2 edit-metadata long-interval health（2026-08-23）

edit-metadata在23:09:17保持RUNNING，elapsed18,718秒，中央CSV第109行仍RUNNING且无terminal/failure；
下一窗口为2026-08-24 00:09:17。no-candidate/permutation本轮未提前重查，仍为23:28:23/23:32:40。
未新增attempt、未读active metric，protected read=0。

## 00:26–00:34 C2 remaining controls second long-interval health（2026-08-24）

no-candidate/permutation分别在00:26:42/00:34:02保持RUNNING，elapsed为22,361/22,204秒，中央CSV
第110/111行仍RUNNING且无terminal/failure；下一窗口为01:26:42/01:34:02。no-candidate因累计等待
估计失准而早于原窗口131秒，未立即重查，下一窗口改按实际观测锚定；后续调用前须最终校准本地时钟。
edit-metadata本轮未提前重查，下一窗口仍为01:09:26。未新增attempt、未读active metric，protected read=0。

## Route 2 heartbeat final clock-check alignment（2026-08-24）

`route2` heartbeat继续保持ACTIVE/60分钟/failed-only/当前thread target，并新增远端检查前最终本地时钟读取与
远端偏移校准约束；未越过远端`next_check_not_before`时禁止SSH。未创建重复automation，不新增optimizer
attempt，不改变A100 HEAD，protected read=0。

## C2 edit-metadata-only terminal control（2026-08-24）

中央CSV第109行于00:42:40 terminal COMPLETED。固定结果：seed20260830、8 passes、22,416 updates、
29,489,049 trainable parameters、Spearman0.1078162132、standardized MAE1.9265768541、7/9 task为正、
A100/BF16/GPU3、wall24,317.77秒、peak925.234MiB、参数更新、protected read=0。C2 full的Spearman比
edit-metadata-only低0.0035506020且MAE高0.0439439561，故未通过“full beats edit-metadata-only”control。
no-candidate/permutation仍RUNNING，不新增attempt或启动C3。

## 01:28–01:34 C2 two remaining controls health（2026-08-24）

no-candidate/permutation分别在01:28:50/01:34:29保持RUNNING，elapsed为26,089/25,831秒，中央CSV
第110/111行仍RUNNING且无terminal/failure；下一窗口为02:28:50/02:34:29。两次SSH前均完成最终本地
时钟校准。edit-metadata已terminal且不再健康轮询；未新增attempt、未读active metric，protected read=0。

## C2 no-candidate/permutation terminal package（2026-08-24）

中央CSV第110/111行分别于01:51:43/01:59:06 terminal COMPLETED；C2 full及四个预注册control现在全部
terminal。no-candidate固定结果为seed20260830、8 passes、22,416 updates、29,489,049 trainable parameters、
Spearman0.0384855077、standardized MAE1.8590261707、4/9 task为正、A100/BF16/GPU5、
wall27,457.55秒、peak695.638MiB。full的Spearman高0.0657801035，但MAE差0.1114946395。

完整candidate-bundle permutation固定结果为同seed/预算/参数量、Spearman0.0592276162、
standardized MAE1.9698397875、7/9 task为正、A100/BF16/GPU0、wall27,302.53秒、peak925.984MiB；
29,271个recipient中29,259个candidate sequence发生变化，六个适用task保持精确source/task strata并打乱
完整candidate bundle。full的aggregate Spearman高0.0450379950；正式适用task win count留给current-HEAD
gate计算，不手读per-task rows。C2仍因primary gate和edit-metadata control失败而ineligible，不授权confirmation。
两个terminal summary未重读，Development TEST/new Evaluation outcome read=0。审计：
`audits/route_a_v3_route2_xeditcritic_v3_c2_control_package_terminal_v1.json`。

## A100 current-HEAD tests 与 SetFlow formal gate（2026-08-24）

所有`22317ed` launch-head jobs terminal后，A100工作树在clean/correct-branch/无旧PID前提下通过
`git pull --ff-only`同步到`0f21b8e`。current-HEAD固定验证为Critic focused 70/70（55项模型/训练/gate +
15项projection/cache/protocol）、SetFlow focused 30/30、精确V3.3.2 96/96；唯一warning为PyTorch
nested-tensor性能提示，不影响正确性。

随后原子生成SetFlow `screen_gate.json`（3,551 bytes）：status=`XEDITSETFLOW_V3_SCREEN_NO_GO`、
selected arm=`null`、reason=`NO_SELECTABLE_ARM_PASSED`。F2只失败unique≥0.90；F3失败recovery≥0.25、
top-k≥0.15和unique≥0.90。confirmation/additional seed/Development TEST/guidance授权均为false；不新增
optimizer attempt，不重训F0/F2/F3。Development TEST/new Evaluation outcome read=0。审计：
`audits/route_a_v3_route2_a100_current_head_sync_tests_setflow_gate_v1.json`。

## C3 full + four controls launch/initial health（2026-08-24）

在current launch HEAD=`4047f55`、五个output目录/attempt均不存在且Critic gate未生成的前提下，启动唯一
C3 screen cohort。full/source-only/edit-metadata-only于02:55:39分别在GPU3/0/5启动；no-candidate于
03:02:48在GPU1启动；permutation在GPU2释放到10,004MiB后于03:09:08启动。未使用GPU6/7、CPU fallback、
同卡叠加或终止其他进程。

五项≥5分钟首检全部alive/CUDA：显存1,776–1,844MiB，中央CSV第112–116行均RUNNING/BF16，output目录
存在但无run summary/failure；stderr仅同一组已知nested-tensor和transformers加载warning，未读stdout/active
metric。冻结预算为30,472,089 trainable parameters、8 passes、预期22,416 updates、effective batch32、
physical microbatch1。按预计超过4小时的60分钟节奏，下一检查分别不得早于04:02:02、04:09:08、04:15:10。
Critic adjudication/confirmation、TEST和guidance均未授权，protected read=0。审计：
`audits/route_a_v3_route2_xeditcritic_v3_c3_screen_launch_v1.json`。

## XEdit V3 manuscript status addendum（2026-08-24）

C3低频等待期间，将完整论文草稿同步为“保留V2 terminal历史 + 显式V3 active addendum”：Methods加入
projection/endpoint isolation、C2/C3精确容量与controls、SetFlow set-marginal/hybrid设计；Results加入
C0/C1/C2 terminal和SetFlow正式NO-GO，不写任何C3 active metric；Discussion加入C2 control failure与
SetFlow likelihood-generation misalignment边界。旧22个claim markers、evidence registry与旧terminal数值不改。

paper focused tests 30/30。首次精确V3.3.2为95/96，唯一失败是首行标题binding；恢复原标题后affected test
3/3、最终精确V3.3.2 96/96。此项不新增optimizer attempt；A100 tests/sync因`4047f55` C3 jobs active而
deferred，Development TEST/new Evaluation outcome read=0。审计：
`audits/route_a_v3_route2_xedit_v3_manuscript_status_addendum_v1.json`。

## C3 screen first long-interval health（2026-08-24）

以最近远端时钟偏移校准后，04:39:59在五个run各自next-check窗口之后完成一次统一低频健康检查。
C3 full/source-only/edit-metadata-only/no-candidate/permutation均保持alive、中央ledger `RUNNING/BF16`，
elapsed分别为6,259/6,260/6,260/5,834/5,452秒，CUDA显存分别为1,846/1,776/1,846/1,776/
1,846 MiB；均无terminal summary或failure artifact，Critic screen gate仍不存在。下一次统一不得早于远端
05:39:59。

未读stdout、stderr、active curve、性能metric或任何terminal payload；没有新增attempt、代码变化、A100 sync
或重复测试。最近A100 fixed cohort仍为Critic 70/70、SetFlow 30/30、精确V3.3.2 96/96；protected
outcome read=0。审计：
`audits/route_a_v3_route2_xeditcritic_v3_c3_screen_health_20260824_043959.json`。

## C3 screen second long-interval health（2026-08-24）

上一远端观察04:39:59后3,627秒，于05:40:26完成第二次60分钟低频检查，未早于05:39:59窗口。
C3 full/source-only/edit-metadata-only/no-candidate/permutation仍全部alive、中央ledger `RUNNING/BF16`；
elapsed分别为9,886/9,889/9,889/9,460/9,079秒，CUDA显存分别为1,846/1,776/1,846/1,776/
1,846 MiB。均无terminal summary、failure artifact或Critic screen gate；下一次统一不得早于06:40:26。

未读stdout、stderr、active curve、performance metric或terminal payload；无新增attempt、代码变化、A100 sync
或重复测试。Development TEST/new Evaluation outcome read=0。审计：
`audits/route_a_v3_route2_xeditcritic_v3_c3_screen_health_20260824_054026.json`。

## C3 screen third long-interval health（2026-08-24）

05:40:26后3,616秒，于06:40:42完成第三次低频检查，未早于06:40:26窗口。C3 full/source-only/
edit-metadata-only/no-candidate/permutation均alive、中央ledger `RUNNING/BF16`；elapsed分别为13,503/
13,503/13,503/13,075/12,693秒，CUDA显存分别为1,846/1,776/1,846/1,776/1,846 MiB。
五项均无terminal summary、failure artifact或Critic screen gate；下一统一窗口为`>=07:40:42`。

未读stdout、stderr、active curve、metric或terminal payload；无新增attempt、代码变化、A100 sync或重复测试。
Development TEST/new Evaluation outcome read=0。审计：
`audits/route_a_v3_route2_xeditcritic_v3_c3_screen_health_20260824_064042.json`。

## C3 screen fourth long-interval health（2026-08-24）

06:40:42后3,637秒，于07:41:19完成第四次低频检查，未早于07:40:42窗口；五项均已超过4小时。
C3 full/source-only/edit-metadata-only/no-candidate/permutation仍alive、ledger `RUNNING/BF16`；elapsed为
17,140/17,140/17,140/16,712/16,330秒，CUDA显存为1,846/2,120/2,190/1,776/1,846 MiB。
活动显存分配正常，均无terminal summary、failure artifact或screen gate。下一统一窗口`>=08:41:19`。

未读stdout、stderr、active curve、metric或terminal payload；无新增attempt、代码变化、A100 sync或重复测试。
Development TEST/new Evaluation outcome read=0。审计：
`audits/route_a_v3_route2_xeditcritic_v3_c3_screen_health_20260824_074119.json`。

## C3 screen fifth long-interval health（2026-08-24）

07:41:19后3,653秒，于08:42:12完成第五次60分钟检查，未早于08:41:19窗口。C3 full/source-only/
edit-metadata-only/no-candidate/permutation均alive、ledger `RUNNING/BF16`；elapsed为20,792/20,793/
20,793/20,365/19,983秒，CUDA显存为2,190/2,120/2,190/1,776/1,846 MiB。活动分配正常，
无terminal summary、failure artifact或screen gate；下一统一窗口`>=09:42:12`。

未读stdout、stderr、active curve、metric或terminal payload；无新增attempt、代码变化、A100 sync或重复测试。
Development TEST/new Evaluation outcome read=0。审计：
`audits/route_a_v3_route2_xeditcritic_v3_c3_screen_health_20260824_084212.json`。

## C3 screen sixth long-interval health（2026-08-24）

08:42:12后3,650秒，于09:43:02完成第六次60分钟检查，未早于09:42:12窗口。C3 full/source-only/
edit-metadata-only/no-candidate/permutation均alive、ledger `RUNNING/BF16`；elapsed为24,442/24,445/
24,446/24,017/23,635秒，CUDA显存为2,190/2,120/2,190/1,776/1,846 MiB。状态稳定，
无terminal summary、failure artifact或screen gate；下一统一窗口`>=10:43:02`。

未读stdout、stderr、active curve、metric或terminal payload；无新增attempt、代码变化、A100 sync或重复测试。
Development TEST/new Evaluation outcome read=0。审计：
`audits/route_a_v3_route2_xeditcritic_v3_c3_screen_health_20260824_094302.json`。

## C3 screen seventh long-interval health（2026-08-24）

09:43:02后3,666秒，于10:44:08完成第七次60分钟检查，未早于10:43:02窗口。C3 full/source-only/
edit-metadata-only/no-candidate/permutation均alive、ledger `RUNNING/BF16`；elapsed为28,108/28,108/
28,109/27,680/27,301秒，CUDA显存为2,190/2,120/2,190/1,776/1,846 MiB。在线LoRA路径运行较长
但状态正常，无terminal summary、failure artifact或screen gate；下一统一窗口`>=11:44:08`。

未读stdout、stderr、active curve、metric或terminal payload；无新增attempt、代码变化、A100 sync或重复测试。
Development TEST/new Evaluation outcome read=0。审计：
`audits/route_a_v3_route2_xeditcritic_v3_c3_screen_health_20260824_104408.json`。

## C3 screen eighth long-interval health（2026-08-24）

10:44:08后3,678秒，于11:45:26完成第八次60分钟检查，未早于11:44:08窗口。C3 full/source-only/
edit-metadata-only/no-candidate/permutation均alive、ledger `RUNNING/BF16`；elapsed为31,786/31,786/
31,787/31,358/30,979秒，CUDA显存为2,190/2,120/2,190/1,776/1,846 MiB。无terminal summary、
failure artifact或screen gate；下一统一窗口`>=12:45:26`。

未读stdout、stderr、active curve、metric或terminal payload；不据有限健康字段制造进度停滞结论。无新增attempt、
代码变化、A100 sync或重复测试，Development TEST/new Evaluation outcome read=0。审计：
`audits/route_a_v3_route2_xeditcritic_v3_c3_screen_health_20260824_114526.json`。

## C3 screen delayed long-interval health（2026-08-24）

用户方法容量讨论期间没有后台频繁轮询；上一远端观察11:45:26后，于15:20:19完成一次延后但未提前的
健康检查。C3 full/source-only/edit-metadata-only/no-candidate/permutation均alive、ledger `RUNNING/BF16`；
elapsed为44,679/44,679/44,680/44,251/43,870秒，CUDA显存为2,190/2,120/2,190/2,120/
1,846 MiB。无terminal summary、failure artifact或screen gate；下一统一窗口`>=16:20:19`。

未读stdout、stderr、active curve、metric或terminal payload；不依据墙钟时间单独制造停滞结论。无新增attempt、
代码变化、A100 sync或重复测试，Development TEST/new Evaluation outcome read=0。审计：
`audits/route_a_v3_route2_xeditcritic_v3_c3_screen_health_20260824_152019.json`。

## C3 screen ninth long-interval health（2026-08-24）

15:20:19后4,987秒，于16:43:26完成下一次低频检查，未早于16:20:19窗口。C3 full/source-only/
edit-metadata-only/no-candidate/permutation均alive；elapsed为49,666/49,667/49,667/49,238/48,857秒，
CUDA显存为2,190/2,120/2,190/2,120/1,846 MiB。五项均无terminal summary、failure artifact或screen
gate；下一统一窗口`>=17:43:26`。

未读stdout、stderr、active curve、metric或terminal payload；未新增attempt，不做A100 sync或重复测试。
Development TEST/new Evaluation outcome read=0。审计：
`audits/route_a_v3_route2_xeditcritic_v3_c3_screen_health_20260824_164326.json`。

## 23:28–23:33 C2 remaining controls first long-interval health（2026-08-23）

no-candidate/permutation分别在23:28:53/23:33:05保持RUNNING，elapsed为18,893/18,547秒，中央CSV
第110/111行仍RUNNING且无terminal/failure；下一窗口为2026-08-24 00:28:53/00:33:05。
edit-metadata本轮未提前重查，下一窗口仍为00:09:17。未新增attempt、未读active metric，protected read=0。

## 00:09 C2 edit-metadata second long-interval health（2026-08-24）

edit-metadata在00:09:26保持RUNNING，elapsed22,327秒，中央CSV第109行仍RUNNING且无terminal/failure；
下一窗口为01:09:26。no-candidate/permutation本轮未提前重查，仍为00:28:53/00:33:05。
未新增attempt、未读active metric，protected read=0。

## 13:31–13:43 scheduled screen health（2026-08-23）

C2 full/F3 training/C2 source-only分别在13:31:47/13:36:46/13:43:58保持RUNNING，elapsed分别为
13,386/15,427/13,272秒，均无terminal/failure。F3已转60分钟并下一次不早于14:36:46；C2 full与
source-only下一次分别为14:01:47/14:13:58，届时若仍运行也转60分钟。未读active metric、未新增attempt
或叠加GPU任务；protected outcome read=0，A100 HEAD=`22317ed`。

## 14:02–14:37 long-run scheduled screen health（2026-08-23）

C2 full/C2 source-only/F3 training分别在14:02:06/14:15:24/14:37:09保持RUNNING，elapsed分别为
15,205/15,158/19,051秒，均无terminal/failure。三项均已超过4小时并转60分钟；下一窗口分别为
15:02:06/15:15:24/15:37:09。未读active metric、未新增attempt或叠加GPU任务；protected outcome
read=0，A100 HEAD=`22317ed`。

## 15:03–15:40 long-run scheduled screen health（2026-08-23）

C2 full/C2 source-only/F3 training分别在15:03:44/15:19:38/15:40:03保持RUNNING，elapsed分别为
18,903/19,012/22,824秒，均无terminal/failure；source-only与F3中央ledger第108/106行仍为RUNNING。
三项继续使用60分钟窗口，下一窗口分别为16:03:44/16:19:38/16:40:03。未读active metric、未新增attempt
或叠加GPU任务；protected outcome read=0，A100 HEAD=`22317ed`。等待期间只修正interim paper evidence中
“F2 validation仍在运行”的过时句子，使其与已冻结F2 terminal unique-rate gate failure一致；不改变结果或gate，
不新增中央optimizer attempt。

## F3 training terminal / unguided validation launch（2026-08-23）

F3中央CSV第106行在15:41:08 terminal COMPLETED。固定训练事实：seed20260903、5 completed passes、
selected pass3、21,345 updates、42,196,934 trainable parameters、BF16/GPU3、peak VRAM3,134.016MiB，
common Validation set-NLL=2.05042941274086，相对F0改善0.6201436646747034；参数确实更新，TEST/Evaluation
read=0。训练侧NLL gate通过，但不形成arm gate PASS。

F3 unguided validation用仅改变物理device为GPU3的运营config启动，PID49555；5分钟初检健康，下一次不早于
17:20:25。该validation不新增optimizer attempt；recovery/top-k/unique/G0均pending，screen和confirmation
未授权。同期C2 full/source-only仍RUNNING，下一窗口为17:06:57/17:21:45；没有安全GPU启动更多control。

## 17:10–17:22 mixed-interval screen health（2026-08-23）

C2 full/F3 unguided validation/C2 source-only分别在17:10:17/17:20:36/17:22:22保持RUNNING，elapsed分别为
26,497/2,133/26,376秒，均无terminal/failure；F3 stderr为空。C2两项下一窗口为18:10:17/18:22:22，
F3 validation下一窗口为17:50:36。未读active metric、未新增attempt或叠加GPU任务；protected outcome
read=0，A100 HEAD=`22317ed`。

## F2 terminal diversity-by-domain diagnostic（2026-08-23）

F2终态mean unique在GSE269595/GSE114002/ENCSR854RUF/GSE217518分别为0.4359375/0.6580713190/
0.7063078704/0.8220720721，四域均低于0.90；study与endpoint在该cohort一一对应，不能独立归因。
GSE114002占652/891 sources，总体source-macro需与domain-resolved值共同报告。本项不新增attempt，
F2/F3与protected outcome状态不变。

## 12:31–12:41 scheduled screen health（2026-08-23）

C2 full/F3 training/C2 source-only分别在12:31:11/12:35:16/12:41:59保持RUNNING，elapsed分别为
9,751/11,737/9,553秒，均无terminal/failure；下一窗口分别为13:01:11/13:05:16/13:11:59。未读active
metric、未新增attempt或叠加GPU任务；protected outcome read=0，A100 HEAD=`22317ed`。
