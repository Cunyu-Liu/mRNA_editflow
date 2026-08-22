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

GitHub builder commit `363c741` 已推送且 A100 focused test 2/2；本机十组联合 suite 32/32。
paper evidence sources=38（25 local/contract + 13 `/mnt`）、claim markers=22、
figures/builders=6/5、MBP=14/3/1、blockers=MBP-10/13/14/15 与
`submission_ready=false` 不变。中央 100 个 attempts 仍为 92 `COMPLETED`、3 `FAILED`、
3 `INCOMPLETE_NO_TERMINAL_RECORD`、1 `STOPPED_FOR_THROUGHPUT_REPAIR`、1
`STOPPED_PRIORITY_REALLOCATION`。A100 联合 32-test suite 等待 paper integration commit
推送后单次执行并补记。
