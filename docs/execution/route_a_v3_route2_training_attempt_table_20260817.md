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
| 训练 | loss、batch、epochs、学习率、weight decay、优化器、BF16/FP32、GPU、seed、optimizer steps、最佳 epoch |
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
| 数据搬运 | 已启用 pinned memory 与 non-blocking transfer | Development 特征已缓存；当前数据集在主进程内完成长度分桶，`num_workers=0` 是实测 profile 的组成部分，不把多 worker 当作默认更快。 |
| 批处理 | 已使用 length-bucket batch sampler | 减少不同长度序列混批造成的 padding 浪费；排名 loss 另使用 source-group sampler。 |
| mRNABERT 位置编码 | 保留官方双向 ALiBi | 这是预训练模型的一部分；直接改成 RoPE 会造成架构与预训练权重不一致，因此不做无依据替换。 |
| critic 位置特征 | 已启用归一化绝对位置 + edit-gated 位置 | 让模型直接知道编辑发生在序列的相对位置，同时避免不同长度序列使用不可比的绝对索引。 |
| Flash/SDPA attention | 正在做候选后端验证 | 不修改 ALiBi 语义；先做 attention primitive，再做完整 encoder 输出对齐与速度比较。只有缓存与新序列一致、且完整编码器实测至少提速 10% 才启用 PyTorch SDPA，否则继续使用官方实现。 |
| `torch.compile` | 暂未启用 | 当前 trainer 明确记录为未实现；在没有端到端收益证据前不把编译开关写成已优化。 |
| uncertainty head | 三种匹配 loss 正在比较 | Huber、固定方差 NLL、学习方差 NLL 使用同一数据、split、架构、seed 和训练预算；选择依赖均值预测表现，NLL 更低不能单独胜出。 |
| Edit Flow 计算 | 已加入重复状态的 rate cache | 逻辑 trajectory 决策次数与真实 generator forward 次数分开记账，使 matched-budget baseline 使用实际计算成本。 |

## 当前三种 loss 的冻结比较口径

- Development 共 126,165 条 canonical source-candidate records。
- 当前 loss 选择只使用冻结的 TRAIN 89,580 条与 VALIDATION 18,293 条；Development TEST 18,292 条保持未打开。
- 三个主实验均为 100 epochs、batch size 16、learning rate `1e-4`、weight decay `1e-4`、BF16、fused AdamW、相同 seed 和相同约 9.34M 可训练 critic。
- 比较顺序为：最大化 task-macro Spearman，其次最小化 task-macro standardized MAE，最后比较 global Spearman。
- 学习方差模型还报告 prediction spread、预测标准差与绝对残差的相关性，用于识别 uncertainty 是否吸收误差或伴随 mean collapse；这些诊断不改变上述选择规则。
- loss 选定后才运行 candidate permutation 与 parameter-matched source-only controls；controls 通过后才进入三个 final seeds、一次冻结 Development TEST 和全 126,165 条最终 refit。

## 记录边界

- `training_attempts.csv` 是 Excel 可直接打开的跨实验总表；每个 run 的 `training_config.json` 保存该次尝试的完整参数，避免总表为了可读性遗漏长配置。
- attention benchmark、encoder 一致性验证等没有参数更新的工程实验保存独立报告，不冒充训练尝试。
- 后续若要扩展中央表字段，应在当前长训练全部结束后升级，避免正在运行的旧进程以旧表头覆盖新表头；当前运行期间不修改中央 CSV schema。
