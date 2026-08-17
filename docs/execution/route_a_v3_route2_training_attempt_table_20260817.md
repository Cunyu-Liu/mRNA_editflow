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
