# Route 2 mRNABERT 主模型：参数位置与训练/测试命令

## 1. 当前替换关系

- 主冻结编码器：`YYLY66/mRNABERT`，固定版本 `a1eb7df25804d23f08646e1cb996b234d7208a40`。
- RNA-FM：仅保留为已经完成的历史对照，不再是新的主训练配置。
- mRNABERT 只负责预先计算冻结的全局序列表征；冻结编码器为 113,389,056 参数。Huber/固定方差 critic 为 9,343,106 个可训练参数，总有效规模 122,732,162；可学习方差版本为 9,343,491 个可训练参数。
- 服务器使用 mRNABERT 官方代码内置的 PyTorch attention fallback；原因是官方仓库附带的旧 Triton kernel 与服务器当前 Triton API 不兼容。该选择不改变模型权重或注意力计算定义。
- Development 共 126,165 条；HPO 只读 TRAIN/VALIDATION，冻结测试才读 TEST；Evaluation pool 不参与训练、调参或测试选择。

## 2. 参数在哪些文件修改

| 参数类别 | 文件 | 主要字段/位置 |
|---|---|---|
| mRNABERT 模型版本、GPU、分块、特征缓存输出 | `configs/route_a_v3_route2_mrnabert_full_development_feature_cache_v1.json` | `model_id`、`mrnabert_model_path`、`physical_gpu_index`、`maximum_chunk_nucleotides`、`maximum_sequences_per_batch`、`batch_token_budget` |
| 主模型结构 | `core/route2_delta_predictor.py` | `Route2PretrainedEditCenteredDeltaPredictor`；全局预训练背景、edit-centered pooling、差分表征和反对称输出 |
| 主训练超参数 | `configs/route_a_v3_route2_mrnabert_edit_max_mean_only_gpu6_v1.json` | `hidden_dim=384`、`depth=10`、`batch_size=16`、`learning_rate=1e-4`、`weight_decay=1e-4`、`epochs=100`、`loss_kind=huber` |
| 固定方差 NLL 对照 | `configs/route_a_v3_route2_mrnabert_edit_max_fixed_variance_gpu6_v1.json` | 与主模型相同，只把 `loss_kind` 改为 `fixed_variance_gaussian_nll` |
| 可学习方差 NLL 对照 | `configs/route_a_v3_route2_mrnabert_edit_max_learned_variance_gpu6_v1.json` | 与主模型相同，只把 `loss_kind` 改为 `learned_variance_gaussian_nll` |
| 冻结 Development TEST | `configs/route_a_v3_route2_mrnabert_edit_max_mean_only_frozen_test_gpu6_v1.json` | `result_stage=FROZEN_DEVELOPMENT_TEST`；TRAIN+VALIDATION 拟合后评测 18,292 条 TEST |
| 全部 126,165 条最终拟合 | `configs/route_a_v3_route2_mrnabert_edit_max_mean_only_all126165_gpu6_v1.json` | `result_stage=FINAL_ALL_DEVELOPMENT_REFIT`；不再产生内部验证/测试指标 |

训练配置中的 `physical_gpu_index` 与 `device` 必须一起改，例如换到 GPU 4 时同时改为：

```json
"device": "cuda:4",
"physical_gpu_index": 4
```

本项目不使用 `CUDA_VISIBLE_DEVICES` 重映射；直接使用物理 GPU 编号。

## 3. 服务器环境

```bash
ssh A100
cd /home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_route2_method_repair_20260817
PY=/home/cunyuliu/miniconda3/envs/editflow/bin/python
```

## 4. 代码测试

```bash
$PY -m pytest -q \
  tests/route_a_v3/test_build_route2_mrnabert_feature_cache_v1.py \
  tests/route_a_v3/test_build_route2_rnafm_feature_cache_v1.py \
  tests/route_a_v3/test_route2_delta_predictor_v1.py
```

## 5. 构建 126,165 条记录的 mRNABERT 冻结特征

前台执行：

```bash
$PY scripts/route_a_v3/build_route2_mrnabert_feature_cache_v1.py \
  --config configs/route_a_v3_route2_mrnabert_full_development_feature_cache_v1.json
```

后台执行：

```bash
mkdir -p /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/logs
nohup $PY -u scripts/route_a_v3/build_route2_mrnabert_feature_cache_v1.py \
  --config configs/route_a_v3_route2_mrnabert_full_development_feature_cache_v1.json \
  > /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/logs/mrnabert_feature_cache_gpu6.log 2>&1 &
```

输出：

- `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/pretrained_features/mrnabert_full_development_v1.pt`
- `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/pretrained_features/mrnabert_full_development_v1.summary.json`

## 6. 三种 loss 的训练命令

主模型 Huber：

```bash
$PY scripts/route_a_v3/train_route2_delta_predictor_v1.py \
  --config configs/route_a_v3_route2_mrnabert_edit_max_mean_only_gpu6_v1.json
```

固定方差 Gaussian NLL：

```bash
$PY scripts/route_a_v3/train_route2_delta_predictor_v1.py \
  --config configs/route_a_v3_route2_mrnabert_edit_max_fixed_variance_gpu6_v1.json
```

可学习方差 Gaussian NLL：

```bash
$PY scripts/route_a_v3/train_route2_delta_predictor_v1.py \
  --config configs/route_a_v3_route2_mrnabert_edit_max_learned_variance_gpu6_v1.json
```

每个 HPO 配置训练 100 epochs，TRAIN 为 89,580 条，约 559,900 次 optimizer updates；只在 VALIDATION 选 checkpoint，不读取 Development TEST 或 Evaluation pool。

## 7. 冻结测试命令

只有在三个 loss 的 VALIDATION 比较完成并确定主 loss 后运行：

```bash
$PY scripts/route_a_v3/train_route2_delta_predictor_v1.py \
  --config configs/route_a_v3_route2_mrnabert_edit_max_mean_only_frozen_test_gpu6_v1.json
```

该命令用 TRAIN+VALIDATION 共 107,873 条拟合 100 epochs，约 674,300 次 updates，然后一次性评测 TEST 18,292 条。主要结果：

- `training_summary.json`：参数量、GPU、更新次数、TEST 指标；
- `test_predictions.jsonl`：逐记录冻结测试预测；
- `delta_predictor_checkpoint.pt`：冻结 checkpoint。

## 8. 全 Development 最终拟合

冻结测试结束后，若要为后续 Edit Flow guidance 生成使用全部 Development 的 critic：

```bash
$PY scripts/route_a_v3/train_route2_delta_predictor_v1.py \
  --config configs/route_a_v3_route2_mrnabert_edit_max_mean_only_all126165_gpu6_v1.json
```

该命令使用全部 126,165 条记录训练 100 epochs，约 788,600 次 updates；因为没有内部留出集，所以它不用于重新选择模型或汇报测试性能。

## 9. 结果查看

不需要频繁轮询。任务结束后读取一次：

```bash
python -m json.tool \
  /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/mrnabert_scaleup_v1/max_mean_only_seed20260816_gpu6_v1/training_summary.json
```

运行中只在需要确认是否正常推进时低频查看：

```bash
tail -n 20 \
  /mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/runs/mrnabert_scaleup_v1/max_mean_only_seed20260816_gpu6_v1/metrics.jsonl
```
