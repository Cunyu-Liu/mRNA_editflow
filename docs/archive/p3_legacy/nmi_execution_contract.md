# NMI Execution Contract — Phase 0 Freeze Baseline

本契约为 Phase 0 的唯一可信执行基线。任何论文候选数字、任何 Gate
判定、任何下游实验，必须能回溯到本契约冻结的代码、数据、配置与
checkpoint。

## 1. 冻结内容

`artifacts/nmi_phase0_freeze_manifest.json` 记录：

| 类别 | 内容 |
|---|---|
| Git | HEAD commit、分支、dirty 文件清单（不掩盖未提交改动） |
| 数据 | 所有 P3 benchmark tier 的路径与 SHA256 |
| Split | split audit 文件的 SHA256 |
| Checkpoint | 全部 delta-oracle 与 GRPO checkpoint 的 SHA256 |
| 配置 | `configs/nmi_execution.yaml` 自身的 SHA256 |
| 环境 | Python、PyTorch、CUDA、cuDNN、GPU 型号、主机名 |
| Seed | 全部实验种子 |
| 生成命令 | 每个 artifact group 的可复现命令 |
| 依赖图 | raw data → preprocessing → split → checkpoint → inference → stats → table |

## 2. 执行规则

1. **唯一真值源**：`configs/nmi_execution.yaml` 是 manifest 的唯一输入；
   新增 artifact 必须先登记到该文件，否则视为 unknown provenance。
2. **禁止静默覆盖**：任何 frozen artifact（manifest 中已记录 hash 的
   文件）被覆盖前，必须先重新生成 manifest 并比对差异；覆盖必须伴随
   新的生成命令记录。
3. **验收门槛**（strict 模式）：
   ```bash
   python scripts/build_freeze_manifest.py \
     --config configs/nmi_execution.yaml \
     --strict
   ```
   必须零缺失、零 unknown provenance、依赖图引用零悬垂。
4. **环境一致**：训练与评估必须使用 manifest 记录的 conda 环境
   （`editflow`）；GPU 训练不允许回退 CPU。
5. **Seed 冻结**：所有随机过程必须使用 manifest `seeds` 中登记的
   种子；新增种子视为契约变更，需按
   `docs/p3_00_change_governance.md` 审批。

## 3. 变更流程

任何对 frozen 基线的修改（代码、数据、配置、checkpoint）：

1. 在 git 中提交修改；
2. 更新 `configs/nmi_execution.yaml`（如有新 artifact）；
3. 重新运行 strict 模式生成新 manifest；
4. 在差异说明中记录旧 hash → 新 hash 的映射。

## 4. 责任边界

- manifest 只证明「冻结时刻的状态」；不证明历史实验在该状态下运行。
- 历史结论如需保留，必须在本基线上重跑并通过 Phase 0 全部验收。
