# Benchmark v2 矩阵行预注册骨架 v1（Task 2.2 执行闸门）

- **日期**: 2026-09-20
- **性质**: 预注册（preregistration）骨架——在 benchmark v2 新行评测执行**之前**冻结行口径，禁止事后调整。
- **地位**: 本文档是 Task 2.2（矩阵行移植评测执行）的**闸门文档**：Task 2.2 不得在本文档 commit 冻结之前启动任何跑批。
- **上游文档**: [benchmark_v2_port_ledger_v1.md](benchmark_v2_port_ledger_v1.md)（Task 2.1 终审清单，与本文档同 commit 冻结；新行集合以该 ledger 的 PORT_READY 列表为唯一合法来源）
- **关联纪律文档**: `delta_density_prereg_framework_v1.md`（只增行 / 禁挑行 / held-out 预测检验——本文档的行口径条款与其完全兼容并互为引用）。
- **执行环境**: W0 worktree `route-a-v3-w0-diagnosis-20260902`；产物根目录 `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/`。

---

## 1. 行口径总则（五条冻结条款）

### 1.1 frozen-Δ 零调参

- 新移植行一律使用**官方发布权重原样推理**：不微调、不重训、不做任何梯度更新，也不做输出后标定（如分位数映射 / 线性重缩放到目标集分布）。
- 允许且仅允许**输入适配**（见 §1.2）与**数值 dtype/设备搬运**。
- 冻结含义：frozen 权重 → Δ（编辑前后）预测差，即对每个编辑对 (WT, EDIT) 分别前向，行级指标基于 paired Δ 口径与绝对口径两列同时记录（记录格式见 §4）。

### 1.2 输入适配逐行声明

- 每个新行在首次跑批前，必须在跑批 manifest 中写入**逐行输入适配声明**，内容至少含：
  1. tokenization 方案（官方 tokenizer / one-hot / k-mer，逐行取官方口径，禁止自造 tokenizer）；
  2. 截断/补齐规则（长度上限、pad 侧、pad 字符）；
  3. 输入域（5'UTR-only / 3'UTR-only；禁止拼接 CDS 造超域输入）；
  4. 输出头选择（标量回归头；多任务头仓库须声明取哪个头）。
- 声明基准 = [benchmark_v2_port_ledger_v1.md](benchmark_v2_port_ledger_v1.md) §6 的移植适配声明草稿；跑批 manifest 只允许在该草稿上细化（如补具体 pad 值），**不得改变口径方向**。
- 未声明输入适配的行不得开始跑批。

### 1.3 VALIDATION only

- 新行评测仅使用 **GSE114002 VALIDATION** 划分（与 frozen-9 评测相同的 K=10 对齐口径）；**TEST 划分在论文主结果揭盲前对新行同样冻结**（即新行本阶段不产生 TEST 数字；TEST 仅在全部 14-17 行齐备、矩阵冻结后一次性揭盲）。
- 禁止新行接触 TEST 划分的任何统计信息（包括分布摘要）。
- 理由：与 frozen-9 的泄漏控制纪律一致（R3，95% 近重复组件级划分）；新行若提前触碰 TEST 将摧毁矩阵整体的可比性。

### 1.4 评估器同 Task-1

- 评估器**复用 Task-1 的同一评估器实例**（与 frozen-9 评测完全相同的指标实现、配对口径与 bootstrap 流程），不得为新行另写评估器。
- 允许的指标集与 Task-1 完全一致（含 Spearman ρ、行级 delta 可学性指标等既有口径）；**不因新行表现增删指标**。
- 任何评估器代码变更（哪怕 bug fix）须先出变更单并在 matrix 冻结记录中留痕，且重跑受影响的**全部行**（新旧行同权）。

### 1.5 缺格结构化标注规则

- 新行因输入超长、非 ACGUT 字符、tokenization 失败等原因无法产生预测的样本，一律记为**缺格（missing）**，按以下结构化规则处理：
  1. 缺格数、缺格率、缺格原因分布必须随行级结果一同落盘（`miss_count / miss_rate / miss_reason_histogram`）；
  2. 指标计算**剔除缺格样本**（paired Δ 口径下 WT 或 EDIT 任一侧缺格即整对剔除），剔除集 ID 列表落盘；
  3. **缺格率 > 15% 的行**：行级数字正常报告，但须挂 `HIGH_MISS` 标记，且在矩阵表脚注披露；不自动除名；
  4. 禁止用模型默认值/均值填充缺格（填充即口径造假）。
- frozen-9 行沿用 Task-1 已有缺格口径（若有）；两口径若存在差异，以本规则为准对新行执行，frozen-9 保持原样并在矩阵方法学附注中说明差异。

---

## 2. 新行入榜只增行条款（append-only）

- **入榜集合冻结**: 入榜新行 = [benchmark_v2_port_ledger_v1.md](benchmark_v2_port_ledger_v1.md) §4 判定为 PORT_READY 的 5 行（GEMORNA-5utr、GEMORNA-3utr 计 1 个族行集合、LAMAR-UTR5TEPred、UTR-STCNet(MPRA-H)、UTR-Insight、HydraRNA）；本骨架 commit 后**不得增删**（增行须走 §5 变更流程且只能追加）。
- **只增不删**: 任何已产生 VALIDATION 数字的新行不得因结果不佳而移除或重跑（除 §1.4 评估器变更单触发的全矩阵重跑）；行 ID 一经分配永久占用。
- **禁挑行**: 与 `delta_density_prereg_framework_v1.md` (d) 一致——新行纳入理由与规律检验无关；新行作为 held-out 行参与 delta-density 预测检验时，预测 ρ、实测 ρ、|Δρ|、是否落带须记录进该框架的累积表。
- **优先序**: LAMAR-UTR5TEPred → HydraRNA → GEMORNA → UTR-STCNet → UTR-Insight（license 条件确认并行推进；附条件行在确认存档完成前**不得跑批**，确认完成即解锁，顺序不变）。
- **撤行例外**: 附条件 PORT_READY 行若收到作者拒绝/商业限制回复 → 按 ledger 降级 RIGHTS_BLOCKED 并撤行（该行若尚未跑批则直接除名；若已跑批则结果封存不入矩阵主表）。这是唯一合法撤行路径。

---

## 3. 跑批前置闸门（Task 2.2 启动条件清单）

Task 2.2 允许启动当且仅当以下全部满足：

1. ✅ 本预注册骨架与 [benchmark_v2_port_ledger_v1.md](benchmark_v2_port_ledger_v1.md) 已同 commit 冻结并 push；
2. ⬜ 每个待跑行完成**单测参照官方输出**验证（以官方 CLI 示例 / 官方预生成结果 / 官方 test csv 对齐，容差：数值 epsilon 级或逐例一致，具体逐行按 ledger §6 可行性列执行）并在跑批 manifest 记录对齐证据；
3. ⬜ 附条件 PORT_READY 行（GEMORNA / UTR-STCNet / UTR-Insight）的 license 确认动作已发起并**完成存档**（issue/邮件链接入 manifest；无回复不阻塞，收到明确拒绝才阻塞）；
4. ⬜ 输入适配逐行声明（§1.2）写入跑批 manifest；
5. ⬜ 评估器实例与 Task-1 一致性核验（同一 commit 的评估器代码/脚本路径）；
6. ⬜ 环境/资源预估落盘（HydraRNA 的 mamba/flash-attn 环境为最大工程项）。

---

## 4. 行级结果记录格式（预注册）

每个新行落盘一条记录（JSON/CSV 双格式，产物根目录 `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/` 下新建 `benchmark_v2_rows/`）：

| 字段 | 说明 |
|---|---|
| `row_id` | 矩阵行 ID（永久） |
| `family` | 族名（如 `lamar_utr5te`） |
| `weights_path` + `weights_sha256` | 权重位置与哈希（可复现锚点） |
| `input_adaptation` | §1.2 声明全文 |
| `license_status` | ledger 判定 + 确认存档链接 |
| `pred_col_abs` | 绝对口径（WT/EDIT 各自预测值） |
| `pred_col_delta` | paired Δ（EDIT − WT） |
| `metrics_validation` | VALIDATION only 指标（Task-1 评估器输出） |
| `miss_count / miss_rate / miss_reason_histogram / miss_ids` | §1.5 缺格结构化记录 |
| `density_reg_entry` | delta-density 累积表回写条目（预测 ρ / 实测 ρ / Δρ / 落带） |

TEST 字段本阶段**强制为空**（§1.3）。

---

## 5. 变更流程

- 本骨架 v1 的 §1-§4 条款自 commit 起冻结；修改须起草 v2 并按主合同 §0.4 变更流程走审，旧版下已产生的评测结果保持有效。
- 与 `delta_density_prereg_framework_v1.md` 冲突时，行口径以本文档为准，规律检验数值（0.10 / 70% 等）以该文档为准。

---

## 版本与冻结声明

- v1 自 commit 起冻结。Task 2.2 的第一行跑批时间不得早于本 commit；若任何跑批先于本 commit 发生，该批结果无效并须在矩阵冻结记录中声明。
