# P3-01 Local-Edit Benchmark（Task 2/3/4/5）

冻结时间：2026-07-23T04:21:52Z ｜ Schema：`p3_benchmark_v1` ｜ 构建 commit：`3e85c794f2c94e73c851050410885ae46a2cbf08`
机器可读产物：`data/p3/manifests/`（3 个 JSON，见 §6）、`docs/p3_01_split_audit.json`

> 本 benchmark 是 P3-00A 冻结契约 `p3_task_v2` 的数据对应物：为**局部编辑**
> （而非自然序列绝对标签）提供训练/评测基础。primary task = Task A
> （5'UTR substitution，edit_budget ≤ 10，protein_output 主端点）。Task B/C
> 的数据仅作为门控任务资产（unlabeled），不是 local-delta ground truth。

## 1. Record schema（16 必需字段 + 治理扩展字段）

必需字段（`p3_benchmark_v1`，`BenchmarkRecord.validate()` 强制）：

`record_id, source_id, cargo_id, cell_context, source_sequence, candidate_sequence,
edit_list, edit_count, edited_region, protein_identity,
measured_or_proxy_source_value, measured_or_proxy_candidate_value,
delta, data_source, assay_type, confidence`

治理/溯源扩展：`edit_type, task_eligibility, value_qualifier, value_std,
family_cluster_id, motif_flags, internal_features, split_role`。

坐标系：RNA 字母表；`edit_list.pos` 为 `source_sequence`（= 编辑区域 scope）内的
0-based 偏移；把 edits 应用到 `source_sequence` 必须精确重现 `candidate_sequence`。

**反伪造不变式（硬校验）**：`confidence == "unlabeled"` ⟺ values/delta 全为 null。
绝不把缺少局部标签的数据伪装成 local-delta ground truth。

## 2. 三个 tier

| tier | 记录数 | 语义 | 端点限定词 |
|---|---:|---|---|
| measured | 4802 | Sample2019 MPRA snv 库：同一 mother 的 WT 锚定 5'UTR 变体；`delta = variant_rl − WT中位数_rl`；无锚变体剔除 | wet-lab measured（MPRA ribosome load） |
| proxy | 473228 | 重建源 5'UTR 前 50nt 受控邻域，CNN-50mer cross-fitted ensemble（P1-04，15 ckpts）mean/std | predicted/internal proxy |
| unlabeled | 839288 | 合法编辑（5'UTR 超出 proxy 窗口；同义 CDS scopes；joint 5'UTR+CDS），values 全 null | none（契约禁止当真值用） |

Tier 文件（gitignored，见 §6 manifest 内的 SHA-256）：
`data/p3/benchmark/{measured,proxy,unlabeled}_tier.jsonl`

## 3. Controlled neighborhoods（5 类编辑 × 5 区域）

每个合格 source，按区域生成：

- `all_legal_single`：合法单编辑全枚举（`cds_remaining` 普查上限 400，等距抽样，已记录）
- `random_double`：种子随机双编辑对（位置不同；CDS 区域额外要求**不同密码子**）
- `structure_guided_double`：局部结构扰动（±6nt 最近邻能差）top-pool 配对
- `topranked_double`：区域内部排序特征 |delta| top-pool 配对
- `matched_negative_single`：预测 |delta| 近零的单编辑

区域与任务归属（`p3_task_v2`）：
`five_utr`→task_a_active；`cds_first30/first50/remaining`→task_b_frozen_fallback；
`joint_5utr_cds`→task_c_locked_extension。start/stop 密码子永不编辑。

**合法性构造保证**（motif_policy_v1 hard_forbidden 动作空间排除）：无上游同框
AUG、无新隐剪接供体/受体（确定性 regex 代理，见 data_limitations）、无新 ≥6nt
homopolymer；CDS 编辑构造即同义（只枚举同义密码子的单碱基差异），且**每条
CDS/joint 记录都对全长 CDS 重新翻译核验 protein identity，失败即丢弃**
（fail-closed，本次构建 `protein_identity_failures=0`）。
guarded_risk tier（m6a DRACH 计数变化、4–5nt run）只记录于 `motif_flags`，不判非法。

## 4. Split（邻域泄漏防护）

分组原子性：同一 `source_id` 的全部 candidate（跨全部 tier）永不跨角色。
角色语义：

- train/val：source-disjoint
- test：**cargo/family-disjoint**——整个 protein family 从其他所有角色中扣出
  （含 OOD：被 OOD 触到的 family 不再进入 test 候选）
- ood：按 **cohort 内** GC/length 分布尾部分位（q=0.05，严格不等式）标记，
  避免 measured 50nt 定长库被整体误判、避免 128nt 众数堆叠被边界比较吞并

全局 assignment：`data/p3/manifests/p3_01_split_assignment.json`，
`assignment_sha256=75f697089bdc8a31f34475c5cbcf44f30c7e83e50a279f76517ab8adb7018b0e`。

| role | 记录数 | 组数 |
|---|---:|---:|
| train | 883992 | 2129 |
| val | 132920 | 329 |
| test | 135104 | 322 |
| ood | 165302 | 383 |

审计（`docs/p3_01_split_audit.json`）：source 跨角色违规 0；
跨角色精确序列碰撞 0；test family 违规 0；
trainval∩ood family 重叠 39（允许但披露）。**audit_passed=true**。

## 5. Source 合格性过滤（fail-closed，已记录剔除数）

输入重建记录 148843 → 合格 93318。剔除：5'UTR <50nt
0；非 ACGU 0（计入同类）；无效 CDS；源含上游同框 AUG
0；源含 ≥6nt homopolymer 0；源含隐剪接共有序列 0。
measured 侧：snv 行 8738 → mother 2647 → 有锚 867
（无锚变体全部剔除）；WT 锚值取 WT 重复的中位数；hamming 1..10 保留。

## 6. 冻结 manifests（Task 5）

`data/p3/manifests/`：

| 文件 | 内容 | SHA-256 |
|---|---|---|
| p3_01_benchmark_manifest.json | dataset card、tier 文件 hash、输入 raw hash、15 个 ckpt hash、过滤规则、split hash、known confounders、构建命令 | bd7748f74b272c26415bd3297d031b31be71b38e0781d6fa1b8ac9f57b7f9033 |
| p3_01_split_assignment.json | 全部 group→role 分配（小文件，入库冻结） | de89c499d542887000a4a639a2cbdf64e9db0453419b8871145443417e59a098 |
| p3_01_dataset_registry.json | Task 1 四级注册表（license/raw hash） | 37f87a702c901036a50a93da20efd08df744ae830c60d3a985523104aa67758d |

外加 `docs/p3_01_split_audit.json`（961668f1fa60570064e59867f3256e3423700f35f842b01132615390be20a0f5）。Tier JSONL 本体按仓库惯例
gitignore（`.gitignore: data/p3/benchmark/`），其 SHA-256 冻结在 manifest 中，
任何再生成的文件可用 `scripts/p3_01_build_benchmark.py` 以同一种子复现核验
（resume 复现性由集成测试 `test_resume_reproduces_identical_hashes` 保证）。

## 7. 与 primary task 的对齐声明

Task A（5'UTR substitution）的训练/评测只能使用：`measured` tier（`task_a_active`）
与 `proxy` tier（带 predicted/internal proxy 限定词）。`unlabeled` tier 中
`task_eligibility=task_a_active` 的记录（5'UTR 50nt 窗口外编辑）**没有标签**，
只能作为动作空间/合法性资产。Task B/C 区域数据全部 unlabeled，服务于门控任务的
后续解锁条件，不改变 P3-00A 的任何冻结判定。
