# P3-01 数据局限性与敏感性报告

冻结时间：2026-07-23T04:21:52Z ｜ 构建命令：`scripts/p3_01_resolve_collisions.py`

## 1. 观测 vs 干预标签区分

| tier | 标签类型 | 可否充当 local-delta ground truth |
|---|---|---|
| measured | wet-lab measured (MPRA ribosome load) | **是**——同一 mother 的 WT 锚定变体；delta = variant − WT 中位数 |
| proxy | predicted/internal proxy (CNN-50mer ensemble) | **有条件是**——必须带 `value_qualifier="predicted/internal proxy"` 限定词 |
| unlabeled | 无标签 (null) | **否**——契约禁止作为 local-delta ground truth |

## 2. 已知 confounders

1. measured tier: single cargo (EGFP), single cell context (HEK293T), 50 nt 5'UTR window, ribosome-load endpoint (not protein output)
2. proxy tier: CNN-50mer ensemble was trained on the SAME MPRA assay family as the measured tier (assay-family correlation; NOT an independent oracle)
3. measured mothers are human SNP-context 50-mers, not therapeutic UTRs
4. GC/length distributions differ between measured and reconstructed cohorts (handled by per-cohort OOD flagging)
5. unlabeled tier carries NO values; any downstream use as local-delta ground truth is a contract violation
6. 5'UTR proxy coverage limited to first 50 nt; longer-UTR edits are unlabeled by design

## 3. 数据敏感性分析

### 3.1 measured tier 局限
- **单一 cargo**：全部 snv 变体仅测定 EGFP 报告基因；其他 cargo（治疗性蛋白）
  的编辑效应无法从 measured tier 推断。
- **单一细胞上下文**：全部为 HEK293T；其他细胞类型（如 dendritic cells、
  hepatocytes）的 TE 可能不同。
- **50nt 5'UTR 窗口**：snv 库设计为 50nt 随机上下文，不代表全长治疗性 UTR。
- **端点为 ribosome load**：不是 protein output；翻译效率与蛋白产量之间的
  非线性关系（如翻译停滞、蛋白降解）未被捕获。

### 3.2 proxy tier 局限
- **assay-family 相关性**：CNN-50mer ensemble 训练于**同一 MPRA assay 家族**
  （Sample 2019），不是独立 oracle；proxy 与 measured 之间存在系统性偏差
  （assay-family correlation），不可用于跨 assay 绝对比较。
- **前 50nt 覆盖限制**：仅覆盖 5'UTR 前 50nt 窗口；更长 UTR 的编辑效应
  为 unlabeled，无 proxy 预测。
- **确定性非校准**：ensemble 提供 mean/std，但未做 post-hoc calibration
  （如 isotonic regression）；std 仅反映模型不确定性，不反映认知不确定性。

### 3.3 unlabeled tier 局限
- **无值字段**：`confidence="unlabeled"` 的记录 `delta` 等全为 null。
  任何下游使用作为 local-delta ground truth 构成契约违规。
- **CDS 编辑仅同义**：由构造保证 protein identity，但不捕获密码子使用偏好
  （codon usage bias）对翻译速率的效应（该效应需要 ribosome profiling 数据，
  当前不包含）。
- **joint 编辑仅 5'UTR+CDS_first50**：跨区域协同效应的覆盖有限。

### 3.4 split 敏感性
- **cohort 间分布差异**：measured（50nt, sample2019_snv cohort）与
  reconstructed（变长, reconstructed cohort）的 GC/length 分布不同。
  per-cohort OOD 标记避免定长库被整体误判，但 cohort 间的系统性差异
  仍是潜在 confounder。
- **跨角色精确序列碰撞**：9360 个 candidate
  序列跨 role 出现（近同源 source 在不同 family 中生成相同编辑），已通过
  确定性消解删除 11458 条 train 记录
  （test/ood/val 零丢失）。消解策略：保留最高优先级 role
  （test > ood > val > train），删除其余。

## 4. 伪局部标签防护

- `BenchmarkRecord.validate()` 强制 `confidence == "unlabeled"` ⟺ values/delta
  全为 null（反伪造不变式，由单元测试 `test_unlabeled_must_have_null_values`
  保证）。
- `task_eligibility` 字段标记每条记录的任务归属（`task_a_active` /
  `task_b_frozen_fallback` / `task_c_locked_extension`），防止跨任务误用。
- `value_qualifier` 字段在 proxy tier 每条记录上标记
  `predicted/internal proxy`，满足项目硬约束"任何 improves TE/stability/expression
  必须加 predicted/internal proxy 限定词"。

## 5. split audit 结果

- source 跨角色违规：0
- 跨角色精确序列碰撞：0
- test family 违规：0
- trainval∩ood family 重叠：39（允许但披露）
- **audit_passed = true**

## 6. 蛋白质一致性保证

- CDS 编辑由构造保证同义（只枚举同义密码子的单碱基差异）
- **每条 CDS/joint 记录都对全长 CDS 重新翻译核验 protein identity**
- 本次构建 protein_identity_failures = 0（全部 fail-closed 丢弃）
- 单元测试 `test_legal_cds_subs_synonymous_and_start_stop_untouched` 验证
  start/stop 密码子永不编辑

## 7. Motif policy v1 执行

- **hard_forbidden**（动作空间排除）：上游同框 AUG、新隐剪接供体/受体
  （GT-AG 共有序列）、新 ≥6nt homopolymer
- **guarded_risk**（记录但不判非法）：m6A DRACH motif 变化、4-5nt run 变化
- **soft_objective**（reward shaping 用）：后续 P3-02 RL 阶段使用
- 确定性 regex 代理实现，已知局限：不捕获非经典剪接位点、RNA 二级结构
  变化、RBP 结合位点丢失等（这些需后续 oracle 验证）

## 8. 可复现性

- 全部构建使用固定种子 `seed=20260723`
- `scripts/p3_01_build_benchmark.py` 以同一种子可复现 byte-identical tier JSONL
- resume 复现性由集成测试 `test_resume_reproduces_identical_hashes` 保证
- 碰撞消解为确定性操作（`resolve_cross_role_sequence_collisions`），相同输入
  产生相同输出（由 `test_deterministic_and_audit_passes_after_drop` 验证）
