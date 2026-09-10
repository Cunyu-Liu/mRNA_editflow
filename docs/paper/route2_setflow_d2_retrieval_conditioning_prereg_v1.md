# Route2 SetFlow D2 检索条件化（Retrieval-Conditioned Guidance）预注册 v1

> **状态：PREREGISTRATION（2026-09-10，用户批准执行：「执行 D2 检索条件化」）**。触发依据 = spec §4-D2 条款（D1 q 验收 FAIL → 检索升主选，journal 批 52 翻转登记）+ amendment v2（判定口径）。执行纪律照抄项目惯例：预注册门槛不事后改、CPU fallback 禁用、protected reads = 0、产物 /mnt 代码 worktree、两级设计（calib100 筛选 → 891 确认）。

## 1. 科学假设与机制

**假设 H-D2**：把「同任务、相似源的真实 measured 编辑」作为 in-context 证据注入引导势能（检索条件化），能提供 V5/V8 critic 都不具备的**域内经验信号**——因为检索内容是「实验者真实测过且有方向性 delta 的编辑」，其信号形式与「判别力」正交（检索提供先验方向，critic 提供绝对评分）。

**机制（势能定义）**：对候选 c（源 s，任务 t）：

```
V_retrieval(c | s, t) = Σ_{k=1..K} sim(s, s_k) · w(c, e_k) · sign_k
```

- 检索池：TRAIN split 源的 measured 编辑对（source_sequence_k, edited_k, direction_normalized_delta_k），按任务 t 过滤（endpoint_id 一致）；
- 检索键：源序列 Hamming 距离（同 region 且序列同长约束下的前 K 近邻；K = 8 预注册）；
- `w(c, e_k)` = 候选 c 与编辑后序列 e_k 的相似度（同长 Hamming → 1/(1+d)，d=0 时 w=1）；
- `sign_k` = direction_normalized_delta_k 的符号与幅度（clip 到 [−1, +1]）；
- **组合**：V = V_retrieval 与 V5 critic 组合 `β_r · V_retrieval + β_c · V_V5`（两 β 预注册 sweep）；纯检索臂（β_c=0）单列。

## 2. 去污染硬门（发射前置，全部机械可查）

1. **池隔离**：检索池 = TRAIN split only（`projections/xedit_v3/.../train.jsonl`，split=='TRAIN'）；与 VALIDATION 源的 `connected_source_component_id` 交集必须 = 0（已审计 09-08：TRAIN 5,758 ∩ VAL 3,439 = 0 ✓，q 模型审计复用）；
2. **序列级排除**：检索池中与**任何验证源 measured 候选序列**完全相同的编辑后序列剔除（防「抄答案」——measured_neighborhood 2449 行全量比对）；audit 输出剔除计数，须落盘 `retrieval_pool_audit.json`；
3. **源重叠排除**：检索池的 source_id ∩ 891 验证源 source_id = ∅（manifest 键比对）；
4. **protected reads = 0**：检索池内 direction_normalized_delta 是 TRAIN 标签（训练侧允许），验证侧评估只用既有口径（recovery 家族 + frozen 独立评估器），不触任何 evaluation outcome。

## 3. 实验设计

| 项 | 预注册值 |
|---|---|
| 判定口径 | **amendment v2**：B2 Δ门（Δsc-hit@1 ≥ +0.03 CI 排零，B 档）+ **B2-I 独立口径门**（MRL Optimus Δ ≥ +0.02 CI 排零）+ B3 勘误线（0.55）双报 |
| cohort 第一级 | calib100（`beta_sweep_20260908/calib100_keys.txt` 冻结集复用） |
| cohort 第二级 | 891 全量确认（仅第一级阳性臂晋级） |
| 臂 | A 纯检索（β_c=0, β_r ∈ {0.25, 0.5, 1, 2}）；B 组合（β_r 固定 sweep 最优, β_c ∈ {0.25, 1}）；C 对照（V5 β=0.25 = 已有 891 产物免重跑，直接引用批 60 数字） |
| K | 8（近邻数） |
| GPU | 检索索引 CPU 构建；推理 GPU4/5 空闲即用（watcher 模式同 Phase C） |
| 预算 | calib100 级 5 臂 × ~1h ≈ 5 GPU·h；891 级仅阳性臂 ~10h/臂 |
| 终止条件 | 第一级全臂 B2 与 B2-I 双 FAIL → H-D2 否证，负收口（不追加）；任一门过 → 891 确认 |

## 4. 预期与解释框架（发射前冻结）

- 若组合臂过 B2-I 而 V5 单独不过（对照 C 已知 recovery FAIL + B2-I PASS）：检索信号与 critic 信号互补 → H-D2 证实；
- 若纯检索臂即过 B2 Δ门（recovery 家族）：检索直接改善覆盖/排序——与 A2「引导零覆盖贡献」结论冲突，需重查（预登记此张力）；
- 若全负：与 q 模型负结果合并解读 =「measured 邻域信息（无论学习 q 还是直接检索）在当前验证体制下不足以驱动 recovery 家族改善」，H5/H4 收口强化。
