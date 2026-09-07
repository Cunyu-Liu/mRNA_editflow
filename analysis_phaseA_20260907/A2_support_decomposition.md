# A2 — 支持度分解：measured-in-pool rate 与 support(32) vs support(256)

日期 2026-09-07 · 离线 CPU · protected reads = 0 · 产物 `A2_support_decomposition.json`

## 1. 定义与输入

- **measured-in-pool rate（每源）** = 该源生成池（按序列去重）∩ 真实 measured 候选集合 的候选数 / 池大小（32）。
- **support(32)** = measured-in-pool rate > 0 的源占比（即“32 候选池里至少出现 1 个真实 measured 候选”）。
- 输入池：unguided `b2_full_891/unguided/generated_candidates.private.jsonl`（891×32）、guided `b2_full_891/guided/...`（891×32）。TreeG 产物为 summary-only（见 §4），无逐源 selections 可算 measured-in-pool，故标 N/A。

## 2. per-task × 池 支持度表

| task | n | unguided measured-in-pool(32) | unguided support(32) | guided measured-in-pool(32) | guided support(32) |
|---|---|---|---|---|---|
| MRL | 652 | 0.01280 | **0.3129** | 0.01328 | **0.3083** |
| MPRAU | 108 | 0.00174 | 0.0556 | 0.00203 | 0.0648 |
| HL | 111 | 0.00197 | 0.0541 | 0.00310 | 0.0721 |
| polyA | 20 | 0.00000 | 0.0000 | 0.00000 | 0.0000 |
| **OVERALL** | 891 | **0.00982** | **0.2424** | **0.01035** | **0.2424** |

（TreeG 列：k1/k2 见 §4，本质是 unguided32 池上的 top-K 选择，无独立 256 池，故不纳入本表。）

解读：
- **guided 与 unguided 的 support(32) 完全一致（0.2424）/ measured-in-pool 仅 0.0098→0.0103** → **guidance（frozen XEditCritic V5 引导）对“池里是否有真实 measured 候选”这一覆盖度几乎为零贡献**。引导没有把更多真实候选推进 32 池。
- 期望每源 32 池内真实 measured 候选数 = 0.00982×32 ≈ **0.31**（多数源 0 个，个别源 1 个）。
- 低 measured-in-pool 主要集中在小任务：MPRAU/HL 仅 ~5–7% 源有 1 个真实候选，polyA 全灭（20 源 test 全部 0）。

## 3. support(32) vs support(256)：尾部质量是否存在的判断

- 实测 support(32) overall = **24.2%**。这在 32 预算下意味着：**76% 的源，生成器 32 池里连一个真实 measured 候选都没有** → 即便 critic 判别力 ρ=1（完美排序），recovery@budget 上限也被钉在 ~0.24 以下（配合 recovery 0.12 实际值）。
- 58% polyA test 全部 support=0；MPRAU/HL 大部分源不可达。
- **结论：尾部（真实 measured 候选）在物理上存在且可采样到**——measured 以每槽 ~0.01 的密度成泊松性地分布在生成池中；同生成器在 256 池下（A4 模拟）support 升至 ~0.78–0.89。**尾部质量存在，但 32 预算严重欠采**。
- 所以 B2/B3 FAIL 的**主瓶颈不是 critic 判别力，而是生成器 32 池的覆盖不足（support 太低）**；critic 判别力（hit@1 仅 ~0.035）是**次**瓶颈。

## 4. TreeG 说明（256 池假设不成立）

- 实测产物 `result_full891.json`（select_k=1）：pool=unguided 28512（=32/源），TreeG recovery=0.00225、hit@1=0.00449；`result_k2.json`（k=2）：recovery=0.00524。均显著劣于 unguided，因 k 小 → shot-count 塌缩。
- **不存在可用的 256 候选/源 TreeG 池**（脚本注释：N>32 的 sample-then-select 是“注册后续项，本次未运行”）。任务背景里的“TreeG 池=256”在磁盘产物上不成立。
- TreeG selections 未持久化、重算需 GPU critic，CPU 批次无法重建逐源 measured-in-pool → 该单元格记为 **N/A（数据缺失，如实标注）**。

## 5. 支持度结论（供 A3/baseline comparison 引用）

1. 生成器 32 池对真实 measured 集合的覆盖 = ~24% source-support，1/31 有效槽位密度。
2. guidance 不改变覆盖（support 恒 0.2424，measured-in-pool 几乎不变）。
3. 尾部存在：放大池（256）+ 改善生成器覆盖是恢复 0.35 的杠杆；critic ρ 只影响“已在池中时能否排到 top”，无法单独拉回 recovery@budget。