# D5（MPRAU 目标带）Amendment v1 —— 以 s_mprau_in 0.1351 为新参照重估目标带

> **状态：AMENDMENT v1（2026-09-07）**。触发条件满足：V8 Stage 2（含 CMS 首臂数据）出齐。
> 修订走 amendment 流程，**旧 40% 目标不删，留痕**。预注册纪律：本文件为正式修订，
> 判定门变更以本文为准。

## 1. 原目标（合同口径，保留留痕）

- 原 D5 目标带：**MPRAU 40% 完成度**（= 0.40 × 天花板 0.683 ≈ 绝对 Spearman 0.27）。
- 历史沿革：D5 在 V5 critic 时代（MPRAU 0.1025 = 15% 完成度）提出，为"先验假设"目标；
  V8 攻坚线启动时用户拍板"等数据"（V8 Stage 2 + CMS 出齐后 amendment 修订）。

## 2. 修订依据（证据链，全部 VALIDATION 口径 FINAL-EPOCH-FIXED）

| 模型 / 方法 | MPRAU pair-mean ρ | 天花板完成度 | 与 V5 的 paired bootstrap |
|---|---|---|---|
| V5 critic（多任务）| 0.1025 | 15.0% | —（参照）|
| Saluki frozen（外部 CNN+GRU）| 0.1205 | 17.6% | CI 含 V5（点估计领先）|
| V6/V7 loss 机制六臂 | 0.0510–0.0883 | — | 全负，未过门 |
| Route A 直调 mRNABERT | −0.091 | — | 显著负 |
| **s_mprau_in（V8-S + ENCSR854RUF 域内专才，5-seed ensemble）** | **0.1351** | **19.8%** | Δ+0.0325，CI [−0.023,+0.088] 跨零（边际）|
| h_bench9（均衡多任务适配）| 0.0556 | — | 未过门 |
| CMS 外部先验三臂 | 0.033/0.039/−0.007 | — | 全 FAIL |

关键事实：
1. **s_mprau_in 是项目内首个在 MPRAU pair-mean 上持续超过 V5 的模型**（5/5 seed > 0.1025：
   0.153/0.114/0.112/0.108/0.110），5-seed ensemble 0.1351。
2. 显著性缺口（CI 跨零）源于**数据体制检验力上限**：MPRAU VALIDATION 仅 2,008 变体 +
   per-cell 噪声主导 CI 宽度（3-seed 与 5-seed CI 几乎同宽，更多 seed 不收敛）。
3. 40% 完成度（0.27）在当前数据体制下**无实测证据支撑**——最强 in-house 模型 0.1351 已达
   现有验证集可稳定分辨的上界附近。

## 3. Amendment 内容（正式修订）

**目标带重新表述（旧 40% 保留为远期延伸目标）：**

1. **近期验收目标（修订后）**：MPRAU pair-mean **> 0.1351（s_mprau_in 5-seed ensemble 参照）
   且 paired bootstrap（2,000 iters, seed 20260816）ΔCI 不跨零**——即"在现有参照上再进一步
   且统计显著"。这是数据体制内可检验、可验收的目标。
2. **远期延伸目标（原 40% 保留）**：0.27（40% 完成度）不删，作为数据扩充（更大验证池/
   TEST 开启需 Gate P 拍板）后的远期目标，当前不设截止。
3. **检验力条款**：对 ~0.03 级 delta 差异，2,008 变体验证集检验力不足（<2,000 records 需求
   已在 MRL 侧同类披露）；任何"显著超越 0.1351"的裁决若 CI 跨零，如实记录为"方向性提升、
   显著性受限"，不作翻案，不事后改 CI。

## 4. 后继（预注册）

- Stage 3（V8-S 专才接 SetFlow B2）终态后，若 critic 攻坚以生成质量结算出正增量，MPRAU
  目标带再按生成线口径复核。
- LOSO-lite（用户拍板：结果 SOTA 后再做）将提供拼表更大验证池，可缓解检验力限制。
- 本 amendment 生效即日；SPECS_BASELINE_LEADERBOARD 榜单 MPRAU 行以本文件为 canonical
  参照（s_mprau_in 0.1351 入 in-house 行，标注"CI 跨零，边际"）。
