# Gate P 前置核验 v1（TEST 揭盲就绪状态清单，2026-09-28）

> **定位**：Gate P = 打开 protected TEST（18,292 行 withheld）的决策门。项目多处纪律文本以「TEST 在 Gate P 拍板前不碰」表述（主合同 V3.3.2 `提示词/mrna 数据gate转向后的合同.md` 为该门权威出处；W0 `docs/paper/route2_w_ladder_amendment_v1.md` 与 journal 有引用）。
> **本文件性质**：只记录**可核验的就绪状态**，供用户/PI 拍板用；**不改变任何门槛、不触碰 TEST、不构成揭盲决定**。全部证据 09-28 实查。

## 1. 就绪状态核验表（全部为实查读数）

| # | 核验项 | 状态 | 证据 locator |
|---|---|---|---|
| 1 | **protected reads = 0** | **PASS** | `benchmark_v2/leaderboard_matrix_v2/matrix_v2_results.json → protected_test_reads: 0`；矩阵行定义「TEST split untouched」；三新行 row_definition 全部限定 VALIDATION/NEW_EVAL_ROW |
| 2 | **矩阵/榜单完整性** | **PASS** | matrix_v2_results.json：65/65 单元 `status=OK`；5 新族单测 `family_unit_tests.json` 全 PASS（gemorna/hydrarna/lamar/insight/stcnet 五项逐一 PASS，09-28 复核）；新行 M1/M6/S1 行定义 + 泄漏审计齐备 |
| 3 | **在途臂终态前置** | **未满足（进行中）** | M1 干预臂（四门 G1-G4）+ polyA 3-seed（P0-1）09-28 在途；两者终态并收割后，冻结证据面才完整 |
| 4 | **预注册文件冻结齐备** | **PASS** | W0 `docs/paper/` 14 份 prereg/amendment（w_ladder / d16c / erk v1+v2 / polya_3seed / m1_intervention / delta_density / oracle / first_order ×2 / density_sensitivity / mprau_matched_ft / benchmark_v2_matrix_row / pivot）+ comb 机制预注册 v1 在 v8_stage1 worktree `docs/paper/route2_setflow_comb_mechanism_prereg_v1.md` |
| 5 | **统计完备件** | **PASS** | `analysis_baseline_holm_20260909/`（Holm 家族）+ `analysis_baseline_power_20260909/`（power/MDE）+ `analysis_task11_loso_lite_20260909/`（LOSO-lite 协议 B） |
| 6 | **R12 协议公平性** | 写作期收尾 | 外部行预算对齐逐行声明（骨架附 A 标记 OPEN） |
| 7 | **R13 rights + payload** | **未满足（用户侧）** | 评审说明文档在（`route2_v332_study_rights_accountable_human_review_instructions_v1.md`）；14/14 研究**具名 owner 缺位**；0/14 payload 公开授权 |
| 8 | **R2 LOSO 表** | draft 就绪 | 已入骨架素材映射（Table 5，协议 B 冻结） |

## 2. 需用户/PI 拍板的三项（不由执行侧自决）

1. **是否在主结果定稿前打开 TEST**：TEST 为一次性读取（18,292 行），揭盲后不得再用于调参/选型（history-dependent decisions prohibited）；建议顺序 = 全部冻结行齐备（含 M1 四门 + 3-seed）→ 全文主结果定稿 → 一次性揭盲复核。
2. **R13 rights review owner 指派**：指定具名 owner 并推进 14 研究评审；揭盲与投稿时间表依赖该完成时点。
3. **揭盲执行的会话与记录形式**：指定执行会话 + 落盘 schema（一次性读取 receipt + 结果 append-only）。

## 3. 当前建议的 Gate P 前置顺序（执行侧视角）

```
M1 四门收割（今日） → polyA 3-seed 收割（今日） → 全文初稿（本周） 
→ R12/R13 收尾 → 【用户拍板 Gate P】 → TEST 一次性揭盲 → 主结果终稿
```

## 4. 备注

- 本核验**不含**任何 TEST 统计信息（分布摘要亦未读取）。
- 若用户决定提前揭盲，前置第 3/7 项即为风险点（冻结证据面不完整 + 权利状态未清）。
- 后续状态变化（M1/3-seed 终态）由 `check_and_harvest_week.sh` 与定时巡检自动记录，本文件在每项状态变化后按 append-only 方式更新（v1 → v1.x）。
## 6. v1.2 增补（2026-10-06 · 用户拍板两项落档）

- **【用户拍板 2026-10-06】Gate P 揭盲时点 = 全文 v2 完成后**（对应冲刺计划 W3 = 10-20~10-26；执行顺序不变：全文 v2（W2 参考文献/图表/摘要）完成 → 一次性揭盲 18,292 行 → 主结果终稿回填）。protected TEST reads 在揭盲前保持 = 0（本日复核 matrix_v2_results.json 仍为 0）。
- **【用户拍板 2026-10-06】rights review owner = TRAE 执行侧 agent（证据层）**。当日完成 15 研究回源审计（`data_rights_audit_v1.md`，commit 2d7abc5d）+ 评审 CSV 填写（`..._reviewed_20261006.csv`，commit c2cb7355）：15/15 analysis&publication 允许；15/15 payload 再分发保守 NOT_AUTHORIZED（E-MTAB-10902 HOLD 待 BioStudies license 字段）；Data Availability 投稿草稿成文（§2）。**第 7 项状态更新：REVIEWED（证据层完成）——最终发布决策仍归用户/PI**；第 3 项（在途臂）已于 09-29 全部终态，本项 PASS。
- **当前 Gate P 就绪度**：第 1/2/4/5/8 项 PASS；第 3 项 PASS（全部臂终态）；第 6 项 R12 写作期收尾（W1-W2）；第 7 项 = 证据层完成（发布决策待用户）。**唯一硬前置 = 全文 v2（W2 里程碑）**。
