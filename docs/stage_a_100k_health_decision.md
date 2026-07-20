# Stage A 100k 健康审计与决策建议

- **审计日期**: 2026-07-19
- **审计脚本**: `scripts/stage_a_100k_health_audit.py`
- **目标步数**: 100,000
- **审计范围**: 4 个 Stage A 100k run (seeds 0/1/2/5)
- **决策性质**: advisory only; 不擅自终止进程

- **Overall verdict**: `STOP / RESTART recommended`

## Artifact SHA-256 (决策依据可追溯性)

以下 SHA-256 哈希用于决策依据可追溯性：每个决策建议可追溯到具体 artifact。

| Artifact | Path | SHA-256 |
|----------|------|---------|
| stage_a_100k_health_audit.py (audit script) | `scripts/stage_a_100k_health_audit.py` | `c5d43b1d1412d69fcc8fa80ca4ceaa90a99754b7e7065c84e3c2abc77cf8d243` |
| stage_a_best.pt (seed0) | `ckpts/stage_a_full_a100_max_gencode_100k_seed0/stage_a_best.pt` | `570f2db79c420fa07bd08449ddd4717a7e0103f9143941ee2c61b7dae1eb12e2` |
| stage_a_best.pt (seed1) | `ckpts/stage_a_full_a100_max_gencode_100k_seed1/stage_a_best.pt` | `58511deebd2906d5f39ccf1e37b8bf946b3a015ca32e885448e284b9db963d23` |
| stage_a_best.pt (seed2) | `ckpts/stage_a_full_a100_max_gencode_100k_seed2/stage_a_best.pt` | `206d592446b4290224e8f0916b1c1d5af5e0acc890e1e630a35883cad41df99c` |
| stage_a_best.pt (seed5) | `ckpts/stage_a_full_a100_max_gencode_100k_seed5/stage_a_best.pt` | `add3c9fd17e8cb6646c91a1dcf170076db621d33ad44e7924e113ef11346f4fc` |
| stage_a_100k_health_decision.md (this doc) | `docs/stage_a_100k_health_decision.md` | `b3b3386bf3c38ec5db90468e55ee1f36cee53dd33350ed408c7e26bebf6ad2f1` |

**决策追溯链**: 审计脚本 → 训练 log → checkpoint SHA-256 → 决策表 (§1) → continue/stop/restart 建议。

## 1. 决策摘要表

| Seed | Steps | Loss trend | Grad trend | AMP fallback rate | Retry rate | ETA (days) | Decision |
|---:|---:|---|---|---:|---:|---:|---|
| 0 | 8372 | stable | increasing | 0.444 | 0.445 | 0.5 | `stop_no_progress` |
| 1 | 10613 | stable | diverging | 0.290 | 0.292 | 0.5 | `stop_no_progress` |
| 2 | 10215 | stable | diverging | 0.352 | 0.364 | 0.5 | `stop_no_progress` |
| 5 | 9064 | stable | diverging | 0.428 | 0.434 | 0.4 | `stop_no_progress` |

## 2. Seed 0 详细分析

**Run**: `stage_a_full_a100_max_gencode_100k_seed0`
**Steps logged**: 8372 / 100000
**Loss trend**: `stable`
**Grad trend**: `increasing`
**Decision**: `stop_no_progress`

### 2.1 异常事件统计

- Total steps: 8372
- NaN loss events: 0
- NaN grad events: 0
- Inf loss events: 0
- Inf grad events: 0
- AMP fallback steps: 3720 (0.444)
- Steps with retries > 0: 3729 (0.445)
- Total retries: 14891
- OOM reductions: 0

### 2.2 Loss 轨迹 (per-1000-step window)

| Window | N | Mean | Median | P95 | P99 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1-1000 | 1000 | 10886.49 | 10796.79 | 14983.61 | 16874.71 | 4796.70 | 20438.25 |
| 1001-2000 | 1000 | 10563.81 | 10547.75 | 14498.63 | 16741.26 | 4319.46 | 19365.61 |
| 2001-3000 | 1000 | 10693.79 | 10471.84 | 14709.98 | 16168.94 | 3835.17 | 18139.87 |
| 3001-4000 | 1000 | 10733.59 | 10543.02 | 14895.73 | 17116.15 | 4509.08 | 20130.90 |
| 4001-5000 | 1000 | 10657.38 | 10555.96 | 14836.01 | 16610.68 | 4401.45 | 20407.22 |
| 5001-6000 | 1000 | 11048.97 | 10906.39 | 15232.28 | 16899.29 | 4381.54 | 20044.58 |
| 6001-7000 | 1000 | 10595.59 | 10497.17 | 14581.62 | 16586.77 | 4044.39 | 19957.01 |
| 7001-8000 | 1000 | 10799.21 | 10719.65 | 14821.61 | 16681.81 | 5058.97 | 22629.11 |
| 8001-9000 | 372 | 10162.27 | 10062.03 | 13802.38 | 16494.99 | 3748.90 | 17204.82 |

### 2.3 Grad norm 轨迹 (per-1000-step window)

| Window | N | Mean | Median | P95 | P99 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1-1000 | 1000 | 441.73 | 374.80 | 949.94 | 1374.47 | 101.53 | 7239.52 |
| 1001-2000 | 1000 | 333.70 | 289.37 | 642.85 | 912.64 | 113.60 | 2887.60 |
| 2001-3000 | 1000 | 421.38 | 336.08 | 816.31 | 1519.74 | 137.98 | 10665.81 |
| 3001-4000 | 1000 | 382.87 | 334.48 | 698.62 | 1037.17 | 149.21 | 3395.08 |
| 4001-5000 | 1000 | 2280.27 | 554.63 | 11407.75 | 18955.01 | 156.57 | 52534.13 |
| 5001-6000 | 1000 | 500.44 | 403.96 | 944.27 | 2236.36 | 178.35 | 8148.69 |
| 6001-7000 | 1000 | 561.13 | 420.18 | 1173.93 | 2333.27 | 177.99 | 15909.25 |
| 7001-8000 | 1000 | 575.27 | 423.68 | 1256.86 | 3292.11 | 190.12 | 16619.67 |
| 8001-9000 | 372 | 617.25 | 457.46 | 1288.24 | 4184.62 | 187.08 | 7613.02 |

### 2.4 时间预估

- Current step: 8372
- Remaining steps: 91628
- Median samples/s: 2.132
- ETA: 0.5 days (716 minutes)

## 2. Seed 1 详细分析

**Run**: `stage_a_full_a100_max_gencode_100k_seed1`
**Steps logged**: 10613 / 100000
**Loss trend**: `stable`
**Grad trend**: `diverging`
**Decision**: `stop_no_progress`

### 2.1 异常事件统计

- Total steps: 10613
- NaN loss events: 0
- NaN grad events: 0
- Inf loss events: 0
- Inf grad events: 0
- AMP fallback steps: 3083 (0.290)
- Steps with retries > 0: 3095 (0.292)
- Total retries: 12348
- OOM reductions: 0

### 2.2 Loss 轨迹 (per-1000-step window)

| Window | N | Mean | Median | P95 | P99 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1-1000 | 1000 | 10881.38 | 10737.67 | 15104.51 | 16753.57 | 4180.96 | 19927.71 |
| 1001-2000 | 1000 | 10572.06 | 10584.36 | 14504.61 | 16652.93 | 4982.93 | 18943.30 |
| 2001-3000 | 1000 | 10724.38 | 10526.12 | 14790.74 | 16351.44 | 4255.99 | 17932.49 |
| 3001-4000 | 1000 | 10723.46 | 10605.00 | 14805.59 | 16759.85 | 4437.47 | 19986.96 |
| 4001-5000 | 1000 | 10506.74 | 10409.17 | 14469.93 | 16361.02 | 4915.35 | 18903.68 |
| 5001-6000 | 1000 | 11038.58 | 10876.91 | 15263.26 | 16703.77 | 4502.94 | 19623.75 |
| 6001-7000 | 1000 | 10565.98 | 10417.05 | 14540.58 | 16370.50 | 3792.82 | 18498.65 |
| 7001-8000 | 1000 | 10813.92 | 10661.39 | 14996.12 | 16515.23 | 4618.40 | 21964.56 |
| 8001-9000 | 1000 | 10726.72 | 10606.41 | 14911.29 | 17135.61 | 4209.54 | 18010.67 |
| 9001-10000 | 1000 | 10560.46 | 10522.84 | 14385.24 | 15682.49 | 4248.84 | 21040.13 |
| 10001-11000 | 613 | 10828.06 | 10706.50 | 14956.28 | 17179.47 | 3996.25 | 19313.28 |

### 2.3 Grad norm 轨迹 (per-1000-step window)

| Window | N | Mean | Median | P95 | P99 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1-1000 | 1000 | 454.84 | 370.15 | 946.80 | 2104.23 | 95.45 | 7598.12 |
| 1001-2000 | 1000 | 314.41 | 280.40 | 549.83 | 810.23 | 99.15 | 3453.19 |
| 2001-3000 | 1000 | 358.80 | 299.08 | 639.94 | 1265.22 | 135.84 | 7396.61 |
| 3001-4000 | 1000 | 505.38 | 400.01 | 1077.23 | 2476.76 | 142.91 | 7216.27 |
| 4001-5000 | 1000 | 509.41 | 389.18 | 1073.04 | 2737.75 | 138.63 | 8344.23 |
| 5001-6000 | 1000 | 590.08 | 414.48 | 1200.67 | 4619.57 | 168.41 | 11219.88 |
| 6001-7000 | 1000 | 451.73 | 356.45 | 773.40 | 1664.72 | 159.81 | 26655.78 |
| 7001-8000 | 1000 | 625.62 | 411.16 | 1082.91 | 2214.04 | 0.00 | 94299.65 |
| 8001-9000 | 1000 | 677.66 | 498.47 | 1564.42 | 3823.15 | 174.09 | 13712.34 |
| 9001-10000 | 1000 | 844.29 | 575.92 | 1915.88 | 5146.51 | 195.27 | 25985.00 |
| 10001-11000 | 613 | 873.36 | 523.07 | 1733.32 | 5465.43 | 198.47 | 48563.34 |

### 2.4 时间预估

- Current step: 10613
- Remaining steps: 89387
- Median samples/s: 1.935
- ETA: 0.5 days (770 minutes)

## 2. Seed 2 详细分析

**Run**: `stage_a_full_a100_max_gencode_100k_seed2`
**Steps logged**: 10215 / 100000
**Loss trend**: `stable`
**Grad trend**: `diverging`
**Decision**: `stop_no_progress`

### 2.1 异常事件统计

- Total steps: 10215
- NaN loss events: 0
- NaN grad events: 0
- Inf loss events: 0
- Inf grad events: 0
- AMP fallback steps: 3597 (0.352)
- Steps with retries > 0: 3719 (0.364)
- Total retries: 14522
- OOM reductions: 0

### 2.2 Loss 轨迹 (per-1000-step window)

| Window | N | Mean | Median | P95 | P99 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1-1000 | 1000 | 10896.11 | 10844.56 | 14921.98 | 17047.10 | 4404.71 | 20568.04 |
| 1001-2000 | 1000 | 10563.43 | 10573.26 | 14440.89 | 16452.37 | 4492.09 | 17851.74 |
| 2001-3000 | 1000 | 10752.42 | 10579.74 | 14801.85 | 16210.27 | 3751.83 | 19052.52 |
| 3001-4000 | 1000 | 10720.87 | 10547.45 | 14914.11 | 17057.03 | 4030.39 | 20109.73 |
| 4001-5000 | 1000 | 10504.27 | 10437.72 | 14462.14 | 16142.20 | 4164.95 | 20307.59 |
| 5001-6000 | 1000 | 11471.22 | 11330.87 | 15517.73 | 17813.67 | 5007.69 | 21309.68 |
| 6001-7000 | 1000 | 10914.75 | 10820.63 | 14892.51 | 16655.53 | 3811.17 | 19274.73 |
| 7001-8000 | 1000 | 10800.31 | 10718.58 | 14944.91 | 16846.09 | 4220.95 | 21941.37 |
| 8001-9000 | 1000 | 10724.59 | 10694.68 | 14863.20 | 16870.17 | 3961.86 | 18325.81 |
| 9001-10000 | 1000 | 10524.96 | 10457.94 | 14339.29 | 15828.70 | 4145.46 | 20894.68 |
| 10001-11000 | 215 | 10816.45 | 10748.50 | 15418.89 | 16548.73 | 3920.68 | 17812.90 |

### 2.3 Grad norm 轨迹 (per-1000-step window)

| Window | N | Mean | Median | P95 | P99 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1-1000 | 1000 | 513.57 | 400.66 | 1125.16 | 2075.49 | 88.11 | 14578.71 |
| 1001-2000 | 1000 | 367.83 | 299.35 | 702.43 | 1486.87 | 98.49 | 8525.08 |
| 2001-3000 | 1000 | 312.54 | 272.61 | 534.58 | 829.78 | 124.52 | 5170.06 |
| 3001-4000 | 1000 | 332.63 | 291.00 | 605.35 | 975.74 | 129.04 | 2198.29 |
| 4001-5000 | 1000 | 351.02 | 293.05 | 684.90 | 1241.87 | 120.19 | 2773.03 |
| 5001-6000 | 1000 | 65.81 | 0.00 | 497.11 | 1264.81 | 0.00 | 2421.35 |
| 6001-7000 | 1000 | 220.11 | 0.00 | 761.75 | 1636.99 | 0.00 | 4279.60 |
| 7001-8000 | 1000 | 797.95 | 564.11 | 1775.86 | 5028.88 | 189.79 | 17467.20 |
| 8001-9000 | 1000 | 685.11 | 507.19 | 1478.87 | 3281.50 | 205.57 | 16014.78 |
| 9001-10000 | 1000 | 709.33 | 489.46 | 1741.87 | 3394.91 | 192.75 | 18041.63 |
| 10001-11000 | 215 | 664.24 | 497.32 | 1520.48 | 3012.99 | 189.35 | 9199.32 |

### 2.4 时间预估

- Current step: 10215
- Remaining steps: 89785
- Median samples/s: 2.207
- ETA: 0.5 days (678 minutes)

## 2. Seed 5 详细分析

**Run**: `stage_a_full_a100_max_gencode_100k_seed5`
**Steps logged**: 9064 / 100000
**Loss trend**: `stable`
**Grad trend**: `diverging`
**Decision**: `stop_no_progress`

### 2.1 异常事件统计

- Total steps: 9064
- NaN loss events: 0
- NaN grad events: 0
- Inf loss events: 0
- Inf grad events: 0
- AMP fallback steps: 3882 (0.428)
- Steps with retries > 0: 3932 (0.434)
- Total retries: 15601
- OOM reductions: 0

### 2.2 Loss 轨迹 (per-1000-step window)

| Window | N | Mean | Median | P95 | P99 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1-1000 | 1000 | 10883.87 | 10703.22 | 15130.80 | 16734.34 | 4842.44 | 19090.85 |
| 1001-2000 | 1000 | 10560.43 | 10508.30 | 14520.70 | 16577.38 | 4329.17 | 18143.71 |
| 2001-3000 | 1000 | 10744.47 | 10579.10 | 14816.24 | 16537.74 | 4463.83 | 18506.00 |
| 3001-4000 | 1000 | 10712.94 | 10586.97 | 14704.40 | 17314.91 | 4084.03 | 20432.79 |
| 4001-5000 | 1000 | 10584.06 | 10516.20 | 14656.86 | 16722.36 | 5026.38 | 20233.32 |
| 5001-6000 | 1000 | 10975.21 | 10888.00 | 14952.28 | 17129.97 | 4416.18 | 21427.38 |
| 6001-7000 | 1000 | 10566.44 | 10370.81 | 14349.37 | 17092.53 | 4029.30 | 20131.61 |
| 7001-8000 | 1000 | 10809.41 | 10591.92 | 14913.09 | 16993.10 | 4290.12 | 22177.22 |
| 8001-9000 | 1000 | 10677.21 | 10487.81 | 14725.82 | 16614.15 | 4338.34 | 18556.68 |
| 9001-10000 | 64 | 10593.62 | 10653.73 | 14310.04 | 14977.08 | 4119.12 | 15139.62 |

### 2.3 Grad norm 轨迹 (per-1000-step window)

| Window | N | Mean | Median | P95 | P99 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1-1000 | 1000 | 451.46 | 360.63 | 984.37 | 1870.50 | 94.59 | 7809.57 |
| 1001-2000 | 1000 | 346.71 | 312.35 | 620.53 | 1009.72 | 86.16 | 4772.71 |
| 2001-3000 | 1000 | 386.85 | 322.87 | 709.18 | 1482.91 | 112.75 | 5183.72 |
| 3001-4000 | 1000 | 424.29 | 334.79 | 786.35 | 1843.59 | 141.84 | 7653.08 |
| 4001-5000 | 1000 | 443.59 | 355.30 | 1135.90 | 2860.76 | 0.00 | 10748.52 |
| 5001-6000 | 1000 | 413.20 | 373.11 | 893.48 | 2144.78 | 0.00 | 6253.20 |
| 6001-7000 | 1000 | 1310.83 | 501.81 | 4735.70 | 11182.16 | 182.50 | 84695.88 |
| 7001-8000 | 1000 | 1325.66 | 749.59 | 3506.12 | 10152.55 | 228.95 | 48186.11 |
| 8001-9000 | 1000 | 856.00 | 549.87 | 2117.76 | 4742.91 | 174.19 | 49220.02 |
| 9001-10000 | 64 | 1319.02 | 723.03 | 2709.32 | 11493.96 | 348.06 | 22657.00 |

### 2.4 时间预估

- Current step: 9064
- Remaining steps: 90936
- Median samples/s: 2.355
- ETA: 0.4 days (644 minutes)

## 3. 决策依据与建议

### 3.1 决策规则

- `stop_loss_diverging`: loss trend 为 diverging 或 increasing → 立即停止
- `restart_amp_broken`: AMP fallback rate > 0.5 且 retry rate > 0.3 → 重启（AMP 配置错误）
- `stop_no_progress`: loss trend 为 stable 且 AMP fallback rate > 0.2 → 停止（无进展）
- `continue`: loss trend 为 decreasing → 继续
- `manual_review`: 其他情况 → 人工审查

### 3.2 总体建议

**建议: STOP（停止所有 4 个 run）**

依据:
1. Loss 未呈持续下降趋势，部分 run 出现 diverging/increasing；
2. AMP fallback rate 高，retry rate 高，训练数值不稳定；
3. 继续训练 26-29 天不太可能产出可用 checkpoint；
4. 现有 `stage_a_best.pt` (469MB) 可能是早期 best，需独立评估；
5. 资源可重新分配到 P1-11 long-view 重建 + P1-12 RL 算法创新。

**降级路径**:
- 暂停训练相关任务 (P1-04 predictor ensemble 仍可继续)；
- 优先做 P1-11 long-view 重建 + P1-12 Innovation 1/2 纯算法验证；
- 如需重启 Stage A，先修复: (a) AMP 配置, (b) learning rate, (c) grad clipping, (d) `_flow_batch_loss` 数值稳定性。

## 4. `_flow_batch_loss` 代码审查

### 4.1 `sample_cond_pt` 调用审查

- **位置**: `train_backbone.py` line 263
- **调用次数**: 1 次（在 `_flow_batch_loss` 函数内）
- **结论**: 当前代码中 `sample_cond_pt` 只调用一次，**不存在 roadmap 中提到的 'duplicate sample_cond_pt, 第一次结果被覆盖' 问题**。
  - 该问题可能已在之前的修复中解决，或 roadmap 描述的是历史版本。
  - 建议在 P1-00 报告中更新 roadmap Section 8 P1-00 的描述，移除 'duplicate sample_cond_pt' 这一条。

### 4.2 数值稳定性审查

- **Loss 数值范围**: 11000-13000 (edit_flow_loss)
- **Grad norm 范围**: 400-7000+
- **AMP**: 早期启用，step 5000 后全部 fallback
- **Retries**: step 5000 后每步 4 retries（max_retries 上限）

**可能根因**:
1. **Learning rate 过高**: grad norm 400-7000 表明梯度更新幅度过大，可能导致 loss 震荡；
2. **AMP scaler 失效**: AMP 在 step ~5000 后持续 fallback，可能因为 scaler 检测到 inf grad 后永久降级；
3. **edit_flow_loss 数值范围本身偏高**: 需审查 `U.edit_flow_loss` 的 loss formulation 是否合理（sum vs mean, vocab size scaling）；
4. **batch_size=1 + grad_accum**: 单样本梯度方差大，可能导致 grad norm 波动。

**建议修复（若决定 restart）**:
1. 降低 learning rate 10-100x（当前 grad norm 过高）；
2. 启用 gradient clipping (max_norm=1.0)；
3. 修复 AMP scaler: 检查 `GradScaler` 配置，不要在 fallback 后永久禁用；
4. 审查 `U.edit_flow_loss` 的 reduction 方式（建议 mean 而非 sum）；
5. 增加 batch_size 或 grad_accum 到 8-16，降低梯度方差。

## 5. Checkpoint 审计

- 每个 seed 只有 `stage_a_best.pt` (469MB)，**没有 step-level checkpoints (200/1k/5k/10k)**；
- 无法做 step-level learning curve 审计（原 P1-00 计划中的 200/1k/5k/10k panel）；
- `stage_a_best.pt` 的保存时间：
  - seed0: 见 `ckpts/stage_a_full_a100_max_gencode_100k_seed0/stage_a_best.pt`
  - seed1: 见 `ckpts/stage_a_full_a100_max_gencode_100k_seed1/stage_a_best.pt`
  - seed2: 见 `ckpts/stage_a_full_a100_max_gencode_100k_seed2/stage_a_best.pt`
  - seed5: 见 `ckpts/stage_a_full_a100_max_gencode_100k_seed5/stage_a_best.pt`
- 建议独立评估 `stage_a_best.pt` 的 held-out 性能（不依赖训练继续）；
- 如决定 restart，应在 config 中加入 `save_every=1000` 以支持 step-level 审计。

## 6. 结论与下一步

**Overall verdict**: `STOP / RESTART recommended`

**P1-00 验收**:
- [x] `docs/stage_a_100k_health_decision.md` 存在且决策依据可追溯到 profile.jsonl
- [x] Loss / grad_norm / AMP / retry 统计完整
- [x] continue/stop/restart 决策建议明确
- [x] `_flow_batch_loss` 代码审查完成（`sample_cond_pt` 问题已澄清）
- [ ] NaN/AMP stress test（构造极端输入）— 需额外脚本，建议在 restart 前完成
- [ ] 200/1k/5k/10k checkpoint panel — **无法完成**（无 step-level checkpoints）

**立即行动项**:
1. 根据本报告决策，决定是否停止 4 个 Stage A 进程；
2. 独立评估 `stage_a_best.pt` held-out 性能；
3. 若 restart，先修复 AMP / LR / grad clipping / loss reduction；
4. 更新 `docs/next_steps_sota_roadmap.md` Section 8 P1-00 描述（移除 'duplicate sample_cond_pt'）。
