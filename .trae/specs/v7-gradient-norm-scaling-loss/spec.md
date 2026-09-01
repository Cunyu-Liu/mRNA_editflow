# V7：在线梯度范数缩放 loss（架构侧，预注册 Spec）

## Why

V6 screen 主判据已判定 NO-GO（2026-09-01）：v6_full MPRAU pair-mean ρ=0.051，pair 级
bootstrap 95% CI [0.007,0.096]，上界低于 v5 基线 0.103。V6 引入 pair-mean 监督 + 细胞偏移头
+ per-task rank 变换，工程完备（GPU 验证、protected reads=0、8/8 pass），但主判据未过门。

诊断（F8/F15/长尾 + 训练观测）指向 loss 机制层：任务间梯度贡献不平衡 + 中低效应区欠学习。
文献（2025 IEEE 实验分析）表明优化不平衡 ↔ 任务梯度范数强相关，按范数缩放可媲美 grid search。
V7 是 loss 机制级单一候选，与 V6 同 seed 20260907 对照。

## What Changes

在 V6 loss（effective_prediction_objective_v4）之上新增**在线梯度范数缩放**：

- task-homogeneous batch（每 32 行 batch 单一任务）天然提供"每任务梯度范数"的干净估计。
- 每 batch 用 detached prediction 梯度（∂L/∂v̂，32 向量）的 L2 范数作为该任务当前梯度尺度，
  或（实现时二选一，预注册冻结）参数梯度范数；EMA 平滑（系数 α 预注册）。
- 动态权重 multiplier = 几何均值参考 / EMA 范数，使各任务对共享参数的梯度贡献均衡。
- 与既有 task-robust target scaler 协同（scaler 管标签尺度，梯度缩放管梯度贡献）。
- 默认关（=V6 bit 级一致）；开=V7。config 级开关，独立 family。

## 验收门槛（预注册，不可事后改）

- **主判据**：MPRAU pair-mean ρ 的 pair 级 bootstrap 95% CI 下界 > 0（不跨零）且 ρ > V5 基线 0.103；
  同时 ≥ V6 的 0.051 + 统计证据（与 V6 差值 pair 级 bootstrap CI 不跨零）。
- **辅助判据**：细胞偏移头对 HEK293FT vs 其余区分度（AUC/ρ）可计算；天花板归一化 ≥15%→40%+ 目标跟踪。
- **工程**：CUDA/BF16-only、cpu_fallback=false、protected reads=0、per-pass checkpoint 1–8、
  参数 165–175M、exact-HEAD 授权链。
- **对照**：同 seed 20260907、同 V6 数据/架构/其余训练字段，仅 loss 机制变更。

## 与冻结协议的关系

V7 = 独立新 family，照 V5/V6 启动模式（独立 config/auth、复用冻结 preflight 与 bottom-six cache、
不动 V4.0.3/S1/V6 任何 gate）。V7 未过门则记录负结果，进入 V8（LoRA/减块，需协议修订）。

## Removed / 不做

- 不做 ensemble（用户裁定）；不做 SSL；SWA 保持离线分析。
- V7 只选梯度范数缩放一个候选（不并行实现 BiLB4MTL/DB-MTL）。
