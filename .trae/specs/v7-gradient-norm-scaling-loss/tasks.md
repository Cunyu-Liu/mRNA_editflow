# V7 任务分解（梯度范数缩放 loss）

## Task 1: 实现在线梯度范数缩放（loss 机制级）
- [ ] 新增 EMA 梯度范数估计器（每任务，α 预注册，冻结于 config）
- [ ] 在训练循环中按 task-homogeneous batch 更新估计器并计算动态 multiplier
- [ ] 接入 effective_prediction_objective_v4 的权重合成（开关关=与 V6 输出 bit 级一致）
- [ ] run_summary 记录梯度范数估计/EMA 曲线（可审计）

## Task 2: 单测
- [ ] 开关关 → 与 V6 loss 输出逐值一致（回归）
- [ ] 开关开 → 不同任务梯度范数趋于均衡（几何均值中心化）
- [ ] EMA 平滑行为、异常范数（0/非有限）防护
- [ ] 既有 36 项 V6/V4 测试不回归

## Task 3: V7 预注册冻结（config/auth）
- [ ] V7 screen config（独立，seed 20260907，同 V6 其余字段，开梯度缩放）
- [ ] A100 sync audit（新 HEAD）→ 授权链重建

## Task 4: V7 首训（H3 重跑完成后，GPU1/2/4 空闲）
- [ ] 8 pass / 22,416 updates / BF16 / per-pass checkpoint
- [ ] 主判据核算：MPRAU pair-mean ρ + pair 级 bootstrap CI（vs V5 0.103 与 V6 0.051）

## Task 5: 判定与归档
- [ ] V7 screen gate 判定（按 spec 预注册门槛）
- [ ] 过门 → W2-b 3 seeds confirmation；未过门 → 记录负结果，进 V8 协议修订讨论
