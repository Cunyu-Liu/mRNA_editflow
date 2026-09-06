# V8 Stage 2 多任务适配预注册（DRAFT v1，2026-09-06）

> **状态：DRAFT**——Stage 1 全量判定（含 polyA 非破坏门，等 v8p 基线终态）完成后冻结。
> 依据：SPECS_CRITIC_V6/spec.md V8 攻坚线 Stage 2；Stage 1 双臂终态（09-06）：
> S 臂 MRL 0.3078 / polyA 0.1122；H 臂 MRL 0.2876 / polyA 0.5262。MRL 裁决 = S（Δ+0.0202）。

## 1. 触发与架构选择

- 触发：Stage 1 全量判定（adjudication_v8_stage1.json，polyA 门 RESOLVED）后 24h 内冻结本预注册。
- **主臂（待 Stage 1 判定冻结）**：按预注册规则 MRL 裁决选 S；但 polyA 域 H 完胜（+0.414）为次级证据——冻结时须专项裁定：若 polyA 非破坏门在 S 上 FAIL 而 H 上 PASS，按 amendment 将主臂改为 H（或 S+H 双适配臂），MRL 0.02 容忍带条款不覆盖跨域架构选择（修订走 amendment，留痕）。
- 对照：zero-shot（Stage 1 权重直接 9 任务评估）/ V5 复跑（seed 20260907 既有数字）/ 适配臂。

## 2. 训练设计（照 spec）

- 数据级任务均衡采样（loss 级六臂全负后的剩余 lever；V6 λ 教训：聚合越差、条件化才对）。
- **细胞条件输入保留原始每细胞标签**（不聚合）。
- 适配方式对照：LoRA 臂 + 全参臂（参考 Phase 0 臂 B：冻结 embedding+轻头优于继续训 LoRA——臂 B 式配方作第三对照）。
- zero-shot 对照臂必设（Step-2 教训：好 init + 薄任务数据全参适配会过拟合）。
- MPRAU：用 V8 权重 + ENCSR854RUF（TRAIN 55,704 / VALIDATION 2,008 变体）适配；若适配仍负，CMS array（等用户数据）注入为注册后继。

## 3. 判定门（预注册后不可事后改）

- MRL ≥0.28（Stage 1 zero-shot 已 0.3078，适配不得回退）
- polyA ≥0.80 不劣化（V5 0.8219 参照；适配臂目标）
- **MPRAU >0.1025 CI 不跨零**（V5 参照，主判据）
- TE 族 ≥内靶（global_scaled 0.1317）
- macro 显著升 vs 0.167（paired bootstrap，seed 20260816）

## 4. 预算对账（冻结时填）

- 9 任务 TRAIN 池行数实测；均衡采样每 epoch 步数；对照臂预算声明。

## 5. 已知限制

- Stage 1 双臂仅 2 epochs（106,812 步），联合库曝光 MRL 5.04 遍/polyA 1.25 遍（预算不等价条款 §6 生效）；
- polyA 单域基线（v8p）epochs 6 是 polyA 门参照，与联合臂不同预算（已注册口径）。
