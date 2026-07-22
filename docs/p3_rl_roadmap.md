# P3 RL Roadmap: From Pilot to Paper-Ready

**Date**: 2026-07-22
**Status**: PLANNING (awaiting user approval)
**Depends on**: P2 completion (24/24 PASS), P2-10 Option C 10k checkpoint

---

## Goal

将 RL 从 pilot (DEGRADED, 3-seed, n_groups=2, max_steps=5) 升级为 paper-ready
(full-scale, 10-seed, equal-query comparison, healthy backbone)。

## Priority Order

P3-01 > P3-02 > P3-03 > P3-04 > P3-05 > P3-06 > P3-07 > P3-08

---

## P3-01: Stage A Backbone 健康 (BLOCKER)

**目标**: 产出一个 val_loss 持续下降、grad_norm P99 < 1000 的 trained backbone

**方案** (二选一):
- Option A: Pivot 到 frozen foundation encoder (RNA-FM / RiNALMo / ERNIE-RNA)
  + lightweight adapter, 绕过 Stage A 训练问题
- Option B: 修复 grad clipping (max_norm=1.0) + cosine LR schedule + 100k steps
  full retrain, 1 seed

**验收**: val_loss 在 10k/50k/100k 三点单调下降, grad_norm P99 < 1000,
checkpoint SHA-256 locked

---

## P3-02: Full-Scale GRPO 训练

**目标**: 用 P3-01 backbone 作为 warm-start, 完成 spec-compliant GRPO 训练

**配置**:
- n_groups = 4 (spec)
- max_steps = 256 (spec)
- n_iter = 5000+
- 10 policy seeds
- KL coef = 0.05, entropy coef = 0.01

**验收**: 10-seed paired significance test, improvement CI > 0, family-cluster
bootstrap CI, 所有 seeds 完成 5000+ iters

**依赖**: P3-01

---

## P3-03: 等 Query 公平对比

**目标**: 证明 GRPO 在等 oracle call 预算下优于 beam search 和 SA

**方法**:
- Beam search: beam width = {1, 4, 16}, same oracle calls as GRPO
- Simulated annealing: same oracle calls, temperature schedule
- GRPO: same oracle calls (n_groups * max_steps * n_iter)
- 10-seed paired permutation test, Holm-Bonferroni 校正

**验收**: GRPO mean return > beam best > SA best, p < 0.05, CI 不重叠

**依赖**: P3-02

---

## P3-04: Algorithm Identity 验证

**目标**: 证明 edit-flow CTMC 的 greedy decoding 等价于 true flow marginal

**方法**:
- 在 100 个 tiny MDP instances 上比较 greedy vs exact flow marginal
- 验证 action distribution 一致性 (KL divergence < 1e-6)
- Correctness test: 1000 random states, greedy argmax = true flow argmax

**验收**: KL < 1e-6, argmax 一致率 = 100%, dashboard Algorithm identity NO -> YES

---

## P3-05: Reward-Hacking Resistance

**目标**: 证明 RL policy 没有 gaming the oracle

**方法**:
- 三 oracle 一致性: Oracle #1 vs #3 vs development, Pearson r > 0.7
- OOD reward guard: 在 family_rare subset 上 reward 不 collapse
- Adversarial input test: 注入 adversarial sequences, reward 不异常上升
- KL divergence to reference policy < threshold

**验收**: 三 oracle 一致, OOD 不 collapse, adversarial 不 hack, dashboard
Reward-hacking resistance NO -> YES

---

## P3-06: External SOTA Matched-Budget

**目标**: 在 matched oracle call 预算下与外部 SOTA 公平对比

**方法**:
- Re-run LinearDesign, codonGPT, UTailoR, UTRGAN with matched budget
- 至少 4/6 protocol-aligned
- 10-seed paired test vs MEF

**验收**: >= 4/6 protocol-aligned, 10-seed paired test, dashboard External
SOTA fairness PARTIAL -> YES

---

## P3-07: Cross-Region Synergy 升级 (STRETCH)

**目标**: 从 BORDERLINE 升级到 GO

**方法**:
- 升级 MultiRegionOracle (更大数据, 更强 ensemble)
- 扩大 counterfactual panel (5000 wild-types)
- 用 P3-02 trained policy 替代 random policy

**验收**: Cohen's d > 0.5, p < 0.001, verdict = GO

**依赖**: P3-01, P3-02

---

## P3-08: Wet-Lab Validation (STRETCH)

**目标**: Prospective multi-cargo MPRA 验证

**方法**:
- 使用 P3-02 GRPO policy 设计 1000 sequences
- 5 arms (wild-type / MEF-edited / GEMORNA / LinearDesign / random)
- 5-10 therapeutic proteins, HEK293T + HepG2
- Readouts: MRL, TE, half-life, protein expression

**验收**: MEF-edited arm 在 >= 3/5 cargo 上 TE 显著优于 wild-type
(p < 0.05, paired test)

**依赖**: P3-02, wet-lab 合作伙伴

---

## 硬约束 (继承 P2)

- 不擅自终止任何运行中进程; PID 必须 runtime rediscovered
- 不修改 v1 frozen namespace
- 所有新增训练必须接入 --train-idx/--val-idx/--test-idx 强制合同
- 任何 "improves TE/stability/expression" 必须加 predicted/internal proxy 限定词
- 所有性能主张必须基于 10-seed paired significance test
- 所有新代码必须配套单元测试
