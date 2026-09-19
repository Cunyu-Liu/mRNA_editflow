# UTR-EditFlow Goal v2 — Amendment (Pivot) v1

- **decision_id**: AMEND-UTR-EDITFLOW-GOAL-V2-PIVOT-20260919-PHASE0
- **date**: 2026-09-19
- **status**: 用户已批准，生效待 PI 对齐三问答复（见 amendment 附则 / `pi_alignment_three_questions_20260919.md`）
- **amendment 流程依据**: 主合同 §0.4 变更流程
- **worktree / branch**: `/home/cunyuliu/mrna_editflow_goal/worktrees/route_a_v3_w0_diagnosis_20260902` @ `route-a-v3-w0-diagnosis-20260902`（基线 HEAD f30d2e9f）
- **产物根目录**: `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/`

---

## Decision Log (YAML)

```yaml
decision_id: AMEND-UTR-EDITFLOW-GOAL-V2-PIVOT-20260919-PHASE0
date: 2026-09-19
old_text: "mRNA-EditFlow 必须是主方法，不得降级为可选附件"
new_text: "mRNA-EditFlow 生成线在本篇论文中定位为机理规律的应用案例；项目主交付物为 delta 可学性机理归因与 mRNA Source-Relative Edit Effect Benchmark v2。生成线机制结论（B+C 可加、双口径分歧、探索 +0.03 排零）保留完整证据链与独立发表选项（polyA 独立小论文/委员会路由工具）"
reason: "2026-09-19 用户批准 spec pivot-delta-benchmark-mechanism v4（D1 机理主导拍板）：mRNA-EditFlow 从'必须为主方法'转为'机理规律的应用案例'，项目主交付物相应变更为 delta 可学性机理归因与 mRNA Source-Relative Edit Effect Benchmark v2；生成线机制结论保留完整证据链与独立发表选项"
evidence:
  - spec: ".trae/specs/pivot-delta-benchmark-mechanism/spec.md（v4，用户已批准）"
  - frozen_evidence: "MRL E7 上界 0.207 / ERK 0.2015：模型可学性上界被外部基线锁定，生成线无法成为论文主要领先来源"
  - frozen_evidence: "G4 三代全零：模型跨代/跨设置完全失效，排除'再调一版就能领先'的路径"
  - frozen_evidence: "delta-density r=0.9388（25 外部行 Pearson r=0.9388，p=3.9e-12）：delta 可学性与 edit-site 密度的强关联构成机理规律与 benchmark 主交付物的核心证据"
```

---

## 附则：生效条件（PI 对齐三问）

本 amendment 已获用户批准，但生效需待 PI 对齐三问的答复（送达状态与问题全文见
[pi_alignment_three_questions_20260919.md](pi_alignment_three_questions_20260919.md)）：

1. 项目中心科学问题归属：Prediction / generation / benchmark 谁主谁辅；
2. benchmark + 机理归因型论文是否可接受为最终交付（vs 必须有模型领先）；
3. venue 与投稿时间表。

若 PI 对三问的答复与 v4 spec 方向冲突，则按主合同 §0.4 变更流程另行起草修订版本；
若答复一致，则本 amendment 自答复之日起正式生效，无需新版本号。

## 影响范围

- 主交付物：delta 可学性机理归因 + mRNA Source-Relative Edit Effect Benchmark v2。
- 生成线（mRNA-EditFlow）：论文内定位为机理规律的应用案例；机制结论（B+C 可加、双口径分歧、探索 +0.03 排零）证据链保留，可走独立发表路径（polyA 独立小论文 / 委员会路由工具）。
- 相关预注册框架：delta-density 规律的 held-out 检验见
  [delta_density_prereg_framework_v1.md](delta_density_prereg_framework_v1.md)（禁止挑行，降级条款预先固定）。
