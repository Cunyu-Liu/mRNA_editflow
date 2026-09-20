# PI Alignment — 三问（2026-09-19）

- **用途**: DeltaBench 转向（spec `pivot-delta-benchmark-mechanism` v4，用户已批准）后，与导师（PI）对齐项目定位的三个关键问题。
- **送达状态**: 待用户转达（本文件起草于 W0 worktree，尚未发送）。
- **关联文档**: [utr_editflow_goal_v2_amendment_pivot_v1.md](utr_editflow_goal_v2_amendment_pivot_v1.md)（amendment 生效以三问答复为条件）；[delta_density_prereg_framework_v1.md](delta_density_prereg_framework_v1.md)。

---

## 问题 ①：项目中心科学问题归属

**问题**：本项目的中心科学问题应归属谁主谁辅——Prediction（可学性机理归因）、generation（mRNA-EditFlow 生成方法）、还是 benchmark（Source-Relative Edit Effect Benchmark v2）？

**背景证据**（冻结数字）：

- MRL 上两代模型与外部基线打平（0.3158 vs 0.3132），且 MRL E7 可学性上界 0.207 / ERK 0.2015 表明模型侧无领先空间；G4 三代全零显示生成线跨代完全失效。
- 与之相对，25 外部行 delta-density Pearson r=0.9388（p=3.9e-12）显示"delta 可学性与 edit-site 密度强关联"这一规律高度稳定，构成可主张的机理结论。

**若答"generation 必须为主"的备选路径**：生成线机制结论（B+C 可加、双口径分歧、探索 +0.03 排零）保留完整证据链，可拆出 polyA 独立小论文（polyA V5 0.8219，Holm 校正 p=0.004）或委员会路由工具作为独立发表/交付物，主论文仍按 pivot 方向执行。

## 问题 ②：benchmark + 机理归因型论文是否可接受为最终交付

**问题**：以"benchmark（mRNA Source-Relative Edit Effect Benchmark v2）+ delta 可学性机理归因"为核心最终交付的论文，是否可接受为最终交付？还是必须有模型性能领先？

**背景证据**（冻结数字）：

- 模型侧领先路径已被冻结证据封死：MRL 平局 0.3158 vs 0.3132、MRL E7 上界 0.207 / ERK 0.2015、G4 三代全零。
- 机理侧证据充分：delta-density r=0.9388（p=3.9e-12，25 外部行）；polyA 单域机制结论显著（V5 0.8219，Holm p=0.004），可作为应用案例章或独立小论文。

**若答"必须有模型领先"的备选路径**：按问题 ① 的备选路径处理——生成线拆出 polyA 独立小论文 / 委员会路由工具承载模型侧交付，主论文维持 benchmark + 机理归因定位（MRL E7 / ERK 上界证据本身即构成"为何不该期待模型领先"的机理论证素材）。

## 问题 ③：venue 与投稿时间表

**问题**：若按 pivot 后的定位（benchmark + 机理归因主交付、生成线为应用案例），目标 venue 与投稿时间表如何设定？是否需要在时间表中并行安排 polyA 独立小论文的投稿节点？

**背景证据**（冻结数字）：

- benchmark + 机理型工作对 venue 的适配性与"必须有 SOTA 模型"型工作不同：delta-density r=0.9388 与 MRL/ERK 上界（0.207 / 0.2015）共同构成"规律 + 边界"的完整叙事，适合 benchmark/数据-评测导向的会议或期刊。
- polyA 独立小论文有独立显著证据（V5 0.8219，Holm p=0.004），若时间表允许，可作为并行投稿选项而不与主论文互斥。

**若答"venue/时间表不可行"的备选路径**：生成线与机制结论整体拆分为 polyA 独立小论文 + 委员会路由工具两条独立交付路径，主论文按 PI 认可的 venue 与时间表调整定位（仍以冻结证据为边界，不回退到"必须模型领先"叙事）。

---

## 送达与答复处理

- 送达状态：**待用户转达**。
- 答复后处理：答复与 spec `pivot-delta-benchmark-mechanism` v4 方向一致 → amendment（v1）正式生效；不一致 → 按主合同 §0.4 变更流程起草修订版本。

---

## 答复记录（2026-09-20 · 用户直答，Phase 0 三问闭合）

| # | 问题 | 用户答复 | 对执行的影响 |
|---|---|---|---|
| ① | 中心科学问题归属 | **benchmark 为主** | 主论文骨架以 Benchmark v2 为第一主干，机理归因为其核心分析章节（与 spec v4 D1"机理主导、benchmark 为载体"合并理解为：benchmark 为主干、机理归因为核心分析——骨架排序按用户口径微调，主张内容不变） |
| ② | 机理归因型论文可否为最终交付 | **可以接受** | 合同 amendment（utr_editflow_goal_v2_amendment_pivot_v1.md）生效条件达成——答复与 spec v4 方向一致，amendment v1 正式生效。答复来源 = 用户本人（项目所有者）；若导师后续另有意见，走主合同 §0.4 修订流程 |
| ③ | venue 与时间表 | **polyA 独立小论文叙事需先讨论**（主论文方向确认，小论文故事待定） | 主论文绿色通道开启（Phase 2 解冻，矩阵评测立即执行）；polyA 小论文 D5 决策挂起，进入叙事讨论（发起人提案见后续 journal/文档） |

**Phase 2 解冻生效**：Task 2.2-2.4（矩阵评测执行/复核/delta-density 扩样）+ 3.5（新行补格）前置条件全部达成。

---

## 答复记录（2026-09-20 · 用户转达 PI 答复）

- **问题 ① 答复：以 benchmark 为主**。主论文定位微调：benchmark 为第一主体交付、delta 可学性机理归因为分析核心（spec D1 细化，Task 4.1 章节权重随之调整：benchmark 章前置）。
- **问题 ② 答复：机理归因型论文可接受为最终交付**。amendment（utr_editflow_goal_v2_amendment_pivot_v1.md）生效条件达成，正式生效。
- **问题 ③ 答复：polyA 独立小论文的叙事需讨论后定稿**（用户关注点：领先幅度不大，科学故事怎么讲）。主论文按 benchmark 为主方向立即推进（Phase 2 矩阵评测解冻执行）。
