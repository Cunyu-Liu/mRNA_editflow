# Route 2 CFG guidance Plan B 登记 v1（2026-09-04）

> 定位：SetFlow guidance 的**无 critic 备选路线**登记（EditFlow 论文原生玩法）。本文为纯登记（registration），不实现任何代码，不改变任何在途运行与已冻结 gate。
> 依据：EditFlow 论文探索结论 F13（详见 `docs/training_journal/ITERATION_U_D0_D1_D2_RECORD_20260904.md` §2）；SetFlow V5 B2 guided 线（potential 式引导，现役）。
> 登记时间：2026-09-04（Asia/Shanghai）；状态：**REGISTERED / NOT_STARTED**（维持登记，不启动——见 §3 复核）。
> 纪律：protected reads = 0；CUDA BF16-only；预注册门槛不事后改。

## 1. 机制（登记内容，不实现）

Classifier-free guidance（CFG）for SetFlow，EditFlow 论文原生玩法：

- **训练侧**：以概率 p_drop 随机丢弃条件 c，使 base flow 同时学会条件分布与无条件分布。条件 c 的构成：任务 / endpoint / 源序列 / 离散化效果分桶（任一或组合，启动前需在修订协议中冻结具体构成）。
- **采样侧**：naïve rate CFG——组合条件 rate 与无条件 rate（EditFlow 论文结论：naïve rate CFG 最优，无需更复杂的 guidance 形式）。
- **关键性质**：全程不需要 reward/critic/potential——引导信号完全来自 base flow 自身的条件/无条件分布差。

## 2. 与 Potential 式引导的关系

| 维度 | Potential 式（现役 B2 线） | naïve rate CFG（本 Plan B） |
|---|---|---|
| **依赖** | 需要冻结 critic 提供 V(s)/V(s′)——引导质量以 critic 可信度为前提（V6 线负结果后由 V5 终态承担） | 无 critic；需要条件 dropout 训练过的 base flow（同一 checkpoint 内含条件/无条件两种 rate） |
| **机制** | 率修正：U_q = U_p·e^{β[V(s′)−V(s)]}（Nisonoff ICLR 2025 形式；本项目超出 EditFlow 论文的自有设计） | 采样时组合条件/无条件 rate（EditFlow 论文原生 guidance 玩法） |
| **定位** | 现役主路线：B2 guided（Gate B2/B3 判定中，2026-09-03 发射全量 891 源） | 备选路线（Plan B）：已登记、未启动 |

## 3. 触发条件（2026-09-04 复核）

**原触发条件**（登记时）："V6/V7 主判据连续未过门**且**架构侧无可试项"。

**2026-09-04 复核结论：维持登记、不启动。**

- 前半（连续未过门）：**已满足**——V6 线六臂全负（V6 首训、H3 λ∈{0.5, 0.75, 1.0}、V7；MPRAU pair-mean ρ 全部未过 V5 0.1025 参照门，详见 `TRAINING_LOG_202609.md` 2026-09-03 批次）。
- 后半（架构侧无可试项）：**不成立**——Route A / V8 已证明 critic 可信度可通过外部先验注入修复（280K 外部库监督预微调线的判别证据链：frozen 280K 先验 0.3132 vs 架构 from-scratch 0.0984），架构侧仍有可试项。
- **重议启动条件**：若 (a) B2 guided 三分归因确认瓶颈在 critic 质量本身，**且** (b) V8 联合先验注入后主判据仍未过门，再重议启动本 Plan B（届时需新协议：冻结 p_drop、条件 c 构成、组合权重与判定门）。

## 4. 边界声明

- 本登记不消耗任何训练/推理预算，不占用 GPU 窗口。
- 启动前置：需协议修订（训练侧条件 dropout 属新实验形态，沿"协议外新实验 → 先 amendment 预注册后启动"纪律）。
- 与 B2 线的关系：互不干扰——B2 为 potential 式（有 critic），本路线为 CFG 式（无 critic）；若 B2 过门则本 Plan B 长期搁置。

---
*登记：2026-09-04，W0 worktree `route_a_v3_w0_diagnosis_20260902`（SPECS_CRITIC_V6 遗留 Task 7）。*
