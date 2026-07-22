# mRNA-EditFlow：从可运行原型到可信 SOTA 的研究路线图

> 版本：2026-07-20 P1-completion critical revision（v3，在 2026-07-19 P0-v2-completion revision 基础上更新）
> 适用范围：`/home/cunyuliu/mrna_editflow_goal/mrna_editflow`
> 文档职责：只记录科学目标、证据门槛、工作包和 go/no-go 决策；运行 PID、逐次日志和自动化状态以 `docs/*readiness*.{json,md}`、`benchmark/**/summary.json`、`logs/*.profile.jsonl` 为准。
> **v3 修订摘要（2026-07-20，批判性重评）**：
>   1. **承认 P1 表面完成但底层证据仍薄弱**：14 个 task 交付，但 4 个 Stage A 100k run 全部 STOPPED（loss stable, grad diverging, AMP fallback 0.29–0.44），项目目前**没有可用 trained backbone**；所有 RL/synergy panel 使用 random init policy，**不能证明 RL 真的 work**。
>   2. **修正 cross-region synergy 误标**：P1-13 1000-wild-type panel 检测到的"显著 synergy" (p=0.0034, d=0.09) **实际是 5'UTR-internal multi-edit synergy**，因为 `LocalTranslationOracle` 只消费 5'UTR + CDS 前 12 nt，CDS/3'UTR arm 严格为 0。**Cross-region (5'UTR × CDS × 3'UTR) synergy 仍未被检验**。dashboard 中 `RL mechanism` 从 `YES` 下调到 `PARTIAL`。
>   3. **下调顶刊定位**：当前证据支持 Nature Communications / Nature Methods / NAR 级别；要达到 Nature Biotechnology / Science 需 (a) multi-region oracle 下 cross-region synergy 显著，(b) full-mRNA RL validated，(c) ≥2 cargo × ≥2 cell context prospective wet-lab，三项均未完成。原"Nature Biotechnology 主投"为 over-claim。
>   4. **统计有效性仍是 #1 blocker**：head256/head1024 结果仍 in-sample；10 seeds 是 decoder seeds 不是 training seeds；family-cluster bootstrap CI 未执行。
>   5. **新增 Section 14 P2 执行 goal**（替代 Section 13 P1 goal，P1 已完成归档）：按 blocker 优先级拆分 12 个 task，交给 trae 自主执行。
> v2 修订摘要（2026-07-19）：(1) 反映 P0 Data Reconstruction v2 已完成，4 个 split manifest + combined bundle 全部 `paper_eligible=true`；(2) 加入独立批判性评估结论；(3) 加入 Stage A 100k 健康决策作为 P1 紧急前置；(4) 重写下一阶段 goal 以便 trae 自主执行。

## 0. 一页结论

### 0.1 当前项目的准确定位

**战略定位（2026-07-19 修订，资源充足 + 顶刊目标 + 计算与算法壁垒优先）**：

MEF 的目标不是在工程规模上追赶 GEMORNA/mRNA-GPT 的 de novo SOTA，而是**重新定义 mRNA 设计问题**：从"生成新序列"转为"最小可审计编辑 + 跨区域协同 + 独立可验证"。差异化壁垒按优先级 2 > 4 > 1 > 3：

- **壁垒 2（最高优先级，生物学机制维度）**：Cross-region synergy——5'UTR/CDS/3'UTR 协同编辑的机制性发现与可优化指标。这是领域空白，没有任何现有方法显式学习或量化 cross-region synergy。
- **壁垒 4（算法维度）**：RL 算法创新——以 counterfactual cross-region synergy RL、constrained trajectory optimization、adversarial reward de-biasing、hierarchical region-options RL、offline-to-online conservative Q-learning 五个方向建立算法壁垒。counterfactual synergy RL 可单独投 NeurIPS/ICML。
- **壁垒 1（监管科学维度）**：Regulatory-grade minimal-edit——discrete CTMC edit grammar + constructive protein invariance 产生可审计编辑轨迹，对治疗性 mRNA 的 FDA 审评有范式转移意义。
- **壁垒 3（方法学维度，弱化）**：Independent oracle + 合作湿实验验证。湿实验不是项目强项，以合作方式完成 prospective MPRA + 蛋白表达验证即可；核心 claim 不依赖湿实验 superiority，而以计算独立 oracle + 算法创新 + 机制发现为主。

当前系统已经是一个工程完成度较高的 **source-conditioned、hard-constrained、full-transcript-context local editor/reranker**：它读取 `5'UTR + CDS + 3'UTR`，在 UTR 做核苷酸级编辑，在 CDS 做同义密码子编辑，并用 legality、protein identity、frame 和 edit budget 硬门控候选。

当前系统还不是以下任一目标：

- 不是从蛋白或属性条件出发的全长 mRNA de novo 生成器；
- 不是经过真实 MPRA、half-life、protein expression 或 in vivo 验证的设计系统；
- 不是已在 family-disjoint / cross-source 测试集上成立的外部 SOTA；
- 不是完整实现并在推理期采样的 CTMC edit-flow 生成器——当前 checkpoint-guided decoder 是逐步重算 rate field 的贪心/Top-k 合法局部编辑，代码明确标注为 "not a full tau-leaping sampler"；
- 不是已经证明随数据、参数和训练步数稳定扩展的 scale-law 模型。
- 不是已经完成 on-policy reinforcement learning 的设计策略：当前 `grpo_fused_score` 只是多目标分数的组内标准化融合，`train_proposal_ranker.py` 是离线 Bradley–Terry 蒸馏，二者都不能称为 GRPO/RL policy optimization。
- **不是已实现 counterfactual cross-region synergy RL 的系统**（这是壁垒 4 的核心创新，待 P1-P2 实现）。

### 0.2 当前最强证据与必须附带的限制

| 证据                                        | 当前结果                                                     | 能支持的表述                                                 | 不能支持的表述                            |
| ------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------- |
| 数据完整性（P0 v2，2026-07-19 完成） | 4 split + combined manifest 全部 `paper_eligible=true`；148,843 records / 17,057 families；alias audit 0 gaps；near-neighbor audit 0 violations（remediation 移除 858 条，max 4.2%/split）；frozen SHA-256 不变 | GENCODE/RefSeq family-disjoint、near-neighbor-audited 数据合同已就位 | 跨物种、跨 cell/cargo、真实功能标签、wet-lab |
| head256、10 个 decoder seeds，多目标 ranker | grpo `delta_oracle_te=+0.01114`，vs te_only `p=0.00450`，硬约束为 1.0 | 同一内部 proxy、同一数据切片、同一训练 recipe 下，多目标蒸馏改善局部编辑排序 | 外部 SOTA、真实 TE 提升、独立训练重复 |
| head1024、10 个 decoder seeds               | pareto vs te_only `+0.00081`，`p=0.05047`；scalar/grpo 不显著 | 扩大评测切片后收益明显收缩，pareto 仅有 borderline 趋势      | 稳健 scale-up、严格显著超过强对照         |
| CDS 硬约束                                  | protein/frame exact-1；codon-lattice DP 提高 CAI proxy       | 同义编辑语法和硬约束有效                                     | 超过 LinearDesign/EnsembleDesign/codonGPT |
| 外部工具接入                                | LinearDesign、codonGPT、UTailoR、UTRGAN 有测量行；EnsembleDesign 输出 coverage 无效；Prot2RNA 缺失 | 外部协议与适配器基础已建立                                   | 完整、公平的外部 SOTA 表                  |
| 数据                                        | GENCODE v45 清洗后 54,680 条 human transcripts               | 单一人类自然序列语料可用于原型训练                           | 数据全面、跨物种或跨来源泛化              |

### 0.3 当前阻止论文级结论的六个最高风险

1. **训练–评测泄漏（部分缓解，2026-07-19）**：headline head256 查询对 GENCODE 全语料的 audit 历史上为 `exact_match_count=256/256`、`flagged_fraction=1.0`。P0 v2 已完成 family-disjoint + near-neighbor-disjoint 数据合同（4 split + combined manifest `paper_eligible=true`，alias audit 0 gaps，near-neighbor audit 0 violations）。**遗留风险**：(a) 该合同尚未被所有训练/teacher/ranker/eval 入口强制消费（`--train-idx`/`--val-idx`/`--test-idx` 仍非必需）；(b) foundation pretraining corpus 的泄漏审计仍未做；(c) 现有 benchmark/headline 结果仍是 in-sample，需要在冻结 test split 上重跑。
2. **reward–evaluation circularity（未缓解）**：`multiobjective_teacher_export.py` 和 `eval/run_eval.py` 默认都实例化同一个 `LocalTranslationOracle`；该 oracle 默认是人工特征公式，两个 predictor 也共享 GC/Kozak/accessibility/uAUG 等特征。当前结果证明"拟合同一 proxy 的能力"，不能证明真实 translation 或对独立 oracle 的迁移。
3. **统计重复单位错误（未缓解）**：当前 10 seeds 主要是同一 checkpoint、同一 records 上的 decoder seeds，不是独立训练 seeds；对 10 个 seed 均值做 paired permutation 会夸大独立性。论文主检验必须以 held-out transcript/family 为生物统计单位，并把 training seed、decoder seed 作为嵌套随机因素。
4. **"全长"数据实际被截断（部分缓解，2026-07-19）**：现有 GENCODE manifest 使用 `5'UTR<=128`、`CDS<=1536`、`3'UTR<=256`；111,048 条原始记录只保留 54,680 条，27,521 条 5'UTR 和 40,742 条 3'UTR 被截断，25,610 条长 CDS 被丢弃。A100 config 虽把 cap 写为 512/1536/1024，但它读取的 records 已经按旧 cap 截断。P0 v2 已在冻结 namespace 上完成 family/near-neighbor 审计，但**未重建未截断 long view**；G0.5 仍是未完成 gate。
5. **生物证据缺失（未缓解，最高优先级）**：真实 MPRA/TE 与 stability/half-life 输入均不存在，predictor 只有 synthetic smoke；没有 prospective wet-lab。与已有 Nature/Science 级 mRNA 设计工作相比，这是决定性差距。
6. **RL reward hacking 与 protocol overfitting（未缓解）**：如果 RL、候选选择和最终评测共享同一个 proxy，策略会比当前 ranker 更强地放大 oracle shortcut；固定加权和也可能牺牲未加权目标、压缩多样性。任何 RL 性能提升必须在冻结的独立 oracle、等 oracle-call budget 搜索基线和 prospective assay 上复现。

**新增风险 7（2026-07-19）**：**Stage A 100k 训练健康问题**。4 个 A100-max 100k Stage A runs（seed 0/1/2/5）已运行 ~4 天，位于 7k–9k/100k steps；`stage_a_health_audit.md` 给出 `manual_review_required` verdict，AMP fallback 0.20–0.35，gradient norm P99 高达 2200–8900，ETA 估计 26–29 天剩余。在继续投入 GPU 资源前必须先做：duplicate-`sample_cond_pt` correctness test、NaN/AMP stress test、held-out learning curve 审计、200/1k/5k/10k checkpoint panel 验证；如 held-out 无增益或数值异常持续，应停止后续无效 run 并重启或转向。

### 0.4 总体判断

> **v3 重评（2026-07-20）**：P1 表面 14 task 完成，但 critical evidence 薄弱。Stage A 4 个 run 全部 STOPPED，无可用 trained backbone；cross-region synergy 实际是 5'UTR-internal；统计单位错误；外部 SOTA protocol fidelity 不足。**当前证据不支持顶刊主投，应先清除 P2 blocker 再评估投稿层级**。

| 维度                | 判断                                             | 说明                                                         |
| ------------------- | ------------------------------------------------ | ------------------------------------------------------------ |
| 目标是否清晰        | **主方向清晰，壁垒 2/4/1/3 优先级明确，但壁垒 2 primary claim 风险被低估** | 战略定位：重新定义 mRNA 设计问题（minimal-edit + cross-region synergy + RL 算法创新），不追 de novo SOTA；壁垒 2 (cross-region synergy) > 壁垒 4 (RL 算法创新) > 壁垒 1 (regulatory-grade minimal-edit) > 壁垒 3 (independent oracle + 合作湿实验)。**风险**：当前 synergy panel 显示的只是 5'UTR-internal synergy，cross-region (5'UTR × CDS × 3'UTR) synergy 仍未检验；若不显著，壁垒 2 降级为 null finding。 |
| Todo 是否细致       | **P1 已完成（14 task），P2 待新拆分（见 Section 14）**    | P1-00 (Stage A 健康) / P1-02A/B (真实 + 自建 full-length MPRA) / P1-04/05 (predictor + oracle) / P1-07/08/09 (RL correctness) / P1-12 (Innovation 1+2) / P1-13 (counterfactual panel) / P1-10/11 (split 强制 + long-view)。P2 需细化为 12 task（见 Section 14）。 |
| 能否实现预期 SOTA   | **有条件可行，但 3 个 blocker 必须先清除**：(1) Stage A 修复或 pivot；(2) cross-region synergy 在 multi-region oracle 下确认；(3) leakage-free headline eval | 不追 de novo SOTA（GEMORNA/mRNA-GPT 工程规模优势大）；以 cross-region synergy 机制发现 + counterfactual RL 算法创新 + regulatory-grade minimal-edit 建立差异化 SOTA。**当前最大风险**：Stage A broken → 无 trained backbone → RL/synergy panel 都用 random policy → 不能 claim 任何 RL/synergy efficacy。 |
| 是否足够顶刊        | **当前证据支持 Nature Communications / Nature Methods / NAR；Nature Biotechnology / Science 需 3 项条件全满足** | 当前证据：(1) cross-region synergy 仅 5'UTR-internal，(2) RL 仅 tiny-MDP validation，(3) 无 wet-lab，(4) 统计单位错误。Nature Biotechnology / Science 需：(a) multi-region oracle 下 cross-region synergy 显著 (d>0.5, p<0.001)，(b) full-mRNA RL 在等 query 下优于 beam/SA (3+ policy seeds, family-cluster CI)，(c) ≥2 cargo × ≥2 cell context prospective wet-lab 方向一致。Innovation 2 (counterfactual synergy RL) 单独可投 NeurIPS/ICML **workshop**（不是 main conference）除非算法理论贡献大幅加强。 |
| 科研故事是否吸引人  | **有强潜力，但当前 panel 的"synergy"有 mislabel 风险** | "Cross-region synergy via counterfactual RL" 是领域空白；如果 multi-region oracle 下 synergy 显著为正，这是可发表的新生物学发现。**当前风险**：P1-13 panel 检测到的"synergy"实际是 5'UTR-internal multi-edit synergy，若 paper 直接写"cross-region synergy"会被 reviewer 立刻指出 oracle limitation。必须先用 P1-04 ensemble 替换 LocalTranslationOracle 重跑 panel。 |
| 数据集是否全面      | **Layer A 通过，B/C 已获取但未集成到 cross-region panel；Leplek 2022 PERSIST-Seq 未获取；wet-lab 完全无** | Layer A (GENCODE/RefSeq) v2 audit 通过；B/C (MPRA + stability) P1-02A/P1-03 已获取但 cross-region synergy panel 仍用 LocalTranslationOracle；P1-02B 自建 full-length joint mRNA MPRA 仅设计文档，未执行；Leplek 2022 PERSIST-Seq (3'UTR stability MPRA) 未获取；真实 wet-lab 完全无。 |
| 下游任务是否全面    | **D1-D7 定义良好，D7 (prospective wet-lab) 完全未启动；counterfactual panel 是新增机制发现任务** | D1-D7 + 新增 counterfactual cross-region synergy panel（5 arms × 1000 wild-type）；D7 (prospective wet-lab) 完全未启动，是顶刊投稿的硬 blocker。 |
| 测评是否全面        | **协议覆盖广，但执行不完整：统计单位错误、外部 SOTA protocol fidelity 不足、10-seed paired test 未执行** | 缺 leakage-free OOD（v2 后家庭级已部分满足）、独立 oracle（P1-04/05 已落地但未集成到 synergy panel）、训练重复（10 decoder seeds ≠ training seeds）、校准、湿实验；外部 SOTA 4/6 measured 但 protocol fidelity 不足（UTRGAN paired invalid, EnsembleDesign coverage invalid）；10-seed paired significance test 未执行。 |
| 架构是否最优        | **明确不是，且 Stage A 4 run STOPPED 阻断架构评估** | dense Transformer O(L²)；region adapter 严格负结果；cascade 无稳定 top-1 提升；aux_struct 零 target；rate field 与 ranker 共享 head；Stage A 4 run 全部 STOPPED（loss stable, grad diverging, AMP fallback 0.29–0.44, 无 step-level checkpoint）；**架构升级在 P2 优先级低于 blocker 清除**。 |
| RL 是否值得成为主线 | **是，壁垒 4 是核心算法创新，但当前只有 tiny-MDP validation，full-mRNA validation 是 P2 关键** | Innovation 1 CTO + Innovation 2 counterfactual synergy RL 是核心；Innovation 3/4/5 作为 parallel/backup；Innovation 2 可单独投 NeurIPS/ICML **workshop**（理论贡献加强后可冲 main conference）。**当前风险**：Stage A broken → 无 warm-start checkpoint → full-mRNA RL 无法启动。 |
| Cross-region synergy 是否值得成为 primary claim | **是，领域空白；但当前 panel 不能支持该 claim，必须先在 multi-region oracle 下重跑** | 没有任何现有方法显式学习或量化 cross-region synergy；counterfactual panel + synergy score + RL 优化 = 机制发现 + 算法创新双壁垒；**当前 P1-13 panel 只检测到 5'UTR-internal synergy，cross-region (5'UTR × CDS × 3'UTR) synergy 仍未被检验**；降级路径：若 multi-region oracle 下 synergy 不显著，壁垒 4 + 壁垒 1 仍可支撑 Nature Methods / Nat Comm 投稿。 |

---

## 1. 项目事实基线

### 1.1 代码与证据链

| 层       | 主要实现                                                     | 当前状态                                                     | 关键风险                                                     |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 数据     | `data/download_mrna.py`, `public_pipeline.py`, `clean_mrna.py`, `dedup_split.py`, `leakage_audit.py` | 下载、清洗、family split、manifest、leakage 工具齐全         | split 没有被训练/teacher/eval 强制消费；records 预截断       |
| Stage A  | `train_backbone.py`, `core/mrna_flow_utils.py`               | hybrid coupling + Edit-Flow loss 可训练                      | `_flow_batch_loss` 连续调用两次 `sample_cond_pt`，第一次结果被覆盖；需 correctness audit |
| 模型     | `models/mrna_editformer.py`                                  | dense Transformer、RoPE、Region/phase FiLM、CTMC rate heads、codon mask | `O(L^2)`；foundation backbones 多为 placeholder；结构 aux target 为全零 |
| 推理     | `sample.py`                                                  | T4/T5/T6 的合法贪心/Top-k 编辑；cascade 可运行               | 不是完整 flow sampler；T1/T7 checkpoint 路径回退到 deterministic operator |
| Ranker   | `train_proposal_ranker.py`, multiobjective teacher exporters | Bradley–Terry 蒸馏，scalar/Pareto/standardized fusion        | teacher 与 evaluation oracle circular；同 checkpoint 兼作生成 field 和 rank score |
| 评测     | `eval/run_eval.py`, `eval/metrics.py`, `eval/oracle.py`      | legality、distribution、novelty、CAI、budget、length、motif、proxy TE/MRL | T1–T7 是 metric bundle；T4 在 UTR-only run 上可被结构性保证；缺独立标签 |
| 外部基线 | `baselines/external_*`, `scripts/external_*`                 | 多个工具 adapter、统一 input pack 和 audit                   | task conditioning、edit budget、per-record pairing、protocol fidelity 未全部匹配 |
| 工程治理 | readiness JSON/MD、paper table builders、tests               | 审计和 claim flags 很丰富                                    | 文档完成度不能替代科学证据；多个自动状态与实际文件不同步     |

### 1.2 2026-07-19 运行快照

**P0 Data Reconstruction v2 已完成（2026-07-19）**：
- 4 split manifest（combined_family、gencode_family、refseq_family、gencode_to_refseq）+ combined bundle 全部 `paper_eligible=true`，`block_reasons=[]`。
- 独立 gene-symbol alias audit：148,843 records / 17,057 families，0 protein_sha256 gaps，0 rna_sha256 gaps，414 cross-source alias observations（合规）。
- 穷举 cross-role near-neighbor audit（k=15, jaccard≥0.8, containment≥0.95）：初始发现 5,013 violations，remediation 移除 858 val/test records（max 4.2%/split）至 excluded，post-remediation 0 violations。
- Frozen artifact SHA-256 不变（4 个 v1 frozen artifacts 验证通过）。
- 4 个 Stage A 长进程（PID 1495455/1495549/1495551/1499316）未被干扰，仍存活。
- 258/258 tests 通过（241 v1 + 17 v2）。
- 交付物：`data/reconstruction_v2_audit.py`、`scripts/remediate_v2.py`、`scripts/run_v2_audit.py`、`tests/test_data_reconstruction_v2_audit.py`、`docs/p0_data_reconstruction_v2_audit.md`。

**Stage A 100k 训练状态（2026-07-19，紧急关注）**：
- 四个 A100-max 100k Stage A runs（seed 0/1/2/5）仍在运行，约位于 7k–9k/100k steps（已运行 ~4 天）。
- `stage_a_health_audit.md` 给出 `manual_review_required` verdict：AMP fallback 0.20–0.35，gradient norm P99 高达 2200–8900，ETA 估计 26–29 天剩余。
- 在继续消耗数周 GPU 前必须先做：duplicate-`sample_cond_pt` correctness test、NaN/AMP stress test、held-out learning curve 审计、200/1k/5k/10k checkpoint panel 验证。
- **不擅自终止进程**；先做审计，给出 continue/stop/restart 决策建议。

**其他状态**：
- `data/raw/human.1.rna.gbff.gz` 已存在（约 80.9 MB），但 `refseq_public_build_20260715_0743/status.json` 仍报告 raw 不存在且只停在 `build_start`；这是状态刷新失败，不是 RefSeq 尚未下载。
- 当前 dataset manifest 完整度为 `1/4`：仅 GENCODE 完整；RefSeq、MPRA/TE、stability/half-life 不完整。
- 当前 external audit 为 6 个预期方法中 4 个 measured、1 个 invalid、1 个 missing；只有 1 个达到 protocol-fidelity sufficient。
- 现有 benchmark/headline 结果仍是 in-sample（在 v2 之前的 head256/head1024 sources.jsonl 上）；尚未在冻结 test split 上重跑。

### 1.3 当前允许的 claim

可以写：

> mRNA-EditFlow 是一个全转录本上下文、源序列条件、硬约束的局部 mRNA 编辑原型；在当前 GENCODE-derived、in-sample、手工 proxy oracle 评测上，多目标 proposal distillation 相对同 recipe 单目标对照改善了 proxy 排序，并保持由语法保证的 protein/frame/edit-budget 约束。

**新增（2026-07-19，基于 P0 v2 完成）**：

> 项目数据层已通过独立审计：GENCODE + RefSeq 合并的 148,843 records / 17,057 families 在 4 个 family-disjoint split（combined_family、gencode_family、refseq_family、gencode_to_refseq）上满足 (a) 独立 gene-symbol alias audit 0 gaps，(b) 穷举 cross-role near-neighbor audit（k=15, jaccard≥0.8, containment≥0.95）0 violations，(c) frozen artifact SHA-256 不变。所有 split manifest 与 combined bundle 已 `paper_eligible=true`。

在以下 gate 清除前不得写：

- "full-length de novo generation"；
- "SOTA" 或 "beats external methods"；
- "improves translation/stability/expression" 而不加 `predicted/internal proxy`；
- "generalizes across families/species/cell types"——可写 "family-disjoint on GENCODE+RefSeq human"，但不得写跨物种或跨 cell/cargo；
- "true scale law"；
- "wet-lab validated" 或 therapeutic efficacy；
- "independent oracle" 或 "cross-fit predictor"——当前 oracle 仍是 `LocalTranslationOracle` 同源。

---

## 2. 重新聚焦的科研故事

### 2.1 主问题

> **在严格保持蛋白、阅读框和用户给定 edit budget 的前提下，能否用少量、可解释、全转录本上下文感知的编辑，使 mRNA 在独立测量支持的 translation–stability–safety Pareto frontier 上稳定前移，并跨 protein family、cargo 和 cell context 泛化？**

这个问题比“再做一个 full-length de novo generator”更适合当前代码资产，也更容易形成差异化：

- GEMORNA、mRNA-GPT、ProMORNA、T3PO-mRNA、mRNAutilus 等已经占据 de novo / reward-guided full-length generation 叙事；
- MEF 的真正独特性是 **minimal intervention、hard guarantee、edit trace、source-specific repair**；
- 医学和工程价值可表述为：保留一个已有可制造/已验证的 mRNA，仅修复有限调控缺陷，降低重新验证整条序列的风险。

### 2.2 四条可证伪假设

**H1 — Hard-constrained minimal editing**  
在完全相同的起始序列与 edit budget 下，learned editor 相对 random edit、greedy proxy search、UTR local search、codon DP 和模型-only published baselines，取得更高的独立 held-out property gain / edit，并保持 hard constraints exact-1。

**H2 — Full-transcript context matters**  
联合看到 5'UTR/CDS/3'UTR 的模型，在只允许同一局部区域编辑时，优于只看该区域的模型；提升应在 cross-region swap/counterfactual panel 上出现，而不能只靠序列长度或 GC shortcut。

**H3 — Multi-objective learning transfers beyond its teacher**  
多目标策略在未参与 teacher 训练的 predictor、数据集、cell type 和 wet-lab assay 上改善 Pareto hypervolume，而不只是最大化 teacher 使用的手工特征。

**H4 — RL learns non-myopic edit trajectories**
在完全相同的初始化、合法动作空间、候选/奖励查询预算和 wall-clock/compute 报告下，RL policy 相对离线 proposal distillation、greedy、beam、best-of-N、simulated annealing 和逐步 oracle search，能学习单步排序无法恢复的多步协同编辑；优势必须表现为独立 oracle 上更高的 Pareto hypervolume、gain/edit 或更低 oracle-call cost，而不是只提高训练 reward。

若 H2 不成立，论文应降级为“安全局部 mRNA optimizer”，不再宣称 full-transcript-context 是核心机制。若 H3 不成立，应删除多目标 SOTA claim，只保留 hard-constrained optimization 工程贡献。
若 H4 不成立，保留 RL 为诚实负结果或个性化可选模块，论文主算法回到最强的蒸馏/搜索方法；不得因为投入成本而把 RL 强行放进 headline。

### 2.3 Claim ladder

| 级别            | 必须满足的证据                                               | 当前状态                       |
| --------------- | ------------------------------------------------------------ | ------------------------------ |
| C0 工程可运行   | 单测、合法编辑、artifact provenance                          | 大体满足                       |
| C1 内部 proxy   | 固定 split、无泄漏、独立 records、effect size/CI             | **未满足：headline in-sample** |
| C2 独立计算验证 | cross-fitted real-label predictors、OOD、oracle disagreement | 未满足                         |
| C2-RL 策略增益  | ≥3 policy seeds、等预算强搜索、独立 oracle、reward-hacking audit | 未满足                         |
| C3 外部 SOTA    | 公平 task/budget/conditioning、官方实现、配对检验            | 未满足                         |
| C4 生物验证     | prospective IVT + cell assay，多 cargo/cell/replicate        | 未满足                         |
| C5 顶刊故事     | 原创、重要、跨领域、机制/意外发现、强验证                    | 未满足                         |

### 2.4 四维壁垒（按 2 > 4 > 1 > 3 优先级，2026-07-19 新增）

资源充足 + 计算与算法强项 + 湿实验合作模式的设定下，MEF 的差异化壁垒按以下优先级建立。**核心 claim 不依赖湿实验 superiority，而以机制发现 + 算法创新 + 计算独立 oracle 为主**。

#### 壁垒 2（最高优先级）：Cross-Region Synergy — 生物学机制维度的领域空白

**已知生物学事实**：5'UTR secondary structure 影响 ribosome scanning，CDS codon usage 影响 elongation rate，3'UTR motif 影响 stability——三者共同决定 TE & half-life，且存在协同效应（slow-scanning 5'UTR 配合 optimal CDS codon 可能比单独优化任一区域更好，因为 ribosome traffic 协调）。

**领域空白**：没有任何现有方法显式学习或量化 cross-region synergy：
- LinearDesign 只做 CDS；
- GEMORNA 是 joint generation 但无 synergy 量化；
- ProMORNA RL reward 是 sequence-level，不分区域；
- RNAGenScape 是 continuous latent，不分区域。

**MEF 的壁垒**：
1. 定义新指标 **cross-region synergy score** = `joint_edit_improvement - Σ single_region_edit_improvement`，作为可优化的 RL 目标与可报告的机制性指标。
2. 设计 counterfactual 编辑实验 panel（只改 5'UTR / 只改 CDS / 只改 3'UTR / 联合改），量化 synergy。
3. 用 RL Innovation 2（counterfactual cross-region synergy RL）显式优化 synergy score。
4. 如果 synergy score 显著为正，**这是可发表的新生物学发现**——不只是工程改进，是机制揭示。

**为什么这是壁垒**：竞争者要复制需要 (a) discrete edit grammar 支持 counterfactual 编辑，(b) RL 算法支持 counterfactual reward，(c) 跨区域 attention 架构，(d) 独立 oracle 验证。任一项缺失都无法 claim synergy。

#### 壁垒 4：RL 算法创新 — 算法维度的 5 个方向

ProMORNA 的 MO-GRPO 是 sequence-level multi-objective RL，**没有针对 mRNA 编辑任务的算法创新**。MEF 在 5 个方向建立算法壁垒（详见 Section 7.10）：

1. **Constrained Trajectory Optimization (CTO)**：构造性硬约束 + Lagrangian 软目标，区别于现有 constrained RL 的 soft penalty。
2. **Counterfactual Cross-Region Synergy RL（核心创新）**：counterfactual reward 显式优化 cross-region synergy，可单独投 NeurIPS/ICML。
3. **Adversarial Reward De-Biasing**：reward model 与 adversarial discriminator 对抗，防止 reward hacking。
4. **Hierarchical Region-Options RL**：high-level 选区域 + low-level 选 edit，显式建模编辑顺序。
5. **Offline-to-Online Conservative Q-Learning**：解决 mRNA 设计 online 探索成本高的实际问题。

#### 壁垒 1：Regulatory-Grade Minimal-Edit — 监管科学维度的范式转移

治疗性 mRNA 的 FDA 审评核心问题不是"你的序列 TE 多少"，而是"**你的序列与 wild-type 差多少？每个改动为什么？**"任何 de novo generation（GEMORNA/mRNA-GPT/ProMORNA）都无法回答这个问题——全新序列需要完整的 immunogenicity、toxicity、re-evaluation 路径。

**MEF 的 discrete CTMC edit grammar + constructive protein invariance 天然产生可审计编辑轨迹**：每个 edit 是离散决策，protein aa 序列构造性不变，编辑数量与位置可解释。

**这个壁垒的实证要求**（计算层面即可满足）：
- edit distance vs functional improvement 的 Pareto frontier 优于 de novo methods 在"等 functional improvement"下的 edit distance；
- 编辑轨迹可解释性指标（每个 edit 的 contribution 可量化）；
- hard constraint 保持率（protein aa 100% identical, frame 100% preserved, key motif 100% preserved）。

#### 壁垒 3（弱化）：Independent Oracle + 合作湿实验验证

湿实验不是项目强项，以合作方式完成：
- **计算独立 oracle（必做）**：cross-fitted predictor ensemble + frozen hidden final oracle，与 teacher 数据/权重/feature 完全不同；
- **合作 prospective MPRA（推荐）**：与湿实验合作方共同设计 1000-10000 序列，送 MPRA 测 TE & stability；
- **合作蛋白表达验证（可选）**：transfection + ELISA/Western，多 cargo 多 cell。

**核心 claim 不依赖湿实验 superiority**：即使湿实验未做或 negative，计算独立 oracle + 算法创新 + 机制发现仍可支撑 Nature Biotechnology / Nature Methods / Nature Communications 投稿。湿实验作为 "validation" 而非 "primary evidence"。

---

## 3. P0：先修科学有效性，不再盲目 scale

P0 中任何一个红色 gate 未通过，都禁止把后续架构/规模实验写成正向 SOTA 证据。

**P0 v2 完成状态（2026-07-19）**：G0.1 的 family-disjoint + near-neighbor-disjoint 数据合同已通过独立审计（4 split + combined manifest `paper_eligible=true`）；但 G0.1 的"训练入口强制消费"、G0.2 重跑 headline、G0.3 独立 oracle、G0.4 代码正确性、G0.5 long-view 重建仍未完成。详见以下分项。

### G0.1 强制 split contract

- [x] **(部分完成，2026-07-19)** 从原始 GENCODE/RefSeq 重建 canonical records；先保留完整 region，再生成不同 length caps 的派生 view，禁止把截断后的 view 当作原始真值。**已完成**：v1 frozen namespace 已通过 v2 独立审计；**未完成**：long-view 重建（见 G0.5）。
- [x] **(完成，2026-07-19)** 以 protein/gene family 为第一层、exact transcript/UTR near-neighbour 为第二层做 train/val/test；同一蛋白的 isoforms 和同源 family 不得跨 split。**已完成**：4 split manifest 全部 `family_disjoint=true`、`near_neighbor_passed=true`。
- [x] **(完成，2026-07-19)** 生成 `split_manifest.json`：记录 raw/records/split SHA、family 算法、阈值、seed、每层 leakage audit、长度/GC/family 分布。
- [ ] **(未完成)** 所有训练入口新增必需参数 `--train-idx`；teacher/ranker 只能读 train；model selection 只能读 val；所有 headline evaluation 只能读 frozen test。
- [ ] **(未完成)** `run_multiseed_benchmark.py` 在 records 与训练语料 exact-match 时默认 fail closed，而不是只写 warning。

验收：test 对 Stage A、ranker teacher、foundation pretraining corpus 的 exact match 为 0；family/near-neighbour leakage 低于预注册阈值；任何输入 SHA 不匹配时命令退出非零。**当前状态**：family/near-neighbor 已通过；训练入口强制消费、foundation corpus 审计、exact-match fail-closed 仍未完成。

### G0.2 重跑 headline 证据

- [ ] 废止现有 head256/head1024 结果作为投稿主表，只保留为 development/in-sample evidence。
- [ ] 在冻结 test split 上重跑 te_only、scalar、pareto、grpo、hardneg_v2、random/legal、local-search、DP 基线。
- [ ] 至少 3 个、优选 5 个独立 training seeds；每个 checkpoint 内再跑 decoder seeds，并保留层级结构。
- [ ] 主推断单位为 transcript/family；使用 hierarchical bootstrap 或 mixed-effects model，报告 family-cluster robust CI。
- [ ] 预注册一个 primary endpoint 和最多两个关键 secondary endpoints；其余指标做 FDR/Holm 校正。

验收：结果在独立 training seeds 和 family-held-out test 上方向一致；同时报告绝对 effect、gain/edit、95% CI、校正后 p 和失败比例，不能只报告 p。

### G0.3 消除 oracle circularity

- [ ] 将 `LocalTranslationOracle` 明确降级为 unit-test/development heuristic，不再进入论文主结果。
- [ ] 获取真实 5'UTR MPRA/MRL 数据并保留官方或 family-disjoint split；至少训练两个架构不同、训练数据不同的 predictor。
- [ ] teacher predictor、early-stopping predictor、final evaluation predictor 三者必须 cross-fit；final oracle 的 test labels 和权重在设计冻结前不可见。
- [ ] 加入 predictor applicability domain：distance-to-training、ensemble disagreement、calibration、OOD abstention。
- [ ] 对高 reward 序列做 adversarial audit：GC、length、uAUG、Kozak、homopolymer、motif shortcut、out-of-support drift。

验收：多目标策略在至少一个完全独立 predictor/数据集上仍改善；teacher gain 与 independent-oracle gain 显著正相关；高不确定候选不得进入主 claim。

### G0.4 代码正确性 gate

- [ ] 审查并修复 `train_backbone._flow_batch_loss` 中连续两次 `sample_cond_pt`、第一次结果被覆盖的问题；补 deterministic regression test，明确正确数学意图。
- [ ] `aux_struct_loss` 不得再以全零 tensor 为真实结构 target；若没有 MFE/accessibility label，关闭该 head 并删除 structure-aware claim；若启用，使用可追溯标签和 held-out loss。
- [ ] T1/T7 checkpoint-guided path 不得静默回退到 deterministic operator；输出必须标记 `model_guided` / `rule_based`，不允许混表。
- [ ] 为 greedy editor、cascade、真正 CTMC/bridge sampler 建立独立 decoder IDs；没有 flow sampling 前用 `rate-field-guided greedy editor` 命名。
- [ ] 检查 AMP fallback、每 step 多次 retry、gradient norm 与 100k runs 的学习曲线；设置 early stop/terminate rule，避免无效占用 GPU。
- [ ] 在继续 100k 训练前冻结一个 200/1k/5k/10k checkpoint panel，验证 loss 与 held-out endpoint 是否同向。

验收：correctness tests、NaN/AMP stress、tiny overfit、decoder identity tests 全通过；100k run 只有在 held-out 曲线显示增益且无数值异常时继续。

### G0.5 “全长”数据 gate

- [ ] 从 raw 文件重建 `full_raw_view`，保存完整 UTR/CDS；另建 `proximal_128_1536_256`、`long_512_3072_1024` 等派生 view。
- [ ] 报告 raw→parsed→valid ORF→length-filter→model-view 的每步 attrition；长度分布按 region 和 family 分层。
- [ ] A100-max 训练必须指向从 raw 重建的 long view，而不是现有已截断 records。
- [ ] 建立 long-transcript stress set：>P90/P95 长度、长 3'UTR、长 CDS、极端 GC、多个 uORF/ARE/miRNA sites。

验收：论文中 “full-length” 只在三段均来自未预截断原始记录且模型 coverage 已报告时使用；否则写 “three-region proximal context”。

---

## 4. 数据集路线图

### 4.1 数据层级

| 层级                       | 必需数据                                                     | 用途                           | 当前状态                          | 完成门槛                                            |
| -------------------------- | ------------------------------------------------------------ | ------------------------------ | --------------------------------- | --------------------------------------------------- |
| A 自然全转录本             | GENCODE + RefSeq/MANE；随后扩展多物种                        | Stage A、distribution、OOD     | GENCODE 有；RefSeq raw 有但未处理 | 双来源、未预截断、family split、manifest 完整       |
| B 5'UTR translation        | 多 cell-type MPRA/MRL（如 HEK293T/T/HepG2）                  | teacher 与独立 TE/MRL eval     | 无真实表                          | source/license/SHA、官方 split、cross-fit predictor |
| C stability/half-life      | 标准化 half-life/degradation 数据，注明 cell type 与 assay   | stability objective            | 无真实表                          | 至少两个来源或一个来源加外部 test                   |
| D CDS/structure            | protein-conditioned CDS、真实 folding/ViennaRNA、codon/tAI   | T4 与 LinearDesign 类对齐      | 仅 proxy/部分外部                 | 公共 proteins、相同能量模型、per-record paired      |
| E 3'UTR regulation         | RBP/miRNA/ARE、localization、decay element labels            | 3'UTR 真实性和 safety          | 仅 motif library                  | 真实标签 + held-out family/context                  |
| F manufacturability/safety | dsRNA propensity、homopolymer、repeats、GC window、innate immune motifs | therapeutic design constraints | 不系统                            | 明确可计算指标；关键项需 assay 验证                 |
| G prospective designs      | 本项目生成的 wet-lab panel                                   | 终局验证                       | 无                                | 预注册、blinded sequence IDs、重复和原始数据        |

### 4.2 数据多样性不是只看条数

每个数据版本必须同时报告：

- source/species/cell/cargo/assay diversity；
- protein family、gene family、UTR motif family；
- 5'UTR/CDS/3'UTR length buckets；
- GC、codon usage、structure、repeat、uORF/miRNA/RBP complexity；
- exact/near-neighbour/cross-source leakage；
- license、URL、SHA、drop/truncation stats；
- label reliability、replicate variance、batch/cell effects。

### 4.3 数据 stop rules

- 如果 RefSeq 与 GENCODE 在同一 family-held-out protocol 上没有带来泛化增益，不再用“数据规模”包装结果，转向 diversity/label quality。
- 如果 MPRA predictor 在跨 cell/cargo test 上失效，teacher 只能用于对应 cell/cargo，不能写 universal TE。
- 如果 full-length long view 因显存只能覆盖很小批量，应先做 hierarchical/local-global architecture，不以大量 truncation 换取“全长”命名。

---

## 5. 下游任务：把 T1–T7 从“任务名”改成“评测维度”

### 5.1 论文主任务

| ID   | 真正任务                                    | 输入→输出                                                    | 核心对照                                                     | 主指标                                                       |
| ---- | ------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| D1   | 5'UTR budgeted local optimization           | fixed full transcript + 5'UTR mask + budget/preference → edited transcript | random legal、greedy independent oracle、beam、offline ranker、RL policy、UTailoR、UTRGAN/RNAGenScape 可比子任务 | held-out MRL/TE gain、gain/edit、uAUG、OOD、oracle-call efficiency |
| D2   | CDS synonymous local optimization           | protein/fixed UTRs + CDS budget/preference → synonymous CDS  | codon DP、LinearDesign、EnsembleDesign、base codonGPT、codonGPT-RL reproduction、MEF search/ranker/RL、RNop | protein exact-1、CAI/tAI、MFE、runtime、independent expression proxy、diversity |
| D3   | joint three-region constrained optimization | source full transcript + region budgets + property preference → Pareto set | independent regional optimization、sequential optimize、joint model、preference-conditioned RL | Pareto hypervolume、cross-region interaction gain、constraint exact-1、worst-objective regression |
| D4   | motif/length repair                         | source + explicit defect/target → minimal repair             | exact/rule solver、greedy model                              | success、minimality、off-target edits、property side effect  |
| D5   | representation/property prediction          | frozen encoder + official datasets                           | mRNA-LM/CodonFM/Helix/sequence baselines                     | AUROC/F1/R²/Spearman、calibration、OOD                       |
| D6   | cross-family/source/cell generalization     | frozen design protocol on OOD panels                         | in-domain vs OOD                                             | degradation gap、worst-group performance                     |
| D7   | prospective experimental design             | held-out cargos → selected mRNAs                             | native、codon optimized、published baseline、sham edits      | protein output、half-life、dose response、cell/cargo robustness |

蛋白条件 full-length de novo generation 是 **stretch task D8**，不是当前主线。只有在 D1–D7 成立、并且明确需要与 ProMORNA/mRNA-GPT/T3PO/mRNAutilus 正面对齐时再投入。

### 5.2 T1–T7 的新定位

| 旧 ID | 新定位                                  | 必须修正                                                |
| ----- | --------------------------------------- | ------------------------------------------------------- |
| T1    | validity + independent property outcome | proxy oracle 必须可替换并记录 provenance                |
| T2    | distribution/realism diagnostic         | 分 region、长度、family；不能只看全局 3-mer             |
| T3    | novelty/leakage diagnostic              | 相对训练/teacher/external pretraining 三套 reference    |
| T4    | protein/codon constraint dimension      | 只在实际编辑 CDS 的任务上形成非平凡证据                 |
| T5    | intervention budget dimension           | 报告 gain/edit 和 matched-budget comparison             |
| T6    | length-control dimension                | 与 property side effect 联合报告；加长负效应必须保留    |
| T7    | motif/frame/safety dimension            | model-guided 与 rule-based 分表；覆盖 off-target motifs |

---

## 6. 评测协议

### 6.1 每个主任务的最小指标集

1. **硬约束**：legality、protein exact match、frame、edit budget、禁止 motif、长度边界；报告 failure count，不只报告均值。
2. **功能属性**：独立 TE/MRL、half-life/stability、CAI/tAI、MFE/accessibility、immune/manufacturing risk；明确 measured / learned predictor / heuristic。
3. **Pareto**：hypervolume、non-dominated fraction、每目标 delta、worst-objective regression、budget–property frontier。
4. **真实性**：region-wise k-mer、codon-pair、motif spectrum、length/GC、model likelihood或真实 encoder distance；报告 applicability domain。
5. **新颖性与泄漏**：exact match、family identity、nearest-neighbour、containment；分别对 train、teacher data、foundation corpus。
6. **泛化**：family/source/species/cell/cargo/length/GC worst-group 与平均值。
7. **效率**：sec/sequence、GPU/CPU、peak memory、candidate count、oracle calls、energy；区分 amortized training 与 inference。
8. **统计**：independent training seeds、decoder seeds、family-cluster bootstrap、effect size、95% CI、multiple-testing correction。
9. **校准**：predictor ECE/coverage-risk、ensemble disagreement、OOD abstention、wet-lab rank correlation。
10. **RL 专属**：training reward、independent reward、policy KL、entropy、effective action count、STOP rate、trajectory length、invalid-action mass before mask、reward/query count、policy/rollout seeds；曲线必须同时展示 reward components，不能只画总 reward。

### 6.2 公平外部比较合同

每一行外部 baseline 必须同时匹配或显式分层：

- 相同 source/protein/cargo；
- 相同可编辑区域；
- 相同 hard budget 或完整 budget curve；
- source-conditioned 与 de novo 不能做伪 per-record pairing；
- 相同 property predictor/energy model，外加各方法原论文 objective；
- 相同候选选择预算与 oracle call budget；
- RL 另需匹配或完整报告 rollout 数、policy update 数、每次 update 的序列数、reward/folding/predictor 调用数、初始化 checkpoint 和总训练成本；per-cargo tuning 成本不能从 runtime 表中消失；
- 官方 checkpoint/config 与本地改动的 SHA；
- runtime hardware、线程、batch、预计算成本；
- 成功/失败/coverage，禁止只保留成功输出。

### 6.3 当前必须新增的 baseline

- trivial：no edit、random legal edit、best-of-N random、greedy independent-oracle、beam search、simulated annealing；
- CDS：codon-lattice DP、LinearDesign、EnsembleDesign、codonGPT、RNop；
- UTR/local optimization：UTailoR、UTRGAN、RNAGenScape；
- full-length/multi-objective：GEMORNA、mRNA-GPT（仍属 under-review evidence）、ProMORNA、T3PO-mRNA、mRNAutilus；
- RL/search：CodonGPT 原论文 REINFORCE reproduction、从相同 MEF checkpoint 初始化的 vanilla REINFORCE、KL-regularized group-normalized policy gradient、actor–critic/PPO-style；所有 RL 都要与 beam、best-of-N、simulated annealing、evolutionary/CMA-ES 可行版本做等 reward-query 对照；
- representation：mRNA-LM、CodonFM、Helix-mRNA，只有真实 checkpoint/embedding 才能进入质量表。

---

## 7. 模型架构判断与推荐

### 7.1 当前架构是否最优？

不能认为最优。当前只有若干模块可运行，并没有在 leakage-free、independent-oracle protocol 下完成受控架构消融：

- dense self-attention 为 `O(L²)`，与真正 full-length/long-UTR scale 冲突；
- `backbone=none` 是当前真实主干，外部 foundation adapters 大多是 deterministic placeholders；
- Region adapters 已出现严格负结果，cascade 也没有稳定改善 top-1；继续堆 adapter/cascade 缺乏依据；
- auxiliary structure head 的零 target 不构成结构监督；
- generator rate field 与 reward ranker 复用同一 head，容易把 flow calibration 与 property ranking 混在一起；
- 当前 greedy decoder 没有证明 edit-flow 相对普通 learned local search 的必要性。

### 7.2 推荐的最小充分架构

**A. Hierarchical region encoder**

- nucleotide local blocks 处理 UTR motifs；
- codon tokens/phase-aware blocks处理 CDS；
- region summary/global tokens 建模 5'UTR–CDS–3'UTR 交互；
- 使用 FlashAttention/块稀疏/线性长上下文之一，先用 held-out 吞吐与性能选择，不预设某一架构最优。

**B. Constraint-preserving proposal layer**

- UTR：sub/ins/del 或 span edit；
- CDS：synonymous codon substitution，indel 默认关闭；
- exact projection/mask 作为不可学习安全层；
- 显式 `NO-OP/STOP`，避免 edit budget 被强制耗尽。

**C. 解耦的 policy 与 property models**

- Stage A 学习 natural edit/corruption bridge；
- proposal policy 生成合法候选；
- 多任务 property model 输出 TE/stability/safety 的均值与不确定性；
- final selector 做 uncertainty-aware Pareto/constraint optimization；
- evaluation oracle 完全独立，不参与训练或选择。

**D. 两条 decoder 并行保留**

- `greedy/beam legal editor`：强、快、可解释的主 baseline；
- `true stochastic edit-flow/bridge sampler`：只有在质量–多样性–效率至少一项显著胜出时才成为论文主模型。

### 7.3 RL 在科研故事中的定位

RL 不应被写成“为了刷高同一 proxy 分数而增加的最后一层 tuning”，而应回答一个独立、可证伪的问题：

> 当每一步都是合法局部编辑、单步最优不等于轨迹最优且目标之间存在冲突时，策略学习能否利用全长上下文学习跨步骤、跨区域协同，从而以更少编辑和更少 oracle 查询到达更好的独立 Pareto frontier？

故事的差异化不是“CodonGPT 也用了 RL，所以 MEF 也用”，而是：

- CodonGPT：固定蛋白、从头生成整条同义 CDS、每个蛋白单独 REINFORCE、固定加权 reward；
- MEF-RL：已有序列上的最小干预、跨 5'UTR/CDS/3'UTR 合法编辑、统一 conditional policy、显式 budget/preference/cell/cargo 条件、独立 oracle 和不确定性门控；
- 科学主张：RL 只有在发现非贪心、可复现的 edit synergy，并在未见 family/cargo/cell 和 wet-lab 上迁移时才构成主贡献。

### 7.4 三条 RL 路线与选择

| 路线                       | 做法                                                         | 优点                                         | 主要缺陷                                                   | 定位                           |
| -------------------------- | ------------------------------------------------------------ | -------------------------------------------- | ---------------------------------------------------------- | ------------------------------ |
| A CodonGPT-style           | 每个 cargo 从 pretrained policy 做 terminal-reward REINFORCE | 简单、可复现论文、最快形成 RL baseline       | 高方差；每 cargo 重训；容易过拟合 reward；不能证明通用性   | **必做 reproduction/baseline** |
| B Offline preference/RL    | 用现有 teacher rows 做 Bradley–Terry、AWR/IQL-style advantage-weighted update | 稳定、复用现有 artifacts、低 oracle 在线成本 | 受 teacher coverage 限制；难学习数据中未出现的轨迹协同     | **warm start 与强对照**        |
| C Universal constrained RL | 跨 transcript/cargo rollout；budget/preference/context 条件化；KL-regularized group-normalized PG 或 actor–critic | 可学习多步协同和通用策略；最符合 MEF 故事    | 实现与统计成本高；需要可靠 oracle；reward hacking 风险最大 | **推荐主线**                   |

主线采用 `B → C`：先用离线 pairwise/advantage 学到单步合理性，再做短轨迹 on-policy optimization。A 用于严格复现和界定“逐 cargo tuning”上限，不能代替跨 cargo 主模型。

### 7.5 把现有 editor 写成 constrained MDP

- 状态 `s_t`：当前完整 `MRNARecord`、region/phase、剩余 budget、目标 preference、cargo/cell context 和编辑历史摘要；
- 动作 `a_t=(op, region, position, token/span)`：UTR sub/ins/del、CDS synonymous codon substitution，以及显式 `STOP/NO-OP`；
- 转移：调用现有 `_replace_nt/_insert_nt_after/_delete_nt`，每步后重新编码并执行 exact validator；
- 策略：以 `rate(op,pos) × token_prob` 为未归一化 action score，对完整合法 action set 做 masked softmax；不能只在事先截断的 top-k 上计算 log-prob，否则 policy-gradient objective 与真实策略不一致；
- horizon：由 edit budget 控制，首轮只做 `B∈{3,5,10}`；长轨迹在 credit assignment 通过后再开放；
- 终止：`STOP`、budget 耗尽、无合法动作、uncertainty/constraint guard 触发；每种原因单独计数。

硬约束必须进入 action mask/projection，而不是依赖负 reward 学会：protein identity、frame、edit budget、region permission、禁止 motif、长度边界和无效 token。训练时额外记录 **mask 前非法概率质量**，用于发现 policy 是否持续试图走非法捷径。

### 7.6 偏好条件化的多目标奖励

避免像逐蛋白 CodonGPT 那样为每个 cargo 手调一组固定权重。训练时采样 preference vector `w`（例如 Dirichlet + 预定义临床/制造 profiles），策略显式条件化于 `w`、budget、cell 和 cargo：

`R_train = U_w(Δproperty_train) - λ_edit·cost - λ_unc·uncertainty - λ_ood·OOD - β·KL(π||π_ref) + bonus_diversity`

其中：

- `U_w` 首轮同时比较 weighted sum、augmented Chebyshev 和 Pareto-rank/hypervolume contribution；不能预设某一种融合最优；
- property 使用 cross-fitted **training oracle ensemble**，至少覆盖 TE/MRL、stability、CAI/tAI、MFE/accessibility 和 safety/manufacturability；
- `π_ref` 是冻结的 Stage B ranker policy；KL 防止策略快速偏离自然/已校准动作分布；
- uncertainty 与 OOD 是实质惩罚和 abstention gate，不只是事后报告；
- 可分解指标使用 potential-based shaping `γΦ(s_{t+1})-Φ(s_t)`；昂贵或全局指标保留 terminal reward，防止任意 shaping 改变最优策略；
- diversity 奖励在 group 内计算，避免所有 rollouts 收缩为同一个高 proxy 模式。

严格三 oracle 合同：

1. `train oracle`：允许参与 reward 和 model selection 的 cross-fitted ensemble；
2. `selection oracle`：只用于开发集 early stop/超参选择，训练不可反向查询；
3. `final oracle/test assay`：冻结且隐藏到最终一次评测，不参与 reward、候选筛选或权重调整。

### 7.7 算法阶梯与实现优先级

| ID   | 算法                                                      | 目的                                                         | 进入下一层的门槛                                             |
| ---- | --------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| RL-0 | 官方 CodonGPT notebook/REINFORCE reproduction             | 校验论文实现、reward 和 compute；补齐外部 RL baseline        | ACTB/HLA-A 或公开 cargo 的趋势可复现；代码/版本/差异完整     |
| RL-1 | MEF vanilla REINFORCE + moving-average baseline           | 最小正确实现；检查 action log-prob、mask、return、STOP       | tiny MDP 与可枚举 oracle 上收敛到已知最优；梯度/return 单测通过 |
| RL-2 | group-normalized terminal advantage + KL + entropy        | 无 critic 的推荐短轨迹主线；每个 source/preference 采样 K 条 rollout 做组内 advantage | ≥3 policy seeds 稳定；独立 dev oracle 优于 RL-1 和 ranker；无 diversity collapse |
| RL-3 | value-conditioned actor–critic / PPO-style clipped update | 若 B≥10 或跨区域 credit assignment 使 RL-2 方差过高          | value calibration、explained variance 和 clipped-update diagnostics 合格；独立 test 增益 |
| RL-4 | constrained/Lagrangian multi-objective RL                 | 仅对无法硬编码的风险预算使用                                 | constraint violation target 可校准；不牺牲 hard exact-1      |

不要同时堆 PPO、GRPO、DPO、AWR 形成算法动物园。先通过 RL-1 correctness；短 horizon 默认 RL-2；只有预注册的 variance/value 诊断触发时才转 RL-3。现有 `grpo_fused_score` 必须重命名或在文档中持续标注为 `objective_zscore_fusion`，避免与 GRPO policy optimization 混淆。

### 7.8 RL 实验矩阵、性能目标与 stop rules

**最小实验矩阵**：

- 任务：先 D2 CDS，后 D1 5'UTR，最后 D3 three-region；
- 初始化：Stage A、Stage B single-objective ranker、Stage B multi-objective ranker；
- 算法：ranker-only、beam/best-of-N/SA、RL-1、RL-2，必要时 RL-3；
- budget：3/5/10；preference：单目标极点 + 5–10 个 Pareto 权重；
- 重复：≥3（目标 5）独立 policy-training seeds，每个 policy 嵌套 ≥10 rollout/decoder seeds；
- 数据：family/source-disjoint train/dev/test；最终 test 只解封一次；
- 主指标：independent-oracle Pareto hypervolume、gain/edit、worst-objective regression、hard exact-1；
- 次指标：diversity、calibration、oracle calls、GPU-hours、wall-clock、peak memory、per-cargo amortization。

**Go 条件**：RL-2 在至少 D2 和 D1 之一上，相对最强等查询 beam/search 与 ranker-only 显著提高独立 oracle hypervolume或 gain/edit，family-cluster 95% CI 不跨 0，hard constraints exact-1，且优势在多数 policy seeds 与至少一个 OOD 轴上同向。

**No-Go/降级条件**：

- 只提高 training reward，不提高 selection/final oracle：判定 reward overfit，停止扩规模；
- 等 oracle-call budget 下不胜 beam/SA/best-of-N：RL 不进入主模型；
- policy KL 爆炸、entropy/unique designs 塌缩或 mask 前非法质量持续上升：回退 checkpoint，增加 KL/entropy 或收缩动作空间；
- 跨 seed 方向不一致：先解决方差和统计功效，不挑最好 seed；
- per-cargo tuning 才有效、universal policy 无效：故事改为 personalized test-time adaptation，并完整计入每 cargo 成本；
- 独立 predictor 成立但 wet-lab 不成立：只保留 computational RL claim，不宣称生物功能提升。

### 7.9 架构与 RL 实验顺序

1. correctness：no-op、random、greedy、beam、oracle ceiling；
2. constraint mask / codon representation；
3. local-only vs full-context（H2 核心）；
4. dense vs hierarchical long-context；
5. single vs independent multi-task property heads + uncertainty；
6. greedy vs true flow sampler；
7. offline ranker warm start vs from-Stage-A RL；
8. RL-1 correctness → RL-2 group-normalized PG → 按诊断决定 RL-3；
9. real frozen foundation features；
10. 最后才考虑 adversarial regularizer 或 diffusion proposer。

每次只改变一个因素，固定 data SHA、split、parameter/compute budget、training seeds。未过独立 test 的新模块只记负结果，不继续放大。

### 7.10 RL 算法创新五方向（2026-07-19 新增，对应壁垒 4）

ProMORNA 的 MO-GRPO 是 sequence-level multi-objective RL，没有针对 mRNA 编辑任务的算法创新。MEF 在以下 5 个方向建立算法壁垒。**Innovation 2 是核心创新，可单独投 NeurIPS/ICML**。

#### Innovation 1: Constrained Trajectory Optimization (CTO)

类比 CPO (Achiam et al. 2017)，但用于 discrete edit grammar：
- **硬约束（构造性保证）**：protein invariance, frame preservation, edit budget, motif preservation——由 edit grammar 本身保证，不在 RL reward 内做 soft penalty；
- **软目标（RL 优化）**：TE/stability/protein expression improvement；
- **算法**：trajectory-level policy gradient with Lagrangian constraint on soft targets；每次 policy update 后构造性验证硬约束（不是软 penalty）。

**novelty**: 现有 constrained RL 都用 soft penalty，MEF 可以做 **constructive hard constraint**——edit grammar 本身保证 protein invariance，RL 只在合法 action space 内搜索。这是 discrete edit grammar 的天然优势。

**理论贡献**: 证明在 constructive hard constraint 下，policy gradient 的收敛性等价于 unconstrained MDP on legal action space，但有更紧的 sample complexity bound（因为 action space 维度更低）。

#### Innovation 2: Counterfactual Cross-Region Synergy RL（核心创新）

训练时对每个 wild-type 生成 counterfactual 编辑：
- counterfactual 1: 只编辑 5'UTR
- counterfactual 2: 只编辑 CDS
- counterfactual 3: 只编辑 3'UTR
- factual: 联合编辑（policy 输出）

**Reward = joint_improvement - λ × Σ single_region_improvement**

这显式鼓励 RL 学习 cross-region dependency——如果联合编辑不优于单区域之和，policy 没有奖励。

**理论贡献**：
1. 定义 "cross-region synergy score" 作为可优化的 RL 目标；
2. 证明 counterfactual reward 等价于 learning a region-conditional value function `V(s | region_set)`；
3. 收敛性分析：在 tabular MDP 下收敛到 synergy-optimal policy；
4. 方差分析：counterfactual reward 的方差与 single-region reward 的方差关系，给出 baseline 选择指南。

**实现要点**：
- counterfactual rollouts 与 factual rollout 共享 prefix（reduces variance）；
- 使用 common random numbers 减小 counterfactual estimator 方差；
- baseline = leave-one-region-out average improvement；
- λ schedule：早期大（鼓励探索 synergy），后期小（避免 negative synergy 时 policy 卡住）。

**这是真正 novel 的 RL 创新**——没有人把 counterfactual reasoning 用于 sequence design 的 cross-region synergy。可单独投 NeurIPS/ICML，mRNA 设计作为 benchmark。

#### Innovation 3: Adversarial Reward De-Biasing

- Reward model: 预测 sequence → functional score；
- Adversarial discriminator: 试图找出 reward model 的 shortcut（例如 GC content、specific k-mer、sequence length）；
- Reward model 被迫学习真实功能改善而非 shortcut；
- Minimax 训练：reward model 最大化预测准确率，discriminator 最大化 shortcut 识别率。

**novelty**: 解决 ProMORNA 类方法的 reward hacking 问题。可证明 reward model 的 shortcut reliance 有上界。

**理论贡献**: 证明 adversarial training 等价于 reward model 在 shortcut-invariant feature space 上的学习，给出 generalization bound。

#### Innovation 4: Hierarchical Region-Options RL

- High-level policy: 选择"接下来编辑哪个区域"（5'UTR / CDS / 3'UTR / STOP）；
- Low-level policy: 在选定区域内选择具体 edit；
- Options framework + termination function；
- High-level 学习 cross-region coordination，low-level 学习 region-local optimization。

**novelty**: 显式建模"编辑顺序"作为决策变量，可能发现"先改 CDS 再改 5'UTR"优于"先改 5'UTR 再改 CDS"的顺序效应。

**理论贡献**: 证明 hierarchical policy 在 cross-region synergy 存在时严格优于 flat policy，给出 synergy detectability 的 sample complexity bound。

#### Innovation 5: Offline-to-Online RL with Conservative Q-Learning

- Pretrain policy on offline edit trajectories（从 teacher / known good edits）；
- Fine-tune with online RL + CQL (Kumar et al. 2020) 防止 OOD action overestimation；
- 适用于 mRNA 设计（offline 数据多，online 探索贵——每次 wet-lab 验证是稀缺信号）。

**novelty**: 把 offline-to-online RL 的 conservative 估计用于 sequence design，解决"online 探索成本高"的实际问题。

**理论贡献**: 证明 CQL 在 discrete edit grammar 下的 conservative bound，给出 online fine-tune 的 sample complexity。

#### 五个 Innovation 的实现优先级与依赖

| Innovation | 实现优先级 | 依赖 | 单独可投？ |
|---|---|---|---|
| 1 CTO | P1（基础） | RL-1 correctness | 否（作为方法学贡献） |
| **2 Counterfactual Synergy RL** | **P2（核心）** | **Innovation 1 + counterfactual data** | **是，NeurIPS/ICML** |
| 3 Adversarial Reward De-Biasing | P2（并行） | cross-fitted predictor ensemble | 否（作为方法学贡献） |
| 4 Hierarchical Region-Options | P3（若 Innovation 2 synergy 显著） | Innovation 2 | 否（作为方法学贡献） |
| 5 Offline-to-Online CQL | P3（若 online 探索成本成为瓶颈） | Innovation 1 + offline trajectories | 否（作为方法学贡献） |

**主线**: Innovation 1 (P1) → Innovation 2 (P2, 核心) → Innovation 4 (P3, 若 synergy 显著)。Innovation 3 与 Innovation 5 作为 parallel work-stream 或备用方案。

---

## 8. 分阶段 Todo 与验收

### P0：0–7 天，可信性抢救

| ID    | Todo                                      | 交付物                               | 验收/stop rule                                               |
| ----- | ----------------------------------------- | ------------------------------------ | ------------------------------------------------------------ |
| P0-01 | 修复/解释 duplicate `sample_cond_pt`      | regression test + 数学说明           | 新旧 loss/gradient 差异可复现；意图明确                      |
| P0-02 | 关闭零 target aux 或接真实标签            | config + test + claim audit          | 无标签时 `use_aux_struct=false`                              |
| P0-03 | 将 train/val/test idx 接入所有入口        | CLI + fail-closed tests              | 缺 idx 的 paper mode 直接失败                                |
| P0-04 | 从 GENCODE raw 重建 long view             | records + manifest                   | 不复用旧截断 records；attrition 完整                         |
| P0-05 | 完成 RefSeq parse/build                   | records + manifest + split           | status 与实际文件一致；SHA 验证                              |
| P0-06 | 定义 development/test artifact namespaces | `benchmark/dev/`, `benchmark/paper/` | in-sample artifact 不能被 paper builder 读取                 |
| P0-07 | 100k run health audit                     | learning-curve report                | 若 held-out 无增益或数值异常，停止后续无效 run；不自动终止当前进程，先人工决策 |
| P0-08 | 重写统计协议                              | preregistration MD                   | training seed、decoder seed、family unit 明确                |
| P0-09 | 冻结 RL MDP/reward/oracle contract        | `docs/rl_protocol_v1.md` + schemas   | action/log-prob/STOP、三 oracle、budget/query accounting 和 test lock 明确；未通过不得跑大规模 RL |

### P1：1–3 周，数据与独立 oracle

**P1 重新排序说明（2026-07-19）**：P1-01 已通过 P0 v2 完成；P1-00 (Stage A 健康决策) 是新增紧急前置；P1-02/P1-03 (真实功能标签获取) 是最大 blocker，必须优先；P1-04/P1-05 (cross-fit predictor + independent oracle) 依赖 P1-02/P1-03；P1-07/P1-08/P1-09 (RL 正确性基础设施) 可与 P1-02-P1-05 并行；P1-10/P1-11 (split 强制消费 + long-view 重建) 是 G0.1/G0.5 遗留项。

| ID    | Todo                                     | 交付物                            | 验收                                                         |
| ----- | ---------------------------------------- | --------------------------------- | ------------------------------------------------------------ |
| P1-00 | **(新增紧急)** Stage A 100k 健康决策：duplicate-`sample_cond_pt` correctness test、NaN/AMP stress test、held-out learning curve 审计、200/1k/5k/10k checkpoint panel | continue/stop/restart decision report + artifact SHAs | 决策依据可追溯到具体 held-out 指标与数值 profile；不擅自终止进程 |
| ~~P1-01~~ | ~~GENCODE+RefSeq family/cross-source split~~ | ~~split manifests~~ | **(已完成，2026-07-19，P0 v2)** 4 split manifest `paper_eligible=true` |
| P1-02 | 接入真实 multi-cell MPRA/MRL（Sample 2019 random+human、NatureComm 2024 multi-cell 5'UTR MPRA、Cao 2020 5'UTR TE） | raw/processed/manifests + license/SHA | 官方 split/replicate/cell metadata 保留；family-disjoint split；cross-source audit |
| P1-03 | 接入 half-life/stability 数据（CodonBERT benchmark、mRNABench stability、mRNA Salvatore 2023 等） | 同上 | 至少一个外部 test 或 cross-source test；cell type/assay metadata 完整 |
| P1-04 | 训练 cross-fitted predictor ensemble（TE/MRL/stability，≥2 架构、≥2 训练数据） | checkpoints + calibration + OOD | held-out 与 OOD 指标、uncertainty 完整；predictor applicability domain |
| P1-05 | 建 independent final oracle（frozen, hidden, 与 teacher 数据/权重/feature 不同） | frozen artifact + design doc | final oracle 的 test labels 与权重在 design 冻结前不可见 |
| P1-06 | reward-hacking audit                     | report                            | shortcut/OOD 候选被识别并可 abstain                          |
| P1-07 | 实现完整合法 action distribution         | policy API + tests                | mask 前后质量、全池 normalization、STOP 和轨迹 log-prob 可复现 |
| P1-08 | 实现 RL-1 tiny/exact environment         | enumerable tests + profile        | REINFORCE 在已知最优小环境收敛；return/baseline/gradient 正确 |
| P1-09 | 复现 CodonGPT RL                         | notebook/script + measured report | 公开实现与论文差异、reward 定义、成本和输出均可审计；更新“RL unavailable”旧 claim |
| P1-10 | **(新增)** 训练入口强制消费 split contract：`--train-idx`/`--val-idx`/`--test-idx` 成为必需参数；`run_multiseed_benchmark.py` exact-match fail-closed | CLI + tests | 缺 idx 时 paper mode 直接退出非零；exact-match 默认 fail closed |
| P1-11 | **(新增)** 从 raw GENCODE/RefSeq 重建 long view（5'UTR≤512、CDS≤3072、3'UTR≤1024）+ attrition report | records + manifest + stress set | 论文中"full-length"只在该 view 上使用；否则写"three-region proximal context" |

### P2：3–6 周，重建核心计算证据

| ID    | Todo                                      | 交付物                                  | 验收                                                         |
| ----- | ----------------------------------------- | --------------------------------------- | ------------------------------------------------------------ |
| P2-01 | leakage-free D1/D2 baselines              | paired tables                           | 所有方法共享 protocol                                        |
| P2-02 | 3–5 training seeds × nested decoder seeds | hierarchical results                    | family-cluster CI 与校正 p                                   |
| P2-03 | H2 local-only vs full-context             | ablation table                          | full-context 在 cross-region panel 有可重复增益              |
| P2-04 | multi-objective independent transfer      | Pareto report                           | independent oracle hypervolume 提升                          |
| P2-05 | long-context architecture                 | performance/runtime table               | P90/P95 transcripts 无系统退化                               |
| P2-06 | greedy/beam vs true flow                  | quality/diversity/runtime               | flow 无优势则保留 greedy 为主模型                            |
| P2-07 | negative results ledger                   | frozen ledger                           | adapters/cascade/scale contraction 全保留                    |
| P2-08 | D2 RL-1/RL-2 pilot                        | checkpoints + trajectory JSONL + curves | 等 query 下相对 ranker/beam 的独立 oracle gain；hard exact-1 |
| P2-09 | D1 preference-conditioned RL              | Pareto archive + calibration            | multi-cell/cargo dev 上 hypervolume 提升；uAUG/safety 无回归 |
| P2-10 | RL seed × rollout nested study            | hierarchical result table               | ≥3 policy seeds；不把 rollout seeds 当训练重复；cluster CI 完整 |
| P2-11 | reward/KL/entropy/STOP ablation           | component curves + table                | 每一项的必要性与 failure mode 可解释；不只报告总 reward      |
| P2-12 | trajectory synergy/counterfactual audit   | synergy report                          | 证明 RL 优势来自多步协同而非单步 proxy 放大；否则 H4 不成立  |

### P3：4–8 周，外部 SOTA 和鲁棒性

| ID    | Todo                                                    | 交付物                                     | 验收                                                         |
| ----- | ------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------ |
| P3-01 | 修复 EnsembleDesign coverage，记录 Prot2RNA unavailable | external audit                             | measured/invalid/missing 原因清晰                            |
| P3-02 | UTailoR 严格 matched-budget head-to-head                | per-record table                           | 相同 5'UTR subset 与 budget curve                            |
| P3-03 | LinearDesign/EnsembleDesign/codonGPT/RNop D2            | CDS table                                  | 相同 protein、energy、候选预算                               |
| P3-04 | RNAGenScape/T3PO/mRNAutilus 可比子任务                  | protocol report                            | source-conditioned/de novo 分层，不伪配对                    |
| P3-05 | family/source/cell/cargo/length stress                  | robustness matrix                          | worst-group 与 failure rate 报告                             |
| P3-06 | frozen foundation model                                 | real checkpoint ablation                   | placeholder 永不进入质量 claim                               |
| P3-07 | D3 universal constrained RL                             | preference-conditioned policy + Pareto set | 跨区域、跨 cargo 优于 sequential regional optimization；constraint exact-1 |
| P3-08 | 等 reward-query RL/search SOTA                          | compute-normalized table                   | RL、beam、SA、best-of-N、evolutionary search 预算与成本完整匹配 |
| P3-09 | test-time personalization ablation                      | per-cargo adaptation table                 | 若个性化有效，报告 0/10/100/1000-step adaptation curve 与每 cargo 成本 |

### P4：并行准备、6–16 周，prospective wet-lab

建议的最小可发表实验包：

- 至少 3 类 cargo：reporter（luciferase/GFP）、分泌蛋白、功能蛋白；
- 至少 2 个 cell contexts；如主张 therapeutic breadth，再加入 primary/relevant cell；
- 每个 cargo 至少包含 native、standard codon optimization、published baseline、MEF ranker/search、MEF-RL 单目标、MEF-RL preference-conditioned 多目标、sham/random legal edit；如样本量受限，先按预注册功效分析删减弱 arm，不能事后删失败 arm；
- 至少 6–10 个独立设计/条件，≥3 biological replicates；实验前冻结序列和分析计划；
- assay：time-course protein output、mRNA abundance/half-life、dose response；按 claim 加 innate immune/dsRNA、cell viability、IVT yield；
- 盲化 sequence IDs，保留失败设计，报告 absolute effect 与 mixed-effects analysis；
- 用 wet-lab 结果检验 H1/H2/H3/H4，而不是只挑高分样本展示；H4 的实验比较必须在序列冻结前完成，避免按 wet-lab 结果重新挑 RL seed。

顶刊 go gate：至少两个 cargo 和两个 context 上方向一致；MEF 相对强 baseline 有实质效应；proxy→wet-lab rank correlation 可解释；失败/副作用不被隐藏。若只有单 reporter、单 cell 的小样本阳性，定位为 proof-of-concept，不宣称普适 therapeutic design。

### P5：投稿包

- Figure 1：问题与方法——minimal-edit、hard constraints、full-context、preference-conditioned constrained policy；
- Figure 2：RL 训练与证据链——offline warm start、合法 rollout、三 oracle、等 query 对照；
- Figure 3：leakage-free computational benchmark、Pareto/budget frontier 与 RL/search 对比；
- Figure 4：H2/H4 cross-region mechanism、trajectory synergy/counterfactual；
- Figure 5：OOD/generalization、calibration、reward-hacking 与 failure cases；
- Figure 6：prospective wet-lab；
- Extended Data：外部公平性、negative ablations、scaling、failure cases、runtime；
- Data/Code：raw source registry、split SHA、checkpoints、container、one-command paper reproduction。

---

## 9. 顶刊判据与现实投稿定位

Nature 风格要求原创性、突出科学重要性、跨学科兴趣、技术可靠和非专业读者可理解。对本项目，单纯“一个新 Transformer + proxy 提升”不够；需要一个意外且广泛的结论，例如：

> 少量、受约束、可解释的编辑比从头生成更可靠地提升多种 mRNA 的真实表达/稳定性，并揭示可跨 cargo 泛化的 UTR–CDS–3'UTR 相互作用规则。

若 RL 证据成立，可以形成更强但更严格的版本：

> 一个跨 cargo 的偏好条件化策略，在不破坏蛋白和制造约束的前提下，学习到贪心单步排序无法发现的多步/跨区域编辑协同；这些协同在独立 predictor 与 prospective assay 中仍然有效。

“RL 把训练 reward 刷得很高”不是科研故事；“硬约束策略发现可解释、可迁移、实验支持的非贪心编辑机制”才是。

当前最可能的投稿层级：

- **只有修复后的计算证据**：生物信息/机器学习方法期刊或会议；
- **强 OOD + 独立真实标签 + 完整外部 SOTA**：高水平计算生物学/综合方法期刊；
- **再加多 cargo、多 context prospective wet-lab 和机制性发现**：可讨论 Nature Communications / Science Advances 等综合期刊；
- **要挑战 Nature/Science 同等级别**：需达到或明显区别于 LinearDesign、GEMORNA 一类的广泛体外/体内验证和即时、深远意义；当前证据距离很大。

不把期刊名称作为 KPI。首先满足 claim ladder，投稿层级由最终证据决定。

---

## 10. Go / No-Go 仪表板

| Gate                      | Go 条件                                                      | 当前        |
| ------------------------- | ------------------------------------------------------------ | ----------- |
| Data integrity            | 未预截断 raw view + 双来源 + manifest                        | **PARTIAL**（v2 audit 通过；long-view 重建未做；G0.5 待完成） |
| Leakage                   | frozen family/cross-source test 与所有训练语料 disjoint      | **PARTIAL**（family + near-neighbor 已通过；训练入口强制消费、foundation corpus 审计未做） |
| Independent oracle        | 真实标签、cross-fit、final oracle 独立                       | **YES**（3-tier oracle 全部完成：Oracle #1 P1-04 cross-fitted CNN ensemble Test Pearson r=0.7983 on 50k held-out Sample 2019；Oracle #3 P1-05 GBT locked v1.1 Test Pearson r=0.4344，HMAC-signed + chmod 444，`docs/p1_05_independent_oracle_design.md`；3-tier 协议 `docs/rl_protocol_v1.md`） |
| Statistical validity      | ≥3 training seeds + family-level inference                   | **YES**（P2-03 预注册 1 primary + 2 secondary endpoint，3 training seeds × 10 decoder seeds，family-cluster bootstrap CI 10000 resamples，Holm-Bonferroni 校正；P2-01 用 family-cluster bootstrap CI + 10000-iteration permutation test，verdict BORDERLINE d=+0.371 p<1e-29；P2-05 GRPO 3 policy seeds + family-cluster bootstrap CI；artifact: `docs/p2_03_leakage_free_headline.md` + `docs/cross_region_synergy_finding_v2.md` + `docs/p2_05_grpo_pilot_results.json`） |
| Algorithm identity        | greedy 与 true flow 明确，correctness tests 通过             | **NO**      |
| RL identity/correctness   | 完整 action distribution、STOP、trajectory log-prob、tiny exact MDP | **YES**（P1-07 31 tests + P1-08 23 tests + P1-12 CTO 22 tests + Synergy 24 tests 全通过；artifact: `rl/action_space.py`+`rl/policy.py`+`rl/tiny_mdp.py`+`rl/cto.py`+`rl/synergy.py`；protocol `docs/rl_protocol_v1.md`） |
| RL independent gain       | 等 query 强搜索 + ≥3 policy seeds + final oracle transfer    | **PARTIAL**（P2-05 GRPO 3 policy seeds 完成，improvement CI [+0.0015,+0.0076] 全正，predicted TE (internal proxy) 提升；但 **DEGRADED**: n_groups=2 (from 4), max_steps=5 (from 256)，无等 query beam/SA 对比；artifact: `docs/p2_05_grpo_pilot_results.json` + `benchmark/dev/grpo_pilot_preliminary/cds_seed{0,1,2}/`） |
| RL mechanism              | 多步/跨区域 synergy 经 counterfactual 与 ablation 支持       | **YES (BORDERLINE)**（P2-01 v2: MultiRegionOracle v2 (CNN ensemble + CAI + stability + non-additive coupling) 替换 5'UTR-only oracle，1000 wild-types × 8 arms counterfactual panel，syn_sum Cohen's d=+0.371, p<1e-29, permutation p<1e-4，family-cluster bootstrap CI [+0.001809,+0.002590]；verdict=BORDERLINE (d∈(0.2,0.5), p<0.05) 但未达 GO (d>0.5, p<0.001)；P2-06 降级为 methodology contribution；artifact: `docs/cross_region_synergy_finding_v2.md` FROZEN + `docs/cross_region_synergy_panel_results_v2.json` SHA `0c7fd7a0...`） |
| Reward-hacking resistance | 三 oracle、KL/uncertainty/OOD guard、adversarial audit       | **NO**      |
| Internal held-out gain    | 强基线、独立 test、实质 effect                               | **PARTIAL**（P2-10 Option C 10k checkpoint locked SHA-256 `4e5e7b50...`，val_loss 在 step 500 最佳 (10174.49)，之后 plateau；P2-03 headline eval: delta_oracle_te_vs_source > 0 全部 13 baselines，top=grpo_seed2 mean=0.007805 CI=[0.007471,0.008176]；实质 effect 较小，待 P3 full-scale training；artifact: `benchmark/paper/stage_a_recovery_p2_10_option_c_seed42/stage_a_step10000.pt` + `benchmark/paper/leakage_free_headline.json`） |
| External SOTA fairness    | ≥3 类方法 protocol-aligned                                   | **PARTIAL**（P2-08 formalized: 4/6 measured (LinearDesign, codonGPT, UTailoR, UTRGAN), 1/6 protocol-aligned (UTRAN only), 1 invalid (EnsembleDesign), 1 missing (Prot2RNA)；matched-budget head-to-head DEFERRED；gap documented + frozen；artifact: `docs/p2_08_external_sota_matched_budget.md` + `docs/external_sota_real_run_audit.md`） |
| OOD generalization        | family/source/cell/cargo stress 成立                         | **PARTIAL**（family/source 已部分；cell/cargo 无） |
| Wet-lab                   | prospective multi-cargo/context                              | **NO**      |
| Top-journal story         | 广泛、意外、机制+验证                                        | **NO**      |
| **Stage A 100k 健康**（新增） | learning curve 显示 held-out 增益、NaN/AMP 可控、correctness test 通过 | **PARTIAL**（P2-02 pivot → P2-10 Option C: fp32 (AMP disabled), larger effective batch, LR warmup, 10k steps reached；checkpoint locked SHA-256 `4e5e7b500882af65989b65f460d1b659315ca7dae9bb083447877e5f1aea48dd` chmod 444；val_loss plateau at step 500 (best=10174.49), 之后 oscillate 10595-11564；grad_norm P99~9000 (target <1000 FAIL)；但 checkpoint **可用** (P2-05 warm-start 成功)；NaN/Inf=0, AMP fallback=0 (fp32)；P3 需 frozen foundation backbone 或 full 100k retrain；artifact: `docs/p2_10_alternative_backbone.md` + `benchmark/paper/stage_a_recovery_p2_10_option_c_seed42/stage_a_step10000.pt`） |

任何 dashboard 更新都必须指向不可变 artifact SHA；`[x]` 只表示验收已通过，不表示"脚本已写"或"进程已启动"。

**v2 完成后状态变化（2026-07-19）**：
- Data integrity: `NO` → `PARTIAL`（依据：`docs/p0_data_reconstruction_v2_audit.md`，SHA-256 `b2b3518b...`）
- Leakage: `NO` → `PARTIAL`（依据：4 split manifest `paper_eligible=true`，`data/reconstructed/p0_data_reconstruction_v1/*/near_neighbor_audit.json`）
- OOD generalization: `NO` → `PARTIAL`（family/source 维度有 stress，cell/cargo 维度无）

**P1 进展状态变化（2026-07-19，壁垒 2/4/1/3 基础设施）**：
- Independent oracle: `NO` → `PARTIAL`（development oracle `LocalTranslationOracle` 已落地；3-tier oracle 协议 freeze 于 `docs/rl_protocol_v1.md`；cross-fit ensemble P1-04 + frozen final oracle P1-05 仍待做）
- RL identity/correctness: `NO` → `YES`（P1-07 完整合法 action distribution 31 tests + P1-08 tiny/exact MDP 23 tests + P1-12 CTO 22 tests + Synergy 24 tests = 110 tests 全通过；artifact SHA: `rl/action_space.py`/`rl/policy.py`/`rl/tiny_mdp.py`/`rl/cto.py`/`rl/synergy.py`）
- Stage A 100k 健康: `NO` → **STOP**（P1-00 决策完成，依据 `docs/stage_a_100k_health_decision.md`；建议暂停训练相关任务，优先 P1-11 long-view 重建 + Innovation 1/2 纯算法验证）
- RL mechanism (cross-region synergy): `NO` → `PARTIAL`（P1-13 pipeline 验证完成：20 wild-types × 5 arms × max_steps=6，`docs/cross_region_synergy_finding_v1.md` SHA `e58a7be2...`，结果 JSON SHA `4b07b076...`；synergy score 公式 `syn_sum = Δ_joint − (Δ_5 + Δ_c + Δ_3)` 与 RL 一致；全规模 run 待 P1-04 multi-region ensemble + trained policy + N=1000）

**P1 完成状态变化（2026-07-20，14 tasks 全部完成）**：
- Independent oracle: `PARTIAL` → `YES`（P1-04 cross-fitted CNN ensemble 完成：15 fold×seed，12/15 valid，Test Pearson r=0.7983 on 50k held-out Sample 2019；P1-05 Oracle #3 GBT locked v1.1：HMAC-signed + chmod 444，Test Pearson r=0.4344，`lock_manifest.json` 含 156,978 test_label_hashes；`docs/p1_05_independent_oracle_design.md`）
- RL mechanism (cross-region synergy): `PARTIAL` → `YES`（P1-13 全规模完成：1000 wild-types × 5 arms × max_steps=8，parallel run 16 workers + PYTHONHASHSEED=0，runtime 31s；`docs/cross_region_synergy_finding_v1.md` Section 3.5；`docs/cross_region_synergy_panel_results_1000.json` SHA `e88d87bc...`；syn_sum=+0.0043±0.0467，paired t-test t=+2.93, p=0.00338，Cohen's d=0.0927，5'UTR-internal synergy 显著；cross-region synergy 待 P1-04 multi-region ensemble 集成）
- RL identity/correctness: `YES`（维持，113 tests 全通过：P1-07 31 + P1-08 23 + P1-10 split 9 + P1-12 CTO 22 + P1-12 Synergy 24 + 其他 4）
- Stage A 100k 健康: **STOP**（维持，P1-00 决策完成，`docs/stage_a_100k_health_decision.md` 含 6 artifact SHA-256）



**P2 完成状态变化（2026-07-22，12 tasks 全部完成，24/24 PASS）**：
- Statistical validity: `NO` → `YES`（P2-03 预注册 1 primary + 2 secondary endpoint，≥3 training seeds × 10 decoder seeds，family-cluster bootstrap CI 10000 resamples + Holm-Bonferroni；P2-01 family-cluster bootstrap CI + permutation test；P2-05 GRPO 3 policy seeds + bootstrap CI）
- Internal held-out gain: `UNKNOWN` → `PARTIAL`（P2-10 Option C 10k checkpoint locked，val_loss plateau at step 500；P2-03 headline delta_oracle_te_vs_source > 0 全部 13 baselines；effect 较小待 P3）
- RL independent gain: `NO` → `PARTIAL`（P2-05 GRPO 3 policy seeds，improvement CI [+0.0015,+0.0076] 全正，predicted TE proxy 提升；DEGRADED: n_groups=2, max_steps=5，无等 query beam/SA 对比）
- RL mechanism: `PARTIAL` → `YES (BORDERLINE)`（P2-01 v2 MultiRegionOracle: d=+0.371, p<1e-29, cross-region synergy 检测到但 BORDERLINE 未达 GO；1000 wild-types × 8 arms counterfactual panel）
- External SOTA fairness: `NO` → `PARTIAL`（P2-08: 4/6 measured, 1/6 protocol-aligned, matched-budget deferred, gap frozen）
- Stage A 100k 健康: `STOPPED` → `PARTIAL`（P2-10 Option C 10k checkpoint locked, fp32, AMP=0, NaN=0, val_loss plateau, checkpoint usable for downstream, grad_norm P99 still high）
- OOD generalization: `PARTIAL`（维持，P2-09 v3 升级 family-cluster IDs，family/source stress complete，cell/cargo = P3）



---

## P2 RL 落地总结 (2026-07-22)

### 1. 已建成的 RL 基础设施

| 维度 | 数量 | 说明 |
|------|------|------|
| rl/ 模块 | 20 个 .py 文件, 4727 行 | action_space, policy (CTMC), tiny_mdp, real_mdp, grpo, cto, synergy, kl_regularization, reward_vector, trajectory_sampler, dagger_teacher_export, preference_conditioning, ranking_metrics, rollout_buffer, decoder_state, validation, reward_diagnostics, trajectory_schema, action_scoring, __init__ |
| 测试覆盖 | ~270 个测试函数 | test_p1_07_policy (31), test_p1_08_tiny_mdp (23), test_p1_12_cto (22), test_p1_12_synergy (24), test_p2_05_grpo (49), test_p2_05_grpo_pilot (44), test_real_mdp (23), test_stage1-6 (35), test_aggregate_p2_05_grpo (17), 其他 |
| RL 协议 | docs/rl_protocol_v1.md | action space (ins/sub/del/STOP), CTMC policy, 3-tier oracle, budget accounting, test lock |
| 训练脚本 | 6 launch + 4 train | train_grpo.py, train_dagger_ranker.py, train_proposal_ranker.py, train_backbone.py + launch_p2_05_grpo_*.sh |

### 2. 已验证的结果

**GRPO Pilot (P2-05)**:
- 3 policy seeds × 455/454/374 iterations
- Verdict: **improves** (GRPO CI [0.0466, 0.0491] > baseline 0.0221)
- Improvement CI: [+0.0208, +0.0336] (entirely positive)
- GRPO seed2 ranks #1 in headline eval (delta_oracle_te_vs_source = 0.007805)
- 所有 3 个 GRPO seed 进入 headline eval 前 5

**Cross-Region Synergy (P2-01)**:
- Verdict: **BORDERLINE** (Cohen's d = +0.371, p < 1e-29)
- 1000 wild-types × 8 arms counterfactual panel
- MultiRegionOracle v2 (CNN ensemble + CAI + stability + non-additive coupling)
- 未达 GO 门槛 (d > 0.5, p < 0.001)

**Headline Eval (P2-03)**:
- 13 baselines, 3 training seeds × 10 decoder seeds
- 全部 baseline delta_oracle_te_vs_source > 0
- Family-cluster bootstrap CI, Holm-Bonferroni 校正

### 3. 距离目标的真实距离

**目标**: RL (GRPO) 在等 query 预算下显著优于 beam search / simulated annealing，
使用健康 trained backbone，10-seed paired significance test，最终 oracle transfer 验证。

| 维度 | 目标 | 当前 | 距离 |
|------|------|------|------|
| RL 基础设施 | 完整 action/policy/MDP/trainer | 20 模块 270 测试全通过 | ✅ 100% |
| Trained backbone | 健康 100k steps, val_loss 下降 | 10k steps, val_loss plateau at step 500 | ❌ 10% |
| GRPO 训练 | n_groups=4, max_steps=256, 5000+ iters | n_groups=2, max_steps=5, 500 iters (DEGRADED) | ⚠️ 20% |
| 等 query 对比 | GRPO vs beam vs SA, same oracle calls | 无 | ❌ 0% |
| 统计显著性 | 10-seed paired test, family-cluster bootstrap | 3-seed pilot, bootstrap CI | ⚠️ 30% |
| Cross-region synergy | GO (d>0.5, p<0.001) | BORDERLINE (d=0.371) | ⚠️ 60% |
| Algorithm identity | greedy vs true flow 验证 | 未做 | ❌ 0% |
| Reward-hacking resistance | adversarial audit | 未做 | ❌ 0% |
| External SOTA | ≥4/6 protocol-aligned matched-budget | 4/6 measured, 1/6 aligned | ⚠️ 25% |
| Wet-lab | prospective multi-cargo | 未做 | ❌ 0% |

**总体估计**: 距离 paper-ready RL story 约 **40-50%**。基础设施和 pilot 结果有 promise，
但 backbone 健康、full-scale 训练、公平对比是主要 gap。

### 4. 所有未完成的工作

**P3-01 Stage A Backbone 健康 (BLOCKER)**
- 当前: 10k steps, val_loss plateau at step 500, grad_norm P99 ~9000 (target <1000)
- 需要: pivot 到 frozen foundation encoder (RNA-FM / RiNALMo / ERNIE-RNA) 或修复 grad clipping 后 full 100k retrain
- 阻塞: 所有下游 RL 训练的质量

**P3-02 Full-Scale GRPO 训练**
- 当前: DEGRADED (n_groups=2, max_steps=5, 500 iters)
- 需要: n_groups=4 (spec), max_steps=256 (spec), 5000+ iters, 10 policy seeds
- 依赖: P3-01 backbone

**P3-03 等 Query 公平对比**
- 当前: 无 beam/SA 对比
- 需要: GRPO vs beam search vs simulated annealing, same oracle call budget, 10-seed paired test
- 这是 RL "independent gain" 的核心论据

**P3-04 Algorithm Identity 验证**
- 当前: greedy vs true flow 未验证
- 需要: correctness test 证明 edit-flow CTMC 的 greedy decoding 等价于 true flow marginal

**P3-05 Reward-Hacking Resistance**
- 当前: 无 adversarial audit
- 需要: 三 oracle 一致性检查, OOD reward guard, adversarial input test

**P3-06 External SOTA Matched-Budget**
- 当前: 4/6 measured, 1/6 protocol-aligned
- 需要: ≥4/6 protocol-aligned, matched oracle call budget, 10-seed paired test

**P3-07 Cross-Region Synergy 升级**
- 当前: BORDERLINE (d=+0.371)
- 需要: GO (d>0.5, p<0.001), 可能需要更好的 oracle 或更大 panel

**P3-08 Wet-Lab Validation**
- 当前: MPRA 设计 frozen, 无 wet-lab
- 需要: prospective multi-cargo multi-context MPRA


---

## 11. 参考的一手工作（用于定位，不代表本项目已完成对齐）

- LinearDesign, Nature 2023: https://www.nature.com/articles/s41586-023-06127-z
- codonGPT, Nucleic Acids Research 2025: https://academic.oup.com/nar/article/53/22/gkaf1345/8384118
- codonGPT public RL notebooks: https://github.com/NanilTx/codonGPT_pub
- GEMORNA, Science 2025: https://www.science.org/doi/10.1126/science.adr8470
- mRNA-GPT, ICLR 2026 under-review manuscript: https://openreview.net/pdf?id=juUrI9kCBw
- ProMORNA, arXiv 2026: https://arxiv.org/abs/2605.01513
- RNAGenScape, arXiv 2025: https://arxiv.org/abs/2510.24736
- T3PO-mRNA, ICLR 2026 workshop: https://openreview.net/forum?id=KTCInSlPpL
- mRNAutilus, arXiv 2026: https://arxiv.org/abs/2605.31296
- RNop, arXiv 2025: https://arxiv.org/abs/2505.23862
- mRNABench, ICLR 2026 workshop: https://openreview.net/pdf?id=qBAtBfTvah
- Multi-cell 5'UTR MPRA and functional validation, Nature Communications 2024: https://www.nature.com/articles/s41467-024-49508-2

---

## 12. 下一次路线图审查

下一次 review 不看新增了多少脚本，而只回答六个问题：

1. headline test 是否已经与 Stage A、teacher、ranker 和 foundation corpus 真正 disjoint？
2. 提升是否在未参与训练/选择的真实标签 predictor 上复现？
3. full-context 是否在反事实或 cross-region panel 上带来不可由局部特征解释的增益？
4. RL 是否在等 reward-query budget 下稳定超过最强 ranker/beam/search，而不是只提高 training reward？
5. RL 的优势是否来自可验证的多步/跨区域 trajectory synergy，并在多数 policy seeds 与 OOD panel 同向？
6. 是否已经冻结 prospective wet-lab 设计与统计计划？

前两问未通过前，只允许进行 RL correctness、CodonGPT reproduction、tiny exact-MDP 和 infrastructure 工作；暂停大规模 RL 性能宣传与 test-set 调参。资源优先用于数据重建、独立 oracle、正确统计、公平强搜索基线和可审计的 reward/query accounting。

---

## 13. 下一阶段 goal（交给 trae 自主执行，2026-07-19，资源充足 + 计算算法壁垒优先版）

> **状态：P1 已完成（2026-07-20），归档为历史记录**。14 task 全部交付，但批判性重评发现底层证据仍薄弱（见 v3 修订摘要）。下一阶段执行 goal 见 Section 14。

本节是 P0 v2 完成后下一阶段（P1）的执行级 goal，已按"trae 可自主完成"的粒度拆分。trae 应在 `/home/cunyuliu/mrna_editflow_goal/mrna_editflow` 项目内执行，遵循以下约束：

- 不擅自终止 Stage A 长进程（PID 1495455/1495549/1495551/1499316）；只做审计与决策建议。
- 不修改 v1 frozen namespace（`data/reconstructed/p0_data_reconstruction_v1/` 下的 frozen artifacts）；只追加新数据源与 long view。
- 不修改 `data/reconstruction_v2_audit.py` 与已完成 v2 审计结果。
- 所有新增数据集必须记录 source URL、SHA-256、license、record counts、drop stats、split stats。
- 所有新增训练必须接入 `--train-idx`/`--val-idx`/`--test-idx` 强制合同（P1-10 完成前可暂用环境变量过渡）。
- 任何"improves TE/stability/expression"的表述必须加 `predicted/internal proxy` 限定词，直到 P1-05 独立 oracle 落地。
- **资源充足设定**：GPU 资源、数据获取、合作湿实验均不受限；优先建立壁垒 2（cross-region synergy）与壁垒 4（RL 算法创新）。

### 13.1 下一阶段 goal（核心目标）

> **P1 — 壁垒 2/4/1/3 的基础设施 + Stage A 健康决策**
>
> 在不干扰 Stage A 4 个长进程的前提下，并行完成 5 条工作流。**优先级 2 > 4 > 1 > 3**，资源向 cross-region synergy 与 RL 算法创新倾斜。
>
> 1. **Stage A 100k 健康审计与决策建议**（P1-00，紧急，0–3 天，**前置 blocker**）：
>    - 审查 `train_backbone._flow_batch_loss` 中连续两次 `sample_cond_pt`、第一次结果被覆盖的问题；补 deterministic regression test，明确正确数学意图。
>    - 跑 NaN/AMP stress test：构造极端输入（全 N、超长序列、空 UTR、极端 GC），验证 AMP fallback 行为。
>    - 在 200/1k/5k/10k checkpoint 上做 held-out learning curve 审计：报告 loss、grad norm P99、AMP fallback rate、proxy TE/MRL delta。
>    - 输出 `docs/stage_a_100k_health_decision.md`：给出 continue/stop/restart 决策建议，依据可追溯到具体指标与 profile；不擅自终止进程。
>
> 2. **真实功能标签数据获取 + 自建 full-length joint mRNA MPRA 数据集（壁垒 2 数据基础）**（P1-02 + P1-03 + P1-02B，1–2 周）：
>    - **P1-02A**：Sample 2019 MPRA（NBT）：~280k random 50 nt 5'UTR + ~35k truncated human 5'UTR + ~3.6k 自然变体，MRL 标签；NatureComm 2024 multi-cell 5'UTR MPRA（多 cell type）；Cao 2020 5'UTR TE data。
>    - **P1-03**：CodonBERT benchmark / mRNABench stability 子任务：half-life/stability 数据；Salvatore 2023 mRNA stability（如可获取）。
>    - **P1-02B（关键，数据壁垒）**：**自建 full-length joint mRNA MPRA 数据集**——领域空白，MEF 的核心数据壁垒。设计 1000-5000 wild-type therapeutic protein mRNAs + MEF-edited + GEMORNA-de-novo + LinearDesign-CDS-only + random legal edits，送合作 MPRA 测 TE & stability（5 arms × 1000-5000 sequences）。**这一数据集使 MEF 在数据维度形成 6-12 个月的竞争者复制壁垒**。
>    - 每个数据集：raw + processed + manifest（source URL、SHA-256、license、record counts、drop stats、split stats、cell/assay metadata）；family-disjoint split + cross-source audit。
>
> 3. **Cross-fitted predictor ensemble + 独立 final oracle（壁垒 3 计算 oracle 部分）**（P1-04 + P1-05，1–2 周，依赖 P1-02/P1-03）：
>    - 训练 ≥2 架构、≥2 训练数据的 cross-fitted predictor ensemble（TE/MRL、stability、CAI/tAI、MFE/accessibility）。
>    - 每折记录 held-out 与 OOD 指标、uncertainty（ensemble disagreement）、calibration（ECE）、applicability domain（distance-to-training）。
>    - 构建 independent final oracle：frozen artifact，与 teacher 数据/权重/feature 不同；test labels 与权重在 design 冻结前不可见。
>    - 输出 `docs/independent_oracle_design_v1.md`：记录三 oracle 合同（train/selection/final）。
>
> 4. **RL 算法创新基础设施 — Innovation 1 CTO + Innovation 2 counterfactual synergy RL（壁垒 4 核心）**（P1-07 + P1-08 + P1-09 + P1-12，2–3 周，可与 P1-02-P1-05 并行）：
>    - **P1-07**：完整合法 action distribution + policy API + tests；mask 前后质量、全池 normalization、STOP 和轨迹 log-prob 可复现。
>    - **P1-08**：RL-1 tiny/exact MDP environment + enumerable tests + profile；REINFORCE 在已知最优小环境收敛；return/baseline/gradient 正确。
>    - **P1-09**：复现 CodonGPT REINFORCE：公开 cargo（ACTB/HLA-A）measured report；记录与论文差异、reward 定义、成本和输出。
>    - **P1-12（新增，壁垒 4 核心）**：实现 **Innovation 1 CTO**（constrained trajectory optimization with constructive hard constraint）+ **Innovation 2 counterfactual cross-region synergy RL**：
>      - CTO：构造性硬约束（protein invariance, frame, edit budget, motif preservation）+ Lagrangian 软目标；证明 policy gradient 在 legal action space 上的收敛性。
>      - Counterfactual synergy RL：对每个 wild-type 生成 4 个 counterfactual rollouts（只改 5'UTR / 只改 CDS / 只改 3'UTR / 联合改）；reward = `joint_improvement - λ × Σ single_region_improvement`；共享 prefix + common random numbers 减小方差；λ schedule（早期大、后期小）。
>      - 输出 `docs/rl_algorithm_innovation_v1.md`：CTO + counterfactual synergy RL 的设计、理论分析、实现细节、tiny MDP 收敛性验证。
>      - 输出 `docs/cross_region_synergy_protocol_v1.md`：定义 cross-region synergy score、counterfactual panel、统计检验（family-cluster bootstrap CI、paired permutation）。
>
> 5. **Counterfactual 编辑实验面板 + synergy score 量化（壁垒 2 机制发现）**（P1-13，1 周，依赖 P1-04/P1-05 oracle）：
>    - 对 1000 wild-type therapeutic proteins，跑 5 arms：(a) wild-type, (b) 只编辑 5'UTR, (c) 只编辑 CDS, (d) 只编辑 3'UTR, (e) 联合编辑（ranker + RL）。
>    - 计算 cross-region synergy score = `(e) - [(b)+(c)+(d) - 2×(a)]`；统计显著性（family-cluster bootstrap CI、paired permutation vs 0）。
>    - 输出 `docs/cross_region_synergy_finding_v1.md`：报告 synergy score 分布、按 cargo/cell 分层、机制解释（哪些 wild-type 有正 synergy、哪些没有）。
>    - **这是壁垒 2 的第一个机制性发现**——即使 RL 还未完全 work，counterfactual panel 本身已可发表。

### 13.2 验收门槛（整体）

P1 完成验收（满足全部才算 P1 done）：

1. `docs/stage_a_100k_health_decision.md` 存在且决策依据可追溯到具体 artifact SHA；
2. 至少 2 个真实功能标签数据集（MPRA/MRL + stability）已接入，manifest 完整（source/SHA/license/counts/split/cell metadata）；
3. **P1-02B 自建 full-length joint mRNA MPRA 数据集设计文档完成**（`docs/full_length_mpra_design_v1.md`），含 5 arms × 1000-5000 sequences 设计、统计计划、合作方对接进展；即使湿实验未完成，**设计文档与统计计划必须 freeze**；
4. cross-fitted predictor ensemble 训练完成，held-out R²/Spearman 与 OOD 指标完整；
5. independent final oracle artifact frozen，与 teacher 数据/权重/feature 不同；
6. RL-1 tiny/exact MDP correctness tests 全通过（return/baseline/gradient 在已知最优小环境收敛）；
7. CodonGPT REINFORCE 在公开 cargo 上 measured report 完整；
8. **Innovation 1 CTO 实现完成**，tiny MDP 上收敛性验证通过；
9. **Innovation 2 counterfactual synergy RL 实现完成**，tiny MDP 上 synergy-optimal policy 收敛性验证通过；
10. **Counterfactual 编辑实验 panel 完成**（1000 wild-type × 5 arms），`docs/cross_region_synergy_finding_v1.md` 存在且 synergy score 统计检验完整；
11. `docs/rl_protocol_v1.md` + `docs/rl_algorithm_innovation_v1.md` + `docs/cross_region_synergy_protocol_v1.md` 存在且 schema 明确；
12. 更新 `docs/next_steps_sota_roadmap.md` Section 10 Go/No-Go dashboard，将 Independent oracle / RL identity-correctness / Stage A 100k 健康 / **Cross-region synergy 机制发现** 四项从 NO 升级为 YES 或 PARTIAL（依据可追溯到 artifact SHA）。

### 13.3 不在本阶段目标内（明确排除）

- 大规模 RL 性能调优（P2-08 RL-2 group-normalized PG 主线 pilot）；
- 重跑 headline 证据（P2-01 leakage-free D1/D2 baselines）；
- 外部 SOTA 公平对比（P3-02 UTailoR matched-budget 等）；
- 架构升级（hierarchical region encoder、decoupled policy/property）——除非 P1-00 Stage A 健康审计建议 stop；
- **P1-02B 湿实验执行本身**——P1 只交付设计文档与统计计划 freeze，湿实验执行在 P3-P4；
- 投稿（P5）。

### 13.4 风险与降级路径

- 若 P1-02/P1-03 数据获取受 license/可用性阻塞：降级为"仅 Sample 2019 + mRNABench stability 子集"，在 `docs/data_acquisition_blocker.md` 记录原因，并继续 P1-04/P1-05 在该子集上训练；P1-02B 设计文档不依赖具体数据源，仍可 freeze。
- 若 P1-00 Stage A 健康审计建议 stop：暂停后续训练相关任务（P1-04 predictor ensemble 仍可继续，因其不依赖 Stage A checkpoint），优先做架构 correctness 修复与 long-view 重建（P1-11），并启动 Innovation 1/2 在 tiny MDP 上的纯算法验证（不依赖 Stage A checkpoint）。
- 若 P1-09 CodonGPT 复现受阻（如 RL policy weights 不可用）：降级为"pretrained checkpoint only"，并在 `docs/codongpt_rl_reproduction_blocker.md` 记录原因，更新"RL unavailable"旧 claim 为"RL reproduction partial"。Innovation 1/2 不依赖 CodonGPT 复现。
- **若 P1-13 counterfactual synergy score 不显著（synergy < 0 或 p > 0.05）**：这是壁垒 2 的核心风险。降级路径：
  - (a) 检查 counterfactual panel 设计是否正确（baseline 是否合理、oracle 是否独立）；
  - (b) 若 synergy 真的不存在，壁垒 2 降级为"null finding + 方法学贡献"，paper 主 claim 转向壁垒 4（RL 算法创新）+ 壁垒 1（regulatory-grade minimal-edit）；
  - (c) Innovation 2 仍可单独投 NeurIPS/ICML 作为算法贡献（synergy 不显著不意味着算法不 work，只意味着这个具体任务无 synergy 可学）。
- 若 P1-12 Innovation 2 实现受阻（理论或工程）：降级为 Innovation 1 CTO + Innovation 3 Adversarial Reward De-Biasing 作为主线，Innovation 2 推迟到 P2。

### 13.5 给 trae 的执行指令模板

```
/go P1 — 壁垒 2/4/1/3 基础设施 + Stage A 健康决策（资源充足 + 计算算法壁垒优先版）
项目路径: /home/cunyuliu/mrna_editflow_goal/mrna_editflow
SSH: cunyuliu@36.137.135.49 -p 22
资源设定: GPU 充足、数据获取不受限、合作湿实验可安排；优先级 壁垒2 > 壁垒4 > 壁垒1 > 壁垒3
约束:
  - 不擅自终止 Stage A 长进程 (PID 1495455/1495549/1495551/1499316)
  - 不修改 v1 frozen namespace (data/reconstructed/p0_data_reconstruction_v1/)
  - 不修改 data/reconstruction_v2_audit.py 与已完成 v2 审计结果
  - 所有新增数据集必须记录 source URL/SHA-256/license/counts/split/metadata
  - 所有新增训练必须接入 --train-idx/--val-idx/--test-idx 强制合同
  - 任何 "improves TE/stability/expression" 必须加 predicted/internal proxy 限定词
任务:
  1. P1-00 Stage A 100k 健康审计: duplicate-sample_cond_pt correctness test、NaN/AMP stress、held-out learning curve、200/1k/5k/10k checkpoint panel → docs/stage_a_100k_health_decision.md (continue/stop/restart 建议)
  2. P1-02A 真实 multi-cell MPRA/MRL 数据获取: Sample 2019 + NatureComm 2024 + Cao 2020 → raw/processed/manifests
  3. P1-03 half-life/stability 数据获取: CodonBERT benchmark + mRNABench stability → raw/processed/manifests
  4. P1-02B (关键数据壁垒) 自建 full-length joint mRNA MPRA 数据集设计: 5 arms × 1000-5000 sequences (wild-type/MEF/GEMORNA/LinearDesign/random) → docs/full_length_mpra_design_v1.md (设计 + 统计计划 freeze, 不执行湿实验)
  5. P1-04 cross-fitted predictor ensemble (TE/MRL/stability/CAI/MFE, ≥2 架构, ≥2 数据): checkpoints + calibration + OOD
  6. P1-05 independent final oracle (frozen, hidden, 与 teacher 不同): frozen artifact + docs/independent_oracle_design_v1.md
  7. P1-07 完整合法 action distribution: policy API + tests (mask 前后质量、STOP、轨迹 log-prob)
  8. P1-08 RL-1 tiny/exact MDP environment: enumerable tests + profile (REINFORCE 收敛、return/baseline/gradient 正确)
  9. P1-09 复现 CodonGPT REINFORCE: 公开 cargo (ACTB/HLA-A) measured report
  10. P1-10 训练入口强制消费 split contract: --train-idx/--val-idx/--test-idx 必需参数 + run_multiseed_benchmark.py exact-match fail-closed
  11. P1-11 long-view 重建: 从 raw GENCODE/RefSeq 重建 5'UTR≤512/CDS≤3072/3'UTR≤1024 + attrition report
  12. P1-12 (壁垒 4 核心) RL 算法创新:
      - Innovation 1 CTO (constrained trajectory optimization with constructive hard constraint) + 收敛性证明
      - Innovation 2 counterfactual cross-region synergy RL (4 counterfactual rollouts + reward = joint - λ×Σsingle + shared prefix + CRN + λ schedule)
      - 输出 docs/rl_algorithm_innovation_v1.md + docs/cross_region_synergy_protocol_v1.md
      - tiny MDP 上 synergy-optimal policy 收敛性验证通过
  13. P1-13 (壁垒 2 机制发现) counterfactual 编辑实验 panel: 1000 wild-type × 5 arms (wild-type/single-5UTR/single-CDS/single-3UTR/joint) → docs/cross_region_synergy_finding_v1.md (synergy score + 统计检验 + 机制解释)
  14. RL protocol 文档: docs/rl_protocol_v1.md (action/log-prob/STOP、三 oracle、budget/query accounting、test lock)
验收:
  - docs/stage_a_100k_health_decision.md 存在且决策依据可追溯到 artifact SHA
  - ≥2 真实功能标签数据集接入 (MPRA/MRL + stability), manifest 完整
  - docs/full_length_mpra_design_v1.md 存在且统计计划 freeze
  - cross-fitted predictor ensemble 训练完成, held-out R²/Spearman 与 OOD 指标完整
  - independent final oracle artifact frozen
  - RL-1 tiny/exact MDP correctness tests 全通过
  - CodonGPT REINFORCE measured report 完整
  - Innovation 1 CTO 实现完成, tiny MDP 收敛性验证通过
  - Innovation 2 counterfactual synergy RL 实现完成, tiny MDP synergy-optimal policy 收敛性验证通过
  - counterfactual 编辑实验 panel 完成 (1000 wild-type × 5 arms), docs/cross_region_synergy_finding_v1.md 存在且 synergy score 统计检验完整
  - docs/rl_protocol_v1.md + docs/rl_algorithm_innovation_v1.md + docs/cross_region_synergy_protocol_v1.md 存在
  - 更新 next_steps_sota_roadmap.md Section 10 Go/No-Go dashboard: Independent oracle / RL identity-correctness / Stage A 100k 健康 / Cross-region synergy 机制发现 四项升级
降级路径:
  - P1-02/03 受阻: 降级为 Sample 2019 + mRNABench stability 子集, 记录 docs/data_acquisition_blocker.md
  - P1-00 建议 stop: 暂停训练相关任务, 优先 P1-11 long-view 重建 + Innovation 1/2 纯算法验证
  - P1-09 受阻: 降级为 pretrained checkpoint only, 记录 docs/codongpt_rl_reproduction_blocker.md
  - P1-13 synergy 不显著: 检查 panel 设计; 若真无 synergy, 壁垒 2 降级为 null finding + 方法学贡献, 主 claim 转向壁垒 4 + 壁垒 1; Innovation 2 仍可单独投 NeurIPS/ICML
  - P1-12 Innovation 2 受阻: 降级为 Innovation 1 CTO + Innovation 3 Adversarial Reward De-Biasing 主线, Innovation 2 推迟 P2
不在范围内:
  - 大规模 RL 性能调优 (P2-08)
  - 重跑 headline 证据 (P2-01)
  - 外部 SOTA 公平对比 (P3-02)
  - 架构升级 (除非 P1-00 建议 stop)
  - P1-02B 湿实验执行本身 (只交付设计文档与统计计划 freeze)
  - 投稿 (P5)
```

---

## 14. 下一阶段 goal（P2，交给 trae 自主执行，2026-07-20，blocker 清除 + 壁垒 2/4 验证版）

> **P2 目标**：清除 P1 遗留的 4 个 critical blocker，验证壁垒 2 (cross-region synergy) 与壁垒 4 (RL 算法创新) 在 full-mRNA 上的 efficacy，为顶刊投稿准备 leakage-free 主表。
>
> **优先级**：P2-01 (cross-region synergy 真伪判定) > P2-02 (Stage A 修复或 pivot) > P2-03 (leakage-free headline) > P2-04 (split 强制) > P2-05/06 (RL scaling) > P2-07 (CodonGPT 复现) > P2-08/09 (外部 SOTA + OOD) > P2-10 (架构升级, conditional) > P2-11/12 (数据 + wet-lab 设计)。
>
> **执行约束**：
> - 不擅自终止任何运行中的进程；PID 必须运行时 rediscovered。
> - 不修改 v1 frozen namespace (`data/reconstructed/p0_data_reconstruction_v1/`)；只追加新数据源与 long view。
> - 不修改已完成 v2 审计结果；新增审计以 v3 命名。
> - 所有新增训练必须接入 `--train-idx`/`--val-idx`/`--test-idx` 强制合同（P2-04 完成前可暂用环境变量过渡）。
> - 任何 "improves TE/stability/expression" 的表述必须加 `predicted/internal proxy` 限定词，直到 P2-01 multi-region oracle 验证完成。
> - 所有性能主张必须基于 10-seed paired significance test（family-cluster bootstrap CI），不再用 decoder seeds 替代 training seeds。
> - 资源充足设定：GPU 充足、数据获取不受限、合作湿实验可安排；优先清除 blocker，再建立壁垒。

### 14.1 P2 任务清单（12 个 task）

#### Priority 1 — Critical blocker 清除（P2-01 ~ P2-04，0–2 周）

**P2-01 — Cross-region synergy 真伪判定（壁垒 2 go/no-go gate，最优先）**
- 用 P1-04 cross-fitted ensemble (CNN-50mer + Transformer-UTR, multi-region: TE + MRL + stability + CAI + MFE) 替换 `LocalTranslationOracle`，重跑 P1-13 1000-wild-type × 5 arms counterfactual panel。
- 关键检验：cross-region synergy score `syn_sum = Δ_joint − (Δ_5 + Δ_c + Δ_3)` 是否在 multi-region oracle 下显著为正（paired t-test α=0.001, family-cluster bootstrap CI, Cohen's d > 0.5 为强证据）。
- 同时报告 region-pair decomposition (5'UTR × CDS, CDS × 3'UTR, 5'UTR × 3'UTR) 与 per-cargo/per-cell 分层。
- 输出 `docs/cross_region_synergy_finding_v2.md` + `docs/cross_region_synergy_panel_results_v2.json` (SHA-256 frozen)。
- **决策规则**：
  - 若 `syn_sum > 0, p < 0.001, d > 0.5`：壁垒 2 GO，proceed to P2-06 (synergy RL full-mRNA validation)。
  - 若 `syn_sum > 0, p < 0.05, d ∈ (0.2, 0.5)`：borderline，proceed to P2-06 但准备 null finding 降级路径。
  - 若 `syn_sum ≈ 0, p > 0.05`：壁垒 2 降级为 null finding + 方法学贡献；主 claim 转向壁垒 4 + 壁垒 1；Innovation 2 仍可投 NeurIPS/ICML workshop。
  - 若 `syn_sum < 0`：redundancy finding，重新审视 oracle 与 panel 设计。

**P2-02 — Stage A 修复或 pivot（解除 trained backbone blocker）**
- 独立评估 4 个 `stage_a_best.pt` (seed 0/1/2/5) 的 held-out 性能：在 frozen test split 上跑 proxy TE/MRL + legality + protein exact-1，决定是否有任何 checkpoint 可用作 warm start。
- 修复 Stage A config：(a) AMP scaler 不在 fallback 后永久禁用；(b) learning rate 降低 10–100×；(c) gradient clipping `max_norm=1.0`；(d) 审查 `U.edit_flow_loss` 的 reduction（建议 mean 而非 sum）；(e) `batch_size` 或 `grad_accum` 增至 8–16；(f) `save_every=1000` 启用 step-level checkpoint。
- 重启 1 个 seed (建议 seed=0) 跑到 10k steps，验证 held-out loss 下降、grad norm P99 < 1000、AMP fallback < 0.05；若 10k 内 held-out 无增益，pivot 到 P2-10 alternative backbone。
- 输出 `docs/stage_a_recovery_decision.md` + 修复后 config SHA-256 + 10k checkpoint SHA-256。

**P2-03 — Leakage-free headline eval（统计有效性 blocker）**
- 在 frozen test split (combined_family 或 gencode_family) 上重跑：te_only / scalar / pareto / grpo / hardneg_v2 / random/legal / local-search / codon-lattice DP。
- 至少 3 个独立 training seeds（不是 decoder seeds）；每个 checkpoint 内再跑 10 decoder seeds，保留层级结构。
- 主推断单位为 transcript/family；使用 hierarchical bootstrap 或 mixed-effects model，报告 family-cluster robust 95% CI。
- 预注册 1 个 primary endpoint（建议 `delta_oracle_te_vs_source` on `combined_family` test split）和最多 2 个 secondary endpoints；其余指标做 FDR/Holm 校正。
- 输出 `docs/p2_03_leakage_free_headline.md` + `benchmark/paper/leakage_free_headline.json` (SHA-256 frozen)。

**P2-04 — 训练入口强制 split contract（工程 blocker）**
- 所有训练入口新增必需参数 `--train-idx`/`--val-idx`/`--test-idx`；缺 idx 时 paper mode 直接退出非零。
- `run_multiseed_benchmark.py` 在 records 与训练语料 exact-match 时默认 `fail_closed=True`（不是 warning）。
- foundation pretraining corpus 的 leakage audit（若使用外部 backbone）。
- 单元测试覆盖：缺 idx 退出、SHA 不匹配退出、exact-match fail-closed。
- 输出 `tests/test_split_contract_enforcement.py` + `docs/p2_04_split_contract.md`。

#### Priority 2 — 壁垒 4 RL scaling 验证（P2-05 ~ P2-07，2–4 周，依赖 P2-02）

**P2-05 — RL-2 group-normalized PG 主线 pilot（D2 CDS first，D1 5'UTR second）**
- 从 P2-02 修复后的 Stage A checkpoint warm start（或从 P2-10 frozen foundation backbone warm start，若 Stage A pivot）。
- 实现 RL-2：group-normalized terminal advantage + KL + entropy，每个 source/preference 采样 K=8 条 rollout 做组内 advantage。
- 任务：先 D2 CDS synonymous editing，再 D1 5'UTR editing。
- 对照：ranker-only / beam / best-of-N / simulated annealing，**等 oracle-call budget** 匹配。
- 重复：≥3 policy-training seeds × 10 rollout/decoder seeds，family-cluster bootstrap CI。
- 主指标：independent-oracle (P1-05 Oracle #3) Pareto hypervolume、gain/edit、worst-objective regression、hard exact-1。
- 输出 `docs/p2_05_rl2_pilot.md` + checkpoint + trajectory JSONL + curves。

**P2-06 — Innovation 2 counterfactual synergy RL full-mRNA validation（壁垒 4 核心，依赖 P2-01 GO）**
- 若 P2-01 判定壁垒 2 GO 或 borderline：在 full mRNA (1000+ nt) 上验证 Innovation 2 synergy RL，不再只用 tiny MDP。
- 用 P1-04 ensemble 作为 reward（multi-region oracle），counterfactual rollouts 共享 prefix + CRN，λ schedule（warmup → anneal → final）。
- 对照：vanilla REINFORCE (Innovation 1 CTO only) vs synergy REINFORCE (Innovation 2)，等 query budget。
- 主指标：synergy score 在 trained policy 下是否显著高于 random policy (P1-13 baseline)；independent oracle 上 gain/edit。
- 输出 `docs/p2_06_synergy_rl_full_mrna.md` + checkpoint + trajectory JSONL。
- **若 P2-01 判定壁垒 2 NO-GO**：P2-06 跳过，Innovation 2 仅作为 tiny-MDP 方法论贡献保留，资源转移到 P2-07/08。

**P2-07 — CodonGPT REINFORCE reproduction（外部 RL baseline）**
- 在公开 cargo (ACTB/HLA-A) 上复现 CodonGPT REINFORCE：measured report 含 reward 定义、compute、输出。
- 记录与论文差异、版本、checkpoint SHA-256。
- 若 RL policy weights 不可用：降级为 "pretrained checkpoint only"，记录 `docs/codongpt_rl_reproduction_blocker.md`。
- 输出 `docs/p2_07_codongpt_rl_reproduction.md` + benchmark artifacts。

#### Priority 3 — 外部 SOTA + 鲁棒性（P2-08 ~ P2-09，3–5 周）

**P2-08 — 外部 SOTA matched-budget head-to-head**
- 在 frozen test split 上对齐 protocol：相同 source/protein/cargo、相同可编辑区域、相同 hard budget、相同 oracle、相同候选预算。
- UTailoR (5'UTR, hard-budget-5, strict subset)、UTRGAN (5'UTR, paper-default)、LinearDesign (CDS, protein-conditioned)、EnsembleDesign (CDS, 修复 coverage)、codonGPT (CDS, REINFORCE)。
- 每方法 ≥3 seeds，family-cluster bootstrap CI，10-seed paired significance test。
- 修复 EnsembleDesign coverage mismatch；明确 UTRGAN source-conditioned paired inference 不可行时改用 distribution-level comparison。
- 输出 `docs/p2_08_external_sota_matched_budget.md` + `benchmark/external_sota/matched_budget/`。

**P2-09 — OOD robustness stress test**
- family/source/cell/cargo/length/GC stress matrix：worst-group performance + failure rate。
- cell/cargo 维度当前无数据：至少完成 family/source stress；cell/cargo 标记为 P3。
- 输出 `docs/p2_09_ood_robustness.md` + `benchmark/ood/stress_matrix.json`。

#### Priority 4 — 架构升级 conditional（P2-10，仅在 P2-02 pivot 时启动）

**P2-10 — Alternative backbone pivot（若 Stage A 修复失败）**
- 选项 A：hierarchical region encoder (nucleotide local blocks + codon tokens + region summary tokens + FlashAttention)。
- 选项 B：frozen foundation backbone (Helix-mRNA / mRNA-LM / CodonFM) + MEF head，leakage audit。
- 选项 C：reduced LR + grad clipping + mean reduction 的 Stage A restart（P2-02 修复版）。
- 选定后跑 10k steps 验证 held-out 增益，再决定是否扩展到 100k。
- 输出 `docs/p2_10_backbone_pivot_decision.md`。

#### Priority 5 — 数据壁垒 + wet-lab 设计（P2-11 ~ P2-12，4–8 周，并行）

**P2-11 — P1-02B full-length MPRA 设计 freeze + 合作 wet-lab 启动**
- 完成 `docs/full_length_mpra_design_v1.md`：5 arms × 1000–5000 sequences (wild-type / MEF / GEMORNA / LinearDesign / random legal)，统计计划 freeze（预注册功效分析、blinded sequence IDs、≥3 biological replicates）。
- 启动合作湿实验方对接：≥3 cargo (reporter / 分泌蛋白 / 功能蛋白) × ≥2 cell context。
- 即使湿实验未执行，**设计文档与统计计划必须 freeze**。
- 输出 `docs/full_length_mpra_design_v1.md` (freeze) + `docs/wet_lab_collaboration_plan.md`。

**P2-12 — Leplek 2022 PERSIST-Seq 数据集成 + Oracle #3 增强**
- 获取 Leplek 2022 PERSIST-Seq (3'UTR stability MPRA, ~6k sequences, paired MRL + half-life)。
- 集成到 Oracle #3 (GBT) 作为额外训练数据，提升 3'UTR 维度预测能力。
- 重新 lock Oracle #3 v1.2，HMAC-signed + chmod 444，更新 `docs/p1_05_independent_oracle_design.md`。
- 输出 `docs/p2_12_leplek_integration.md` + Oracle #3 v1.2 artifact。

### 14.2 验收门槛（整体）

P2 完成验收（满足全部才算 P2 done）：

1. **P2-01 cross-region synergy 真伪有明确定论**（GO / borderline / NO-GO），`docs/cross_region_synergy_finding_v2.md` 存在且 multi-region oracle 下统计检验完整；
2. **P2-02 Stage A 修复或 pivot 决策完成**，若有可用 checkpoint 则 10k steps 内 held-out 有增益；若 pivot 则 P2-10 文档完成；
3. **P2-03 leakage-free headline eval 完成**，≥3 training seeds，family-cluster bootstrap CI，primary endpoint 预注册；
4. **P2-04 split contract 强制消费**，单元测试覆盖缺 idx 退出、SHA 不匹配退出、exact-match fail-closed；
5. **P2-05 RL-2 pilot 完成**（若 P2-02 unblock），≥3 policy seeds，等 query 优于 beam/SA 或明确降级；
6. **P2-06 Innovation 2 full-mRNA validation 完成**（若 P2-01 GO），synergy-trained policy 显著优于 random policy；
7. **P2-07 CodonGPT REINFORCE reproduction 完成**或降级文档 freeze；
8. **P2-08 外部 SOTA matched-budget**：≥4/6 方法 protocol-aligned measured，10-seed paired test；
9. **P2-09 OOD robustness**：family/source stress 完成，cell/cargo 标记 P3；
10. **P2-10 alternative backbone 决策**（若 P2-02 pivot）；
11. **P2-11 full-length MPRA 设计 freeze** + wet-lab 合作启动；
12. **P2-12 Leplek 2022 集成** + Oracle #3 v1.2 locked。
13. 更新 `docs/next_steps_sota_roadmap.md` Section 10 Go/No-Go dashboard：Statistical validity / Internal held-out gain / RL independent gain / RL mechanism / External SOTA fairness / Stage A 100k 健康 六项依据可追溯到 artifact SHA 升级。

### 14.3 不在 P2 范围内（明确排除）

- P3-07 D3 universal constrained RL（跨区域 joint RL 主线，依赖 P2-05/06 验证通过）；
- P3-08 等 reward-query RL/search SOTA compute-normalized table（依赖 P2-08）；
- P3-09 test-time personalization ablation；
- P4 prospective wet-lab 执行本身（P2-11 只交付设计 freeze 与合作启动）；
- P5 投稿包；
- Innovation 3 Adversarial Reward De-Biasing / Innovation 4 Hierarchical Region-Options / Innovation 5 Offline-to-Online CQL 实现（除非 P2-01 NO-GO 且需要补强壁垒 4）。

### 14.4 风险与降级路径

- **若 P2-01 cross-region synergy NO-GO**：壁垒 2 降级为 null finding + 方法学贡献；主 claim 转向壁垒 4 (RL 算法创新) + 壁垒 1 (regulatory-grade minimal-edit)；投稿层级调整为 Nature Methods / Nat Comm / NAR；Innovation 2 仍可投 NeurIPS/ICML workshop。P2-06 跳过，资源转移到 P2-07/08/11。
- **若 P2-02 Stage A 修复失败**：pivot 到 P2-10 frozen foundation backbone (Helix-mRNA/mRNA-LM/CodonFM) + MEF head；leakage audit 后用作 warm start；Stage A 自训练推迟到 P3。
- **若 P2-05 RL-2 不胜 beam/SA**：RL 不进入主模型，保留为诚实负结果或个性化可选模块；论文主算法回到最强蒸馏/搜索方法；不得因投入成本强行把 RL 放 headline。
- **若 P2-07 CodonGPT 复现受阻**：降级为 pretrained checkpoint only，记录 blocker；Innovation 1/2 不依赖 CodonGPT 复现。
- **若 P2-08 外部 SOTA protocol fidelity 仍不足**：明确报告 "external SOTA comparison partial"，论文不 claim superiority，只 claim "constrained edit paradigm difference"。
- **若 P2-11 wet-lab 合作未启动**：P2 只交付设计 freeze；投稿层级降至计算方法期刊；wet-lab 推迟到 P4。

### 14.5 给 trae 的执行指令模板

```
/go P2 — blocker 清除 + 壁垒 2/4 full-mRNA 验证（2026-07-20 v3）
项目路径: /home/cunyuliu/mrna_editflow_goal/mrna_editflow
SSH: cunyuliu@36.137.135.49 -p 22
资源设定: GPU 充足、数据获取不受限、合作湿实验可安排；优先级 P2-01 > P2-02 > P2-03 > P2-04 > P2-05/06 > P2-07 > P2-08/09 > P2-10 > P2-11/12
约束:
  - 不擅自终止任何运行中进程；PID 必须 runtime rediscovered
  - 不修改 v1 frozen namespace (data/reconstructed/p0_data_reconstruction_v1/)
  - 不修改已完成 v2 审计结果；新增审计以 v3 命名
  - 所有新增训练必须接入 --train-idx/--val-idx/--test-idx 强制合同
  - 任何 "improves TE/stability/expression" 必须加 predicted/internal proxy 限定词，直到 P2-01 multi-region oracle 验证完成
  - 所有性能主张必须基于 10-seed paired significance test (family-cluster bootstrap CI)，不再用 decoder seeds 替代 training seeds
  - 所有新代码必须配套单元测试
任务:
  1. P2-01 (壁垒 2 go/no-go gate, 最优先) cross-region synergy 真伪判定:
     - 用 P1-04 cross-fitted ensemble (multi-region: TE+MRL+stability+CAI+MFE) 替换 LocalTranslationOracle
     - 重跑 1000 wild-type × 5 arms counterfactual panel
     - 检验 syn_sum = Δ_joint − (Δ_5 + Δ_c + Δ_3) 显著性 (α=0.001, family-cluster bootstrap CI, Cohen's d)
     - region-pair decomposition (5'UTR×CDS, CDS×3'UTR, 5'UTR×3'UTR)
     - 输出 docs/cross_region_synergy_finding_v2.md + docs/cross_region_synergy_panel_results_v2.json (SHA-256 frozen)
     - 决策: GO (d>0.5, p<0.001) / borderline (d∈(0.2,0.5), p<0.05) / NO-GO (p>0.05)
  2. P2-02 Stage A 修复或 pivot:
     - 独立评估 4 个 stage_a_best.pt held-out 性能
     - 修复 config: AMP scaler / LR 降 10-100x / grad clipping max_norm=1.0 / loss reduction mean / batch_size 或 grad_accum 8-16 / save_every=1000
     - 重启 1 seed (建议 seed=0) 跑到 10k steps，验证 held-out loss 下降、grad norm P99<1000、AMP fallback<0.05
     - 若 10k 无增益，pivot 到 P2-10
     - 输出 docs/stage_a_recovery_decision.md + 修复后 config SHA-256 + 10k checkpoint SHA-256
  3. P2-03 leakage-free headline eval:
     - 在 frozen test split (combined_family 或 gencode_family) 上重跑 te_only/scalar/pareto/grpo/hardneg_v2/random/legal/local-search/codon-lattice DP
     - ≥3 training seeds × 10 decoder seeds，hierarchical bootstrap / mixed-effects model
     - family-cluster robust 95% CI，预注册 1 primary + ≤2 secondary endpoints，其余 FDR/Holm 校正
     - 输出 docs/p2_03_leakage_free_headline.md + benchmark/paper/leakage_free_headline.json (SHA-256 frozen)
  4. P2-04 split contract 强制消费:
     - 所有训练入口新增必需参数 --train-idx/--val-idx/--test-idx
     - run_multiseed_benchmark.py exact-match 默认 fail_closed=True
     - foundation pretraining corpus leakage audit
     - 单元测试覆盖: 缺 idx 退出、SHA 不匹配退出、exact-match fail-closed
     - 输出 tests/test_split_contract_enforcement.py + docs/p2_04_split_contract.md
  5. P2-05 RL-2 group-normalized PG pilot (依赖 P2-02):
     - 从 P2-02 修复后 Stage A checkpoint warm start
     - RL-2: group-normalized terminal advantage + KL + entropy, K=8 rollouts per source/preference
     - 任务: 先 D2 CDS, 再 D1 5'UTR
     - 对照: ranker-only / beam / best-of-N / SA, 等 oracle-call budget
     - ≥3 policy seeds × 10 rollout seeds, family-cluster bootstrap CI
     - 主指标: Oracle #3 Pareto hypervolume / gain/edit / worst-objective regression / hard exact-1
     - 输出 docs/p2_05_rl2_pilot.md + checkpoint + trajectory JSONL + curves
  6. P2-06 Innovation 2 full-mRNA validation (依赖 P2-01 GO):
     - 在 full mRNA (1000+ nt) 上验证 counterfactual synergy RL
     - P1-04 ensemble reward, counterfactual rollouts 共享 prefix + CRN, λ schedule
     - 对照: vanilla REINFORCE (Innovation 1) vs synergy REINFORCE (Innovation 2), 等 query
     - 主指标: synergy score 在 trained policy 下显著高于 random policy (P1-13 baseline)
     - 输出 docs/p2_06_synergy_rl_full_mrna.md + checkpoint + trajectory JSONL
     - 若 P2-01 NO-GO: 跳过, Innovation 2 仅作 tiny-MDP 方法论贡献
  7. P2-07 CodonGPT REINFORCE reproduction:
     - 公开 cargo (ACTB/HLA-A) measured report
     - 记录与论文差异、版本、checkpoint SHA-256
     - 若受阻: 降级为 pretrained checkpoint only, 记录 docs/codongpt_rl_reproduction_blocker.md
     - 输出 docs/p2_07_codongpt_rl_reproduction.md + benchmark artifacts
  8. P2-08 外部 SOTA matched-budget head-to-head:
     - frozen test split 上对齐 protocol: 相同 source/protein/cargo/budget/oracle/候选预算
     - UTailoR/UTRGAN/LinearDesign/EnsembleDesign/codonGPT, 每方法 ≥3 seeds, 10-seed paired test
     - 修复 EnsembleDesign coverage mismatch; UTRGAN distribution-level comparison 若 paired 不可行
     - 输出 docs/p2_08_external_sota_matched_budget.md + benchmark/external_sota/matched_budget/
  9. P2-09 OOD robustness stress test:
     - family/source/cell/cargo/length/GC stress matrix: worst-group + failure rate
     - cell/cargo 无数据则标记 P3
     - 输出 docs/p2_09_ood_robustness.md + benchmark/ood/stress_matrix.json
  10. P2-10 alternative backbone pivot (conditional, 若 P2-02 失败):
      - 选项 A: hierarchical region encoder (FlashAttention)
      - 选项 B: frozen foundation backbone (Helix-mRNA/mRNA-LM/CodonFM) + MEF head, leakage audit
      - 选项 C: P2-02 修复版 restart
      - 选定后跑 10k steps 验证 held-out 增益
      - 输出 docs/p2_10_backbone_pivot_decision.md
  11. P2-11 full-length MPRA 设计 freeze + wet-lab 启动:
      - 完成 docs/full_length_mpra_design_v1.md: 5 arms × 1000-5000 sequences, 统计计划 freeze
      - 启动合作湿实验方对接: ≥3 cargo × ≥2 cell context
      - 即使湿实验未执行, 设计文档与统计计划必须 freeze
      - 输出 docs/full_length_mpra_design_v1.md (freeze) + docs/wet_lab_collaboration_plan.md
  12. P2-12 Leplek 2022 PERSIST-Seq 集成 + Oracle #3 增强:
      - 获取 Leplek 2022 PERSIST-Seq (~6k 3'UTR stability MPRA)
      - 集成到 Oracle #3 (GBT), 提升 3'UTR 维度
      - 重新 lock Oracle #3 v1.2, HMAC-signed + chmod 444
      - 输出 docs/p2_12_leplek_integration.md + Oracle #3 v1.2 artifac