# DeltaBench 数据可用性 rights review 审计报告 v1（2026-10-06）

- **Owner（用户 2026-10-06 指派）**：TRAE 执行侧 agent（交接审计会话）——负责证据回源核实、边界判定与 Data Availability 起草；**最终权利边界确认与发布决策 = 用户/PI**（本审计是 accountable review 的证据层与建议层）。
- **方法**：逐研究回源核验（GEO 落地页/政策页、ENCODE TOS、BioStudies/NC 论文 Data availability 段，2026-10-06 现行版本），对照 `route2_v332_study_rights_accountable_human_review_packet_v1.csv`（机器采集包），按评审说明文档三档判定（NOT_AUTHORIZED / AUTHORIZED_EXACT_FILES / NOT_APPLICABLE）。
- **范围修正**：原 14 行表缺 **GSE232927（M1 行，Castillo-Hair 2024）**——M1 既是训练臂语料又是评测新行，必须入表 → 本审计覆盖 **15 研究**。

## 1. 逐研究判定表（15 行）

判定口径：**A&P = analysis & publication use**（引用 accession + 发表聚合结果/转换产物/评测读数）；**再分发 = 本项目是否可把原始 payload（supplementary 文件）随论文/仓库再发布**。

| # | 研究（accession） | 角色 | 源政策（2026-10-06 回源） | A&P | 再分发判定 | 依据要点 |
|---|---|---|---|---|---|---|
| 1 | GSE114002（Sample 2019, Nat Biotech） | MRL 主行 + 280K 库 | GEO：NCBI 不限制使用分发，**提交者可保留 IP**；页面无 study-specific license 字段 | ✅ | **NOT_AUTHORIZED**（保守） | GEO disclaimer 例外条款 + 无逐研究许可 → 走 accession+脚本+聚合 |
| 2 | GSE232927（Castillo-Hair 2024, Nat Commun） | M1 新行 + 30 万行训练语料 | 同上；论文 = s41467-024-49508-2（NC 开放获取，CC BY 4.0 论文文本；**数据仍按 GEO 条款**） | ✅ | **NOT_AUTHORIZED**（保守） | 同 GEO 例外；NC 论文的 data availability 指向 GEO/SRA |
| 3 | GSE269595（Kowalski 2024/2026, CPA-Perturb-seq） | polyA 主行 | 同 GEO | ✅ | **NOT_AUTHORIZED**（保守） | 同上（MPRA construct 表在 GEO supplementary） |
| 4 | ENCSR854RUF（Xue et al. MPRAu） | MPRAU 行 | **ENCODE TOS（2026-10-06 现行）**：may freely download, analyze and publish with no restriction, commercial or otherwise | ✅ | 保守走 accession（ENCODE 条款不等于逐文件再授权）→ **NOT_AUTHORIZED（保守）**；正文写 "ENCODE data use policy permits unrestricted analysis and publication" | encodeproject.org/help/rest-api TOS 原文 |
| 5 | GSE200304（Schuster 2023, Cell Rep） | TE 3'UTR 行 | GEO | ✅ | **NOT_AUTHORIZED**（保守） | 同 GEO 例外 |
| 6 | GSE149487（Lim 2021, Nat Commun，PLUMAGE） | LOSO 两行 | GEO | ✅ | **NOT_AUTHORIZED**（保守） | 同上；n=48 行 |
| 7 | GSE217518（Su 2025, eLife） | HL 两行 + S1 稳定性行 | GEO | ✅ | **NOT_AUTHORIZED**（保守） | 同上；S1 = 5,572 子行 |
| 8 | GSE186455（Cre-dependent MPRA） | REF/ALT 行 | GEO | ✅ | **NOT_AUTHORIZED**（保守） | 同上 |
| 9 | GSE246381（Plassmeyer 2025，NDD 5'UTR） | M6 新行 | GEO（**sealed/unseal 历史按 provenance 条款另计**） | ✅ | **NOT_AUTHORIZED**（保守） | 同 GEO 例外；historical_exposure 措辞条款仍生效 |
| 10 | GSE256185（Thoreen 2024, Mol Cell DART） | 训练语料 | GEO | ✅ | **NOT_AUTHORIZED**（保守） | 同上 |
| 11 | GSE232572（rare COSMIC 3'UTR MPRA） | 训练语料 | GEO | ✅ | **NOT_AUTHORIZED**（保守） | 同上 |
| 12 | GSE145046（codon 报告基因库） | 训练语料 | GEO | ✅ | **NOT_AUTHORIZED**（保守） | 同上 |
| 13 | GSE207584（zebrafish synonymous codon） | 训练语料 | GEO | ✅ | **NOT_AUTHORIZED**（保守） | 同上 |
| 14 | GSE261709（Miliotis 2024, Nat Commun，胃癌 ilQTL） | 训练语料（MPRA amplicon 数据） | GEO + NC 论文 Data availability：原始数据已存 GEO GSE261709；代码 github.com/ivlachos/3UTR | ✅ | **NOT_AUTHORIZED**（保守） | NC 论文本身声明数据经 GEO 提供 → 同 GEO 条款 |
| 15 | E-MTAB-10902（von Kügelgen 2021，N-zip 神经元 MPRA） | 训练语料 | **BioStudies/ArrayExpress（EMBL-EBI）**：所有 EMBL-EBI 数据库在 terms of use 下对学术/临床/商业用户**自由可用**（要求正确引用）；研究级 license 页 2026-10-06 JS 渲染未取到逐研究 license 字段 | ✅ | **HOLD（待逐研究 license 字段确认）** | EBI FAQ 明确 free to use + cite；逐研究 license 字段（若存在 CC BY 之类）需在 BioStudies JSON API 确认（下一步） |

**共同判定原则（GEO 12 行）**：GEO 政策页（2024-07-16 版，2026-10-06 复核仍现行）明确「NCBI places no restrictions on the use or distribution of GEO data」**但**「submitters may claim patent, copyright, or other IP rights」且 NCBI「cannot provide unrestricted permission」。逐 submission 页面**无 license 字段**（GEO 不提供逐研究许可声明）→ **在任何提交者未明确授权的前提下，再分发 = NOT_AUTHORIZED**（保守判定，与评审说明文档的 AUTHORIZED_EXACT_FILES 高门槛一致：general provider policy 不构成逐文件授权）。**引用 accession + 发布转换/聚合产物不受影响**（这是 GEO 政策明文允许的使用）。

## 2. Data Availability 草稿（投稿版，英文）

> **Data availability.** All evaluation records are derived from 15 publicly deposited studies (12 NCBI GEO series, 1 ENCODE experiment, 1 EMBL-EBI BioStudies/ArrayExpress study, and their associated SRA/BioProject records), each cited by accession in Supplementary Table S1 together with the primary publication. Consistent with the NCBI GEO disclaimer (submitters may retain IP rights; NCBI cannot grant unrestricted redistribution permission) and the ENCODE/EMBL-EBI data-use terms, **raw source payloads are not redistributed with this manuscript**. Instead, we release: (i) per-study converters and row-construction scripts (seed-frozen, deterministic) that regenerate every canonical evaluation record from the public accessions; (ii) the frozen evaluation manifests (record IDs, splits, and grouping) with per-record summary statistics; (iii) all model predictions, metrics, and adjudication JSONs; and (iv) the full reproduction pipeline. No "available on request" channel is promised for raw payloads; access follows each provider's public route.

**对应的 payload 发布边界执行表**：协议 + 转换脚本 + manifest + 汇总统计 = 发布；原始 supplementary 文件 = 不发布（0 授权行）。与 v332 rights 表的 0/14 授权现状一致（现为 0/15）。

## 3. AMBIGUOUS / 后续动作清单（不阻塞预印本）

| # | 事项 | 动作 | 阻塞？ |
|---|---|---|---|
| 1 | E-MTAB-10902 逐研究 license 字段（BioStudies JSON） | 提交前用 BioStudies API 取 license 字段；若为 CC BY/CC0 → 该行可升级 AUTHORIZED_EXACT_FILES（不改变保守主路线） | 否 |
| 2 | GEO 12 行的提交者逐研究授权（若想发布 payload） | 需逐提交者邮件确认（14+ 通讯作者）——**预印本不需要**；仅正式投稿若期刊强制原始数据发布时启动 | 否（预印本）；期刊期再评估 |
| 3 | ENCODE ENCSR854RUF | TOS 已允许自由分析/发表；正文引用其 data use policy 即可 | 否 |
| 4 | 5 个模型权重的 license（批 0-2 骨干） | 已在 amendment v2/port ledger 登记（MIT/Apache/NVIDIA Open Model/AGPL 等）；CaLM AGPL 与 CodonFM NVIDIA license 对**代码仓库**的影响（AGPL 传染性仅当分发其权重——我们只在评测时加载，不分发 → 无传染；NVIDIA Open Model 同理） | 否 |
| 5 | journal policy check | venue 定稿时核对（NC/GB/Cell Systems 均接受 "scripts + accession + derived data" 形态） | 投稿期 |

## 4. 结论（owner 判定）

- **15/15 研究 analysis & publication use = 允许**（GEO/ENCODE/EBI 条款均明文支持）；**15/15 原始 payload 再分发 = 保守不授权**（0 个 AUTHORIZED_EXACT_FILES；E-MTAB-10902 待 license 字段，可能单独升级）。
- **预印本 Data Availability 可用 §2 草稿立即成文**（bioRxiv 不强制原始数据再分发）。
- 本审计 = 证据层；**发布决策与最终签字 = 用户/PI**（评审说明文档的角色边界保持不变）。
