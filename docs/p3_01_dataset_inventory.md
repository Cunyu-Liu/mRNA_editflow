# P3-01 Dataset Inventory — 四级数据分级注册表（Task 1）

冻结时间：2026-07-23T04:21:52Z ｜ 机器可读注册表：[data/p3/manifests/p3_01_dataset_registry.json](../data/p3/manifests/p3_01_dataset_registry.json)

本清单执行 P3-01 任务1：把全部可用数据资产分为四级。**只有 Level C 可以作为
local-delta 的依据**；Level A/B 不得用于奖励训练中的局部编辑真值，Level D 目前不存在。
本清单与 P3-00A 冻结契约（commit `3e85c79`，tag `p3_00a_frozen_contract`）一致：
`phase_status=PASS` 仅表示工程门禁通过，`scientific_validation_status=PENDING`。

## Level A — Observational（自然序列绝对标签）

仅用于预训练/背景分析。**跨基因/跨转录本的绝对标签，无 source-matched 编辑，
永远不能充当 local-delta ground truth。**

| 数据集 | 角色 | 为什么不是 local-delta | License |
|---|---|---|---|
| cao2021_5utr | 内源 5'UTR TE（Ribo-seq，3 种细胞类型） | 跨基因绝对 TE，无同源编辑对 | GitHub MIT + 文章 CC BY 4.0；FASTQ 为 NCBI GEO 公共数据；引 Cao et al. 2021 Nat Commun (PMID:34230498) |
| saluki_halflife | mRNA 半衰期标签 | 跨转录本绝对半衰期 | CC BY 4.0；引 Agarwal & Kelley 2022 Genome Biology + mRNABench  redistribution (Zenodo 14708163) |
| codonbert_stability | mRNA 稳定性标签 | 跨转录本绝对稳定性 | Sanofi CodonBERT Artifact License（研究用途）；底层 iCodon CC BY / OpenVaccine CC0 |
| refseq_gencode_catalog | p0_data_reconstruction_v1 的序列资产 | 无任何功能标签 | RefSeq/GENCODE 公共序列 |

## Level B — Cross-construct（跨构建设计数据）

许多**不同**构建体的绝对标签。可用于排序/分析；**不是同一 source 的局部扰动**。

| 数据集 | 角色 | 为什么不是 local-delta | License |
|---|---|---|---|
| sample2019_mpra_random | 随机 50-mer / 变长 MPRA 库（GSM3130441/442 mCherry、GSM4084997 25–100nt） | 随机构建体之间无共同 WT 锚点 | NCBI GEO 公共数据；引 Sample et al. 2019 (PMID:31267113, GSE114002) |
| lepplek2022_persistseq | 233 条全长 mRNA 设计的蛋白输出时间序列 | 跨设计比较，不是单 source 的局部编辑 | NCBI GEO 公共数据；引 Leppek et al. 2022 Nat Commun (PMID:35318324, GSE173083) |
| khoroshkin2024_parade | 跨构建 mRNA 设计数据集 | 跨设计绝对标签 | 预印本 CC BY-NC-ND 4.0；截至 2026-07-19 数据未公开存储 |

## Level C — Source-matched local perturbation（核心）

**唯一可以锚定 local-delta 声明的级别。**

| 数据集 | 角色 | 限定 | License |
|---|---|---|---|
| sample2019_mpra_snv | **MEASURED tier**：snv 库（GSM3130443），WT 锚定的 5'UTR 单/双/多编辑变体，MPRA ribosome load | wet-lab measured；单一 cargo（EGFP）+ 单一细胞上下文（HEK293T）+ 50nt 窗口；端点是 ribosome load 而非 protein output | NCBI GEO 公共数据；引 Sample et al. 2019 |
| p3_01_proxy_neighborhood | **PROXY tier**：重建人源 mRNA 5'UTR 受控邻域 + 冻结 CNN-50mer cross-fitted ensemble（P1-04，15 checkpoints）打分 | **predicted/internal proxy**——不是 wet-lab；每条记录带 `value_qualifier` 限定词；覆盖仅限 5'UTR 前 50nt | 派生自 p0_data_reconstruction_v1 + 内部模型 ckpts/p1_04_predictors |

## Level D — Prospective internal intervention（前瞻干预）

模型提议编辑的前瞻性湿实验读出。**目前不存在**（规划于 P3-03），按设计注册为空。
在 Level D 存在之前，任何 "improves TE/stability/expression" 声明必须带
predicted/internal proxy 限定词（与项目硬约束一致）。

## 注册表字段

`p3_01_dataset_registry.json` 每个条目含：`level`、`role`、`why_not_local_delta`、
以及从 `data/raw/*/manifest.json` 合并的 `files[]`（`local_path`、`sha256`、
`record_count`、`source_url`、`license`、`citation`）。原始文件 hash 在注册时
逐文件核验，任何后续改动都会使 hash 失配并暴露。
