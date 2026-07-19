# mRNA-EditFlow 数据集与 mRNA 原生基座调研

日期：2026-07-11

本文档对应 Task 0.2，目标是为 mRNA-EditFlow 的预训练、下游任务与 SOTA 对标建立数据资产清单。写作口径为研究方案，不假定所有数据已经下载完成。凡是公开页面只给论文摘要而未给明确下载或许可证的条目，本文标为“需下载核验/需许可核验”。

核心科学故事：**mRNA 设计 = 区域条件下的变长编辑；CDS 是帧锁定密码子格，UTR 是长度自由调控画布**。因此数据调研不能只统计“多少条序列”，必须回答三件事：

1. 是否有可靠的 5UTR/CDS/3UTR 边界。
2. 是否能支持 UTR 的变长插删改与 CDS 的帧锁定同义编辑。
3. 是否能构造不泄漏的 train/val/test 与独立 oracle 评测。

## 1. 结论摘要

建议的数据角色分工：

| 角色 | 首选来源 | 使用方式 | 关键风险 |
|---|---|---|---|
| Stage A 全长 mRNA 生成头预训练 | RefSeq + GENCODE protein-coding transcripts，多物种可扩展到 Ensembl/NCBI | 解析区域边界、清洗、MMseqs2 family-disjoint 划分、离线结构特征 | 注释版本差异、isoform 冗余、与基座预训练语料重叠 |
| mRNA 原生冻结基座 | Helix-mRNA、mRNABERT、Orthrus/Orthrus+MLM、LAMAR、HydraRNA | 只作为 frozen encoder 或线性探针对照 | 权重许可、预训练语料与测试集重叠、token 粒度不一致 |
| 标准化下游评测 | mRNABench 2026 版 | 与基座表征/参数效率对齐，优先 linear probe 和官方 split | PyPI 为 AGPLv3+，需隔离依赖；版本 2025/2026 指标不一致 |
| TE/5UTR 金标准 | Sample et al. 2019 MPRA + Optimus 5-Prime/FramePool 等 oracle | T2/T3 训练与独立 oracle 评测 | 固定 50 nt 随机库 OOD；human UTR 与 random UTR 分布不同 |
| CDS/密码子优化 | CodonBERT/mRNA-LM 相关公开 benchmark、HPA 高表达转录本、GENCODE/RefSeq CDS | T4 数据、CAI/tAI/GC3/MFE 标签、baseline 对照 | CodonBERT 名称有两类论文；表达标签跨数据集异质 |
| T7 元件库 | Rfam、IRESite/IRESbase、miRBase/TargetScan、ARE/polyA 文献库 | motif 插入/切除目标与检出指标 | 元件上下文依赖强，单 motif 命中不等于功能成功 |

单 GPU 推荐策略：

- v1 训练上限：5UTR≤128 nt，CDS≤1536 nt，3UTR≤256 nt，总长约 1920 nt。
- 评测扩展上限：5UTR≤256 nt，CDS≤3072 nt，3UTR≤512 nt，只在采样/评测小 batch 使用。
- 预训练使用 frozen backbone embedding cache，生成头 5-20M 可训练参数。
- 任何公开 benchmark 的 test 集都不得进入 Stage A 训练语料；若 frozen 基座可能见过 test，只能主张“frozen pretrained feature transfer”，不能主张严格 zero-shot 无泄漏。

## 2. mRNA 原生基座与预训练语料

### 2.1 mRNABERT

公开信息：Nature Communications 2025 论文 *mRNABERT: advancing mRNA sequence design with a universal language model and comprehensive dataset*，摘要称其预训练于 1800 万 mRNA 序列，采用双 tokenization：UTR 用 nucleotide，CDS 用 codon，并引入蛋白序列语义的跨模态对比学习。参见 DOI `10.1038/s41467-025-65340-8` 与公开摘要页。

| 项 | 调研结论 |
|---|---|
| 规模 | 约 18M mRNA sequences，当前公开摘要声称为当时最大 mRNA 数据集。 |
| token 粒度 | UTR=nt，CDS=codon，天然贴合本项目“UTR 自由画布 + CDS 密码子格”。 |
| 权重/许可 | 需核验官方仓库与模型权重许可证；论文开放不等于权重可商用。 |
| 本项目角色 | 一线 frozen encoder；也可作为 mRNABench/下游属性预测强基座。 |
| 质量风险 | 预训练语料可能覆盖 RefSeq/GENCODE、Sample human UTR 与 mRNABench 数据。 |
| 泄漏控制 | 用 mRNABERT 做 frozen feature 时，报告“pretrained overlap audit”：对 test 序列做 k-mer/minhash/MMseqs2 最近邻距离，不能把性能完全归因于 MEF。 |
| 单 GPU 裁剪 | 只缓存 nt 对齐后的 per-token embedding；CDS codon embedding 上采样到 3 个 nt 位置或保留 codon 位点双视图。 |

对 MEF 的意义：mRNABERT 的双 tokenization 与本项目算子文法高度一致，但 mRNABERT 本身主要是表征/预测模型。MEF 的差异是把同义密码子约束、UTR 插删、长度控制写入 CTMC 编辑速率场，面向生成与编辑。

### 2.2 Helix-mRNA

公开信息：Helix-mRNA 2025 arXiv，混合 state-space 与 attention，模型权重开放在 HuggingFace `helical-ai/helix-mRNA`，模型卡/论文许可证为 CC BY-NC-SA 4.0。论文声称可处理比已有方法长 6 倍的序列，约用现有 foundation model 10% 参数，采用单核苷酸 tokenization 并用特殊字符标记 codon separation。

| 项 | 调研结论 |
|---|---|
| 规模 | 多门类 mRNA 预训练语料，论文未在摘要中给出最终条数，需下载 supplement 核验。 |
| token 粒度 | nt 级，加入 codon separator `E`，保留 CDS 相位信息。 |
| 权重/许可 | 权重可获取；CC BY-NC-SA 4.0 表示非商业限制，需避免商用场景。 |
| 本项目角色 | 默认候选 frozen encoder，适合单 GPU 原型，因为长上下文与参数效率较好。 |
| 质量风险 | preprint 结果需复现；Mamba/SSM 依赖可能增加环境复杂度。 |
| 单 GPU 裁剪 | 先用 Helical package 离线导出 embedding；若依赖安装复杂，v1 用 `backbone=none` 冒烟，再接入 Helix。 |

对 MEF 的意义：Helix-mRNA 的长上下文和 codon separator 适合全长 mRNA，但其自身不是编辑生成模型。MEF 可把 Helix 表征投影为每位点编辑速率。

### 2.3 Orthrus / Orthrus+MLM

公开信息：Orthrus 是成熟 RNA foundation model，预训练包含剪接 isoform 与跨物种同源关系的对比学习。mRNABench 2026 OpenReview 版本报告 Orthrus+MLM 在 11 个数据集/79 任务中有强表现，约以 700 倍更少参数达到或匹配 SOTA。

| 项 | 调研结论 |
|---|---|
| 规模 | Orthrus 使用 10 个物种的 mature RNA 注释和 Zoonomia 400+ 哺乳动物同源信息；具体序列数需以作者仓库为准。 |
| token 粒度 | mature RNA 序列级，偏表征学习。 |
| 权重/许可 | 需核验 GitHub/模型权重；mRNABench package 是 AGPLv3+，使用时隔离在评测环境。 |
| 本项目角色 | frozen encoder 对照与 ortholog-aware coupling 的科学参照。 |
| 质量风险 | 预训练目标强调 isoform/ortholog 表征，不一定覆盖全长设计约束。 |
| 单 GPU 裁剪 | 用官方 embedding 接口做离线缓存；只训练 MEF 头。 |

对 MEF 的意义：Orthrus 的进化/isoform 对比学习支持本项目“直系同源耦合”的合理性。MEF 不应重复造一个大基座，而应利用 Orthrus 表征，额外学习变长编辑过程。

### 2.4 LAMAR

公开信息：Genome Biology 2025 *A foundation language model to decipher diverse regulation of RNAs*。摘要与综述页显示 LAMAR 预训练于约 1500 万序列，来自基因组与转录组，覆盖约 225 个物种/来源；总 token 量约 267B，模型有 2k/4k 上下文版本，使用 MLM 目标。

| 项 | 调研结论 |
|---|---|
| 规模 | 约 15M RNA sequences，约 267B tokens。 |
| token 粒度 | nt 级，包含 `<cls>`/`<eos>`，使用 RoPE。 |
| 权重/许可 | Genome Biology 文章开放；模型权重和数据下载需核验。 |
| 本项目角色 | mRNA 原生/广义 RNA 基座候选，适合 TE、半衰期、IRES 等调控任务。 |
| 质量风险 | 语料包括非 mRNA 与 genomic windows，mRNA 原生性弱于 mRNABERT/Helix-mRNA 时需做 ablation。 |
| 单 GPU 裁剪 | 优先使用 2k context 或预先截断到 MEF bucket。 |

### 2.5 HydraRNA

公开信息：Genome Biology 2025 *HydraRNA: a hybrid architecture based full-length RNA language model*。论文摘要称其使用 bidirectional state-space + multi-head attention 混合架构，在 mRNA 与 non-coding RNA 上预训练，覆盖 10 类 RNA 任务，包括 mRNA 稳定性、翻译效率、RBP、剪接、多聚腺苷酸化等。公开综述页称预训练约 2800 万 RNA 序列，长序列切到 4096 nt。

| 项 | 调研结论 |
|---|---|
| 规模 | 约 28M RNA sequences，来自 RNAcentral 与 NCBI，含 mRNA 与 ncRNA。 |
| token 粒度 | nt 级，span masking。 |
| 权重/许可 | 需核验代码/权重；论文开放。 |
| 本项目角色 | 长 RNA 表征基座与 ncRNA/mRNA 混合基座对照。 |
| 质量风险 | 非 mRNA 占比可能稀释 CDS/UTR 语义；切段会弱化全长区域边界。 |
| 单 GPU 裁剪 | 如果权重依赖复杂，先作为文献对照，不作为 v1 默认基座。 |

### 2.6 mRNA-LM 与 CodonBERT 系列

公开信息：Sanofi/CMU 的 CodonBERT Genome Research 2024 预训练于 1000 万 CDS/mRNA 序列，使用 codon token，用于 mRNA 疫苗相关稳定性、表达、半衰期等任务。mRNA-LM NAR 2025 把 CDS-only CodonBERT 扩展到全长 mRNA，组合 5UTRBERT、CodonBERT、3UTRBERT。

另有 Bioinformatics 2024 论文也名为 CodonBERT，使用 cross-attention 做 codon optimization，训练数据涉及 HPA 高 TPM RNA-seq。两个“CodonBERT”不是同一工作，后续引用必须写全标题或 DOI，避免 baseline 混淆。

| 项 | 调研结论 |
|---|---|
| 规模 | Sanofi CodonBERT：约 10M CDS/mRNA；mRNA-LM：多物种全长 mRNA，具体规模需 supplement 核验。 |
| token 粒度 | CodonBERT=codon；mRNA-LM=区域模型集成。 |
| 权重/许可 | 需核验代码和权重；论文开放不等于权重开放。 |
| 本项目角色 | T4 密码子优化强 baseline；CDS 表征负/正对照；CodonBERT benchmark 可作表达/稳定性标签来源。 |
| 质量风险 | 只看 CDS 会忽略 UTR；不同 CodonBERT 数据集口径异质。 |
| 单 GPU 裁剪 | T4 只需 CDS/codon 级，训练成本最低；用于 MEF adapter 阶段优先落地。 |

### 2.7 eFold / RNAndria：cross-family 泛化的数据启发

公开信息：Science Advances 2026 论文 *Diverse database and machine learning model to narrow the generalization gap in RNA structure prediction*（DOI `10.1126/sciadv.adz4967`）。eFold 团队用 DMS-MaPseq/chemical probing 建立了 1098 个 primary microRNA 和 1456 个 human mRNA regions 的二级结构模型，并把这些新数据与 300,000+ 多来源 RNA secondary structures 合并训练。论文核心结论是：**merely expanding database size is insufficient for generalization across families；真正提升跨 family 泛化的是 RNA structure/data 的 diversity 和 complexity**。他们还用 viral mRNA modules、long ncRNA 等 long/diverse test sets 暴露常规模型在 out-of-domain RNA family 上的退化。

| 项 | 对 MEF 的启发 |
|---|---|
| 数据规模 | 规模是必要条件，但不是充分条件；不能只把 RefSeq/GENCODE 条数变大就宣称 data-scale 泛化。 |
| 数据多样性 | 训练语料要显式覆盖 gene/protein family、UTR motif family、长度 bucket、GC/GC3 bucket、region composition、species/source、structure/accessibility proxy 等轴。 |
| 数据复杂度 | 需要纳入长 3'UTR、复杂 5'UTR、high/low structure proxy、uAUG/Kozak/polyA/ARE motif-rich transcripts，而不是只采样最常见、最短、最干净的 protein-coding isoforms。 |
| 下游评测 | 除 random transcript split 外，必须报告 cross-family / cross-domain stress set：held-out gene/protein clusters、held-out motif families、held-out length/structure buckets、cross-source GENCODE→RefSeq、必要时 cross-species。 |
| claim 边界 | “head256→head1024 变好”只能说明当前切片规模趋势；若没有 diversity/complexity holdout 轴，不能写成真正跨家族泛化或 data-scale law。 |

迁移到 MEF 的原则：eFold 是 RNA structure prediction 论文，不是 mRNA design baseline；本项目只迁移其**数据治理与泛化评测原则**，不把 eFold 作为 full-length mRNA 生成或 TE/stability SOTA 对照。

## 3. 公共数据源清单

### 3.1 RefSeq

NCBI RefSeq 提供人工/自动注释的 transcript 和 protein 记录，当前 FAQ 更新到 2025-05-30。人类、鼠等物种可从 RefSeq FTP 获取 transcript/protein FASTA、GenBank flat file 与 GFF3/GTF 注释。mRNA 记录常含 5UTR/CDS/3UTR，需要用 feature table 提取区域。

| 项 | 调研结论 |
|---|---|
| 规模 | 物种依赖；人类 curated NM_ + predicted XM_ 合计可达 10^5 级转录本，多物种可扩展到百万级。 |
| 获取 | NCBI RefSeq FTP、NCBI Datasets、E-utilities。 |
| 许可 | NCBI 公共数据，需遵守 NCBI 使用条款与引用；批量下载遵守频率限制。 |
| 本项目角色 | Stage A 主语料、T4 protein-CDS 配对、ortholog coupling 的目标物种侧。 |
| 质量风险 | XM_ 预测转录本质量不如 NM_；UTR 边界可能缺失；transcript 支持证据不均；版本更新会改变序列。 |
| 清洗 | 只保留 A/C/G/U/T/N 可归一化序列；ATG/AUG 起始、单一规范终止、CDS 长度为 3 的倍数、无提前终止。 |
| 单 GPU 裁剪 | v1 优先 human NM_，再加入 mouse/vertebrate；按基因或 MMseqs2 cluster 划分，保留每基因最长/代表 isoform 做小集。 |

### 3.2 GENCODE

GENCODE 提供人/鼠高质量综合注释和 transcript FASTA。公开课程资料显示 2026 年人类当前版本可为 release 49；正式实验必须 pin 具体 release，例如 `GENCODE v49 GRCh38`，并记录 SHA256。

| 项 | 调研结论 |
|---|---|
| 规模 | 人类 transcript FASTA 约 10^5 级；protein-coding transcript 子集较小。早期模型如 3UTRBERT 使用过约 108,573 unique mRNA transcripts。 |
| 获取 | `ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_*`。 |
| 许可 | GENCODE/Ensembl 公共资源，需遵守 EBI/Ensembl 使用条款与引用。 |
| 本项目角色 | 高质量人类 full-length mRNA 与区域边界基准；T1/T4/T5/T6/T7 构造来源。 |
| 质量风险 | comprehensive 包含 partial/nonsense-mediated decay 等转录本；basic 子集更保守但覆盖少。 |
| 单 GPU 裁剪 | 默认 `protein_coding + basic tag`；每 gene 取 principal/longest isoform 做小语料，full 语料保留 isoform 但按 gene cluster 切分。 |

### 3.3 Ensembl Compara / OrthoDB / OMA

用于构建 ortholog-aware coupling。目标是得到 `(other_species_mRNA, target_species_mRNA)` 对，而不是单条序列。

| 项 | 调研结论 |
|---|---|
| 规模 | 取决于物种集；human-mouse/rat/zebrafish 等高质量 ortholog 可先构建 10^4-10^5 对。 |
| 获取 | Ensembl Compara BioMart/API，OrthoDB，OMA。 |
| 许可 | 各库公共使用条款不同，需保留下载版本与 citation。 |
| 本项目角色 | 第三路 hybrid coupling，让模型学习自然 indel 与同义替换轨迹。 |
| 质量风险 | UTR 同源性弱且边界差异大；一对多 ortholog 会引入歧义；物种表达环境不同。 |
| 单 GPU 裁剪 | v1 只做 one-to-one ortholog，CDS 以 protein alignment 锚定，UTR 用 banded alignment。 |

### 3.4 Sample et al. 2019 MPRA

Nat Biotechnology 2019 *Human 5' UTR design and variant effect prediction from a massively parallel translation assay*。数据包括约 280,000 个随机 50 nt 5UTR reporter library，标签为 mean ribosome load (MRL)，还包括约 35,000 个截短 human 5UTR 与 3,577 个自然变体。Optimus 5-Prime 是常用 CNN oracle。

| 项 | 调研结论 |
|---|---|
| 规模 | random 50 nt 约 280k；truncated human 5UTR 约 35k；自然变体约 3.6k。 |
| 获取 | 论文补充、官方/复现仓库、Zenodo/NCBI GEO 需核验。 |
| 许可 | 论文 PMC 可读；数据/代码许可证需逐项核验，特别是商业场景。 |
| 本项目角色 | T2 TE 优化和 T3 5UTR 设计金标准；也可训练独立 TE oracle。 |
| 质量风险 | random library 与天然 human 5UTR 分布差异大；固定 50 nt 限制长度自由任务；同一 oracle 训练/评测会过拟合。 |
| 切分 | 使用论文/官方 random-test 与 human held-out；MRL z-score 标准化并记录均值方差。 |
| 单 GPU 裁剪 | 全量可单 GPU 训练小 oracle；MEF adapter 可先取 50k train + full val/test 做快速迭代。 |

### 3.5 mRNABench

公开信息存在版本差异：

- 2025 bioRxiv/PMC/PubMed 摘要：10 个数据集、59 个预测任务、18 个模型家族、约 135K 实验。
- 2026 OpenReview ICLR Workshop 版本：11 个数据集、79 个预测任务、24 个模型家族、约 259K 实验，并强调 10 seeds、homology-aware splits、Orthrus+MLM 在 7/11 数据集达到或匹配 SOTA。
- PyPI `mrna-bench` 1.2.2：2025-07-14 发布，license 为 AGPLv3+，数据集在 HuggingFace collection。

本文按 2026-07-11 的最新研究方案采用 2026 OpenReview 口径，同时在实验表中记录 package 版本。

| 项 | 调研结论 |
|---|---|
| 规模 | 11 数据集/79 任务，重点覆盖 mature mRNA 的 TE、half-life、localization、GO 等属性。 |
| 获取 | GitHub `morrislab/mRNABench`、PyPI、HuggingFace collection。 |
| 许可 | 论文 CC BY 4.0；PyPI package AGPLv3+。若只读取数据，仍需检查数据集各自来源许可。 |
| 本项目角色 | 标准化下游评测与参数效率对照；冻结基座 + 轻量头与 mRNA 原生基座比较。 |
| 质量风险 | 主要是 property prediction，不是生成；评测集与 mRNA 原生基座预训练语料高度可能重叠。 |
| 单 GPU 裁剪 | 只运行与本项目最相关的 TE、half-life、stability、localization 子任务；linear probe 10 seeds。 |

### 3.6 CodonBERT/mRNA 稳定性与表达 benchmark

CodonBERT 系列数据覆盖 CDS/mRNA 稳定性、表达、半衰期、蛋白产量等，但来源横跨 HPA、mRNA vaccine 语料、多物种 CDS、商业/公开优化序列。

| 项 | 调研结论 |
|---|---|
| 规模 | Sanofi CodonBERT 预训练约 10M CDS/mRNA；Bioinformatics CodonBERT 训练数据来自 HPA high-TPM 及 JCAT 优化比例不同的数据集。 |
| 获取 | 论文链接、supplement、GitHub/Zenodo 需核验。 |
| 许可 | 需逐数据集核验，尤其 HPA 与商业优化序列。 |
| 本项目角色 | T4 密码子优化 baseline 与标签来源；protein-CDS 配对数据筛选规则参考。 |
| 质量风险 | CAI/GC/MFE 是 surrogate，不等于真实表达；高 TPM 可能反映组织特异性而非单纯密码子偏好。 |
| 单 GPU 裁剪 | 以人类高表达 protein-coding CDS 子集训练 T4 adapter；按 protein MMseqs2 split。 |

### 3.7 UTR-LM、mRNA2vec、UTRGAN、UTailoR

这些不是主预训练语料来源，但必须纳入下游 baseline 与数据泄漏审计。

| 来源 | 规模/数据 | 角色 | 风险 |
|---|---|---|---|
| UTR-LM, Nat Machine Intelligence 2024 | 5UTR language model，预训练使用 Ensembl、Sample et al. synthetic library、Cao 等 5UTR/TE 数据 | T2/T3 表征 baseline，TE oracle 对照 | 若用 Sample 数据训练，再用 Sample 评测，需官方 split 与独立 oracle |
| mRNA2vec, arXiv 2024 | 5UTR+CDS，使用 MFE/secondary structure 辅助 pretext | 结构辅助头设计参考 | 主要是 embedding/预测，不是变长编辑生成 |
| UTRGAN, Bioinformatics Advances 2025 | GAN 生成 5UTR，报告 predicted expression/MRL/TE 提升与部分体外验证 | T3/T7 UTR 生成 baseline | GAN 不天然支持最小编辑与 CDS 约束；oracle 过拟合风险 |
| UTailoR, iScience 2025 | 判别模型 + 生成模型优化 5UTR，报告约 200% TE 提升 | T2/T3 专项 baseline | 任务范围局限 5UTR，需检查代码/数据开放性 |

### 3.8 2026 生成式 mRNA 基线

| 来源 | 公开信息 | 本项目定位 |
|---|---|---|
| mRNAutilus, arXiv 2026 | masked discrete diffusion + Monte Carlo Tree Guidance，全长 mRNA 多目标生成，训练于 millions full-length mRNAs，报告 luciferase/SARS-CoV-2 Spike 等表达提升 | T1 全长生成与多目标优化强 baseline；MEF 不主张 v1 全面碾压，应突出变长编辑任务与参数效率 |
| ProMORNA, arXiv 2026 | protein-conditioned BART + multi-objective GRPO（MO-GRPO：聚合前对每个 metric 做 relative-advantage 标准化），约 6M natural protein-mRNA pairs，约 45M 参数；在 held-out luciferase 上改进 predicted half-life vs TE 的 in-silico Pareto frontier | T1/T4 全长 protein-conditioned baseline；直接启发 MEF upgrade #1（多目标）与 #3（蛋白条件）；对比时关注是否支持最小编辑/元件插删 |
| T3PO-mRNA, OpenReview 2026 | reward-guided masked diffusion fine-tuning | 多目标扩散 baseline，作为“定长/掩码扩散 vs 编辑流”核心消融 |
| GEMORNA, Science 2025 | encoder-decoder CDS（把蛋白→CDS 视作翻译，同义约束解码）+ decoder-only UTR，两者再组合为全长；自然 CDS/UTR 语料预训练，高 MRL 5'UTR / 高稳定性 3'UTR 微调；wet-lab 报告蛋白 ~41x、抗体 ~128x、circRNA ~121x | 顶级 wet-lab 证据基线；本项目若无湿实验，不应在实验结论中直接声称优于其生物效果；T4/T5 分区口径对齐对象 |
| mRNA-GPT, ICLR 2026 投稿 | decoder-only，端到端联合 5'UTR+CDS+3'UTR；预训练于 **10M 全长自然 mRNA（多物种，含区段标注 [5UTR]/[CDS]/[3UTR]/[EOS]，UTR 用预训练 tokenizer 分段、CDS 按 codon）**；迭代 oracle-reward 优化；支持单区/全长/条件生成；报告 predicted translation rate 高于 LinearDesign 和 GEMORNA，且全长 diversity 更高 | **最直接全长竞品**；MEF 差异化=constrained edit-flow（protein/frame/budget 硬约束 + 可解释 edit distance）而非自由 AR 生成；head-to-head 必须同时报告 TE delta 和约束满足率 |
| RNAGenScape, ICML 2025 GenBio / arXiv 2025 | property-guided manifold Langevin dynamics：organized autoencoder（按目标属性组织 latent）+ manifold projector（每步投回流形）；从**真实 mRNA 序列**出发做优化/插值而非 de novo；跨 **3 个真实 mRNA 数据集（跨两个数量级规模，含斑马鱼 5'UTR），最小仅 ~2000 点（用 SUGAR 填补 undersampled 流形）**；报告 median property gain +148%、success rate +30%、on-manifold plausibility、可解释潜轨迹 | 哲学与 MEF 最接近（“optimize, not invent”，从真实序列局部优化）；连续 latent 轨迹 vs MEF 离散 constrained edit-flow；共享 5'UTR TE 任务对比 property gain/success/可解释性 |
| codonGPT, Nucleic Acids Research 2025 | GPT-2 架构 codon-level 生成 LM + 推理时同义 logit masking（100% 蛋白保真）+ RL 多目标（CAI/GC/ΔG/codon entropy/repeat penalty）；训练于 **338,417 条模型生物 mRNA（Ensembl Release 64 CDS，64 codon + 3 特殊 token）**；案例 HLA-A/ACTB/GFP/β-lactamase/EPO | CDS-only 生成+RL baseline；同义掩码思路与 MEF 的 synonymous CDS mask 一致；MEF 覆盖全长 + UTR 编辑，codonGPT 仅 CDS |

关键数据集获取与许可备注（用于后续 leakage-free 对齐，均标“需下载/许可核验”）：

- mRNA-GPT 的 10M 全长语料与 ProMORNA 的 6M protein-mRNA 配对均来自公共数据库汇编（RefSeq/Ensembl 系），若要复现须自建同口径 pipeline 并做 family-disjoint split，不能直接混入 MEF 训练集。
- codonGPT 使用 Ensembl Release 64 CDS（可获取），是可复现性较高的 CDS-only 对照来源。
- RNAGenScape 使用斑马鱼等真实小数据集（含 5'UTR TE），适合作为 MEF 5'UTR 优化任务的低数据对齐场景。
- 以上任何 test 分片都不得进入 Stage A 训练；只要外部 FM 可能见过对应序列，MEF 只能主张 feature transfer，不能主张 zero-shot 无泄漏（沿用 §6 泄漏协议）。

### 3.9 eFold-inspired cross-family 数据计划

eFold 给本项目最直接的提醒是：**data scale-up 要拆成 size、diversity、complexity 三条轴，而不是只统计 records 数量**。因此 MEF 后续数据计划需要在 manifest 和 audit 中增加以下 profile：

| 轴 | 推荐统计 | 进入训练/评测的用法 |
|---|---|---|
| family diversity | `n_gene_clusters`, `n_protein_clusters`, 每 cluster transcripts 数，largest cluster fraction | train/val/test 必须 family-disjoint；额外构造 held-out cluster panel |
| region diversity | 5'UTR/CDS/3'UTR 长度 bucket，region truncation/drop counts，full transcript length bucket | T1/T6/T7 必须按长度/区域复杂度分层报告 |
| motif diversity | uAUG、Kozak、polyA、ARE、miRNA seed、IRES-like motif bucket | T2/T3/T7 构造 motif-family holdout，避免只学常见 motif |
| structure complexity proxy | start accessibility、MFE proxy、GC/GC3、long-range pair proxy、多尺度 k-mer/codon-pair spectrum | T2/T5 需要 high-structure 与 low-structure stress split |
| source/species diversity | GENCODE vs RefSeq、NM_/XM_、human/mouse/zebrafish/viral-like external panel | 报告 cross-source generalization gap，不把 GENCODE-only 结果外推到 RefSeq/multi-species |
| task complexity | single-region UTR、CDS-only、full-transcript mixed edit、length/motif constraints | 下游任务必须覆盖从简单到复杂的 curriculum，而不是只汇报 easy in-family split |

新增 claim gate 建议：

```text
ready_for_cross_family_generalization_claim =
  family_disjoint_split_ready
  AND diversity_profile_complete
  AND complexity_bucket_eval_complete
  AND cross_source_or_cross_species_panel_complete
  AND all_hard_constraints_exact_1
```

在该 gate 之前，任何“scale data 后更好”的表述都只能写成**数据规模/切片趋势信号**，不能写成 cross-family 泛化结论。


## 4. 数据清洗与标准 manifest

所有 Stage A 序列必须落为统一记录：

```text
transcript_id, gene_id, source, release, species, split_cluster,
seq_5utr, seq_cds, seq_3utr, protein_seq,
len_5utr, len_cds, len_3utr,
gc_5utr, gc_cds, gc3, cai, mfe, start_accessibility,
sha256_raw, sha256_clean
```

清洗规则：

1. T→U 归一。
2. 仅允许 `A/C/G/U`，含 `N` 的序列默认丢弃；若用于基座复现可保留但不得进入主训练。
3. CDS 以 AUG 起始，终止 codon 属于 `{UAA,UAG,UGA}`。
4. CDS 长度为 3 的倍数，无提前终止。
5. 5UTR/CDS/3UTR 边界明确。
6. 默认长度：5UTR≤128 nt，CDS≤1536 nt，3UTR≤256 nt；超长可截断只用于 ablation，不进入主结论。
7. MMseqs2 easy-cluster `min-seq-id=0.8`，按 cluster 划分 train/val/test，保证 train/test cluster 零交集。

每次构建输出：

- `manifest.jsonl`：每分片 SHA256、序列数、长度分布、区域比例、丢弃计数。
- `splits/{train,val,test}.idx`：固定 seed、family-disjoint。
- `leakage_report.json`：训练集到测试集最近邻距离、共享 gene/protein/transcript ID、与外部 benchmark 的 overlap。

## 5. 单 GPU 数据裁剪策略

### 5.1 Stage A 预训练小集

| 配置 | 序列数 | 长度 | 目标 |
|---|---:|---:|---|
| smoke | 1k-5k | 5UTR≤64, CDS≤768, 3UTR≤128 | 调通 DP、loss、采样与合法性 |
| small | 50k | 5UTR≤128, CDS≤1536, 3UTR≤256 | 单 GPU 首个可报告模型 |
| medium | 200k-500k | bucket by length | 主 ablation 与参数效率 |
| full-public | 视资源 | 多物种 + 长 bucket | 服务器扩展，不作为 Task 0 要求 |

embedding cache 估算：

$$
\mathrm{size}\approx N\cdot L\cdot d\cdot b,
$$

其中 `b=2` bytes for fp16。若 `N=50k,L=1920,d=768`，单基座 cache 约 147 GB，必须分片存储并支持按 bucket 懒加载。smoke/small 阶段可只缓存 `d=384` 投影后的 embedding。

### 5.2 下游任务裁剪

- T2/T3：Sample 2019 全量可训练，但 MEF adapter 先用 50k train、full validation/test。
- T4：每 protein family 最多取 1-3 个 transcript，降低 isoform 冗余。
- T5/T6/T7：从测试外 train 语料构造合成目标；每类任务先 10k 对，确保评测指标可跑通。
- mRNABench：只跑与 mRNA 设计直接相关的 TE/half-life/stability/localization 子集，10 seeds linear probe。

## 6. 数据泄漏风险与规避协议

| 风险 | 具体场景 | 规避 |
|---|---|---|
| 预训练-评测重叠 | mRNABERT/Helix/Orthrus/LAMAR/HydraRNA 可能见过 GENCODE/RefSeq/mRNABench 序列 | 报告 frozen backbone overlap audit；主张限定为 feature transfer，不声称未见过序列 |
| train-test 同源泄漏 | 同一 gene 的 isoform 或 paralog 分入不同 split | 按 MMseqs2 cluster/gene/protein family 划分，而不是随机 transcript 划分 |
| oracle 泄漏 | 用 Optimus 5-Prime 优化后再用同一模型评测 | 优化 oracle 与评测 oracle 分离；第二 oracle 交叉验证 |
| benchmark 版本泄漏 | mRNABench 2025 与 2026 任务不同 | 固定 package/data version，论文表写明任务数与版本 |
| synthetic-natural 混淆 | Sample random 50 nt 与天然 human UTR 分布差异 | random-test 和 human held-out 分开报告，不把 random 表现外推到天然 UTR |
| surrogate 指标过度解释 | CAI/MFE/GC3 提升被解释为真实表达提升 | 报告为 in-silico surrogate；若无湿实验，结论限定在预测/分布指标 |
| 方向性错误增强 | mRNA 做 reverse-complement augmentation | 明确禁止反向互补增强；mRNA 有方向、区域与翻译框 |

## 7. 数据到任务的映射

| 任务 | 主要数据 | 辅助数据 | 必须排除 |
|---|---|---|---|
| T1 全长生成 | RefSeq/GENCODE full-length | mRNABench distribution metrics、mRNAutilus/GEMORNA 文献指标 | test cluster、benchmark held-out |
| T2 TE 优化 | Sample 2019 MPRA wildtype/variant | Optimus/FramePool/UTR-LM oracle | 评测 oracle 训练样本 |
| T3 UTR 设计 | Sample 2019 + natural 5UTR | Kozak/uAUG/IRES motif labels | 与 T2 test 重叠的 generated targets |
| T4 密码子优化 | RefSeq/GENCODE protein-CDS pair, CodonBERT benchmark | HPA high-TPM CAI reference, LinearDesign/DERNA | protein family 相似 test |
| T5 最小编辑改造 | RefSeq/GENCODE + synthetic target constraints | disease SNV/UTR variant sets | 同一 variant 训练/测试重复 |
| T6 长度可控设计 | RefSeq/GENCODE length distribution | Sample/UTR length models | 训练中直接见过目标长度模板 |
| T7 元件移植/切除 | Rfam/IRESite/IRESbase/miRBase/ARE/polyA libraries | natural host UTR contexts | motif 数据库与 eval motif set 不分离 |

## 8. 数据资产优先级

P0 必做：

- GENCODE/RefSeq human protein-coding full-length mRNA，带区域边界。
- Sample 2019 MPRA，保留官方 split。
- mRNABench 相关子任务的只读评测环境。
- Codon table、human codon usage、CAI reference。

P1 推荐：

- Helix-mRNA frozen embedding cache。
- mRNABERT frozen embedding cache或至少文献/权重可用性审计。
- Ensembl Compara one-to-one ortholog pairs。
- Rfam/IRES/miRNA/ARE/polyA 元件库。

P2 扩展：

- HydraRNA/LAMAR/Orthrus 多基座 ablation。
- mRNAutilus/ProMORNA/T3PO-mRNA baseline 复现。
- 多物种大规模语料与长上下文模型。

## 9. 可落地性约束

1. 文档和 manifest 必须记录 release、URL、下载时间、SHA256，避免“数据漂移”导致实验不可复现。
2. 所有外部数据先进入 `raw/`，清洗后进入 `processed/`，训练只读 `processed/manifest.jsonl`。
3. 原始数据许可未核验前，不进入可公开发布模型训练集。
4. 单 GPU 训练默认只读 embedding cache，不在线前向大基座。
5. 任何 SOTA 主张必须同时报告：同 split、同 oracle、10 seeds、bootstrap 95% CI、paired significance test、可训练参数量。

该数据方案的关键不是堆更多序列，而是让每条序列都能回答“这里是 UTR 画布还是 CDS 密码子格”。只有区域边界、相位、同义集合、结构特征、split 与泄漏报告齐全，mRNA-EditFlow 的变长编辑故事才可被严格评测。
