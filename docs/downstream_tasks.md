# mRNA-EditFlow 下游任务定义与评测协议

日期：2026-07-11

本文档对应 Task 0.3，定义 mRNA-EditFlow 的 T1-T7 任务、数据、指标、SOTA 对比与创新性边界。统一科学故事为：**mRNA 设计 = 区域条件下的变长编辑；CDS 是帧锁定密码子格，UTR 是长度自由调控画布**。

本项目不把 mRNA 简化成普通字符串生成。mRNA 的三区域约束不同：

$$
x=(x^{5UTR},x^{CDS},x^{3UTR}),
$$

其中 UTR 是长度自由的调控画布，合法操作为 nt 级 `insert/delete/substitute`；CDS 是帧锁定密码子格，默认只允许同义密码子替换：

$$
c_j\rightarrow c'_j,\quad aa(c'_j)=aa(c_j).
$$

因此 MEF 的核心能力不是“更会预测下一个 token”，而是把区域编辑文法写进连续时间编辑速率场。T1-T4 对齐已有 mRNA 设计任务，T5-T7 是变长编辑能力真正打开的新任务族。

## 1. 通用任务协议

### 1.1 统一输入输出

所有任务的候选输出为规范化 mRNA record：

```text
seq_5utr, seq_cds, seq_3utr, protein_seq,
region_boundaries, codon_phase, edit_script, constraints, metadata
```

合法性硬约束：

1. 序列字符属于 `A/C/G/U`。
2. CDS 从 `AUG` 起始，以 `{UAA,UAG,UGA}` 单一终止。
3. CDS 长度为 3 的倍数。
4. CDS 无提前终止。
5. strict CDS 编辑任务中 `Translate(CDS_out)=Translate(CDS_in)`。
6. UTR 操作不跨越 CDS 边界。

### 1.2 通用评价口径

主实验必须报告：

- 10 seeds，至少 5 seeds 用于早期 fast-loop。
- mean ± bootstrap 95% CI。
- 同一输入上的 paired bootstrap 或 Wilcoxon signed-rank，必要时 Holm-Bonferroni 多重校正。
- 可训练参数量、GPU 显存、吞吐、采样步数。
- 优化 oracle 与评测 oracle 分离；若使用内部 TE/结构头引导，评测必须使用独立模型。

对每个任务都记录编辑距离：

$$
d_{edit}(x,y)=n_{ins}+n_{del}+n_{sub}.
$$

对于 UTR 任务，`d_edit` 是 nt 级；对于 CDS strict 任务，`d_edit` 是 codon 级同义替换数；对于 mixed 任务，两者分别报告并给加权和：

$$
d_{mixed}=d_{UTR}^{nt}+\beta d_{CDS}^{codon}.
$$

### 1.3 SOTA 叙事边界

本文建议三腿 SOTA 主张：

1. **新能力轴**：T5/T6/T7 的最小编辑、长度控制、元件插删是 MEF 的核心差异，定长 diffusion/BERT 重打分结构上不自然。
2. **TE/UTR 专项**：在 Sample 2019/独立 oracle 上与 Optimus+GA、UTR-LM、UTRGAN、UTailoR、FramePool 类方法公平比较。
3. **参数效率**：冻结 mRNA 原生基座，只训练 5-20M 编辑头/adapter，与 mRNABench linear-probe/参数量口径对齐。

非目标：v1 不主张在全长 mRNA 原始生成所有指标上全面碾压 mRNAutilus、GEMORNA 或大规模自回归模型；若无湿实验，不宣称真实体内表达优于已做 wet-lab 验证的模型。

### 1.4 eFold-inspired cross-family 泛化协议

eFold 论文 *Diverse database and machine learning model to narrow the generalization gap in RNA structure prediction*（Science Advances 2026, DOI `10.1126/sciadv.adz4967`）给本项目的数据/下游任务设计提供了一个重要边界：**仅扩大数据库规模不足以跨 family 泛化；数据多样性和结构/任务复杂度才是关键**。迁移到 MEF 时，所有 T1-T7 任务都必须在常规 in-family split 之外增加 cross-family / cross-domain stress layer。

必须报告的 split 层级：

| 层级 | 定义 | 目的 |
|---|---|---|
| random transcript split | 随机或固定 seed 切分 transcript | 只作为 smoke / fast-loop，不支持泛化主张 |
| gene/protein-family split | 同一 gene、isoform cluster、protein family 不跨 train/test | 检验 CDS/protein 条件任务是否摆脱同源记忆 |
| motif-family split | uAUG/Kozak/polyA/ARE/miRNA seed/IRES-like motif family holdout | 检验 UTR 调控元件是否能迁移，而不是背诵常见 motif |
| length/complexity split | held-out long UTR/full transcript、high MFE/low accessibility、extreme GC/GC3 bucket | 检验复杂结构和长序列分布外泛化 |
| cross-source split | GENCODE train → RefSeq test，或 RefSeq train → GENCODE test | 检验 source annotation/domain shift |
| cross-species split | human train → mouse/zebrafish/viral-like external panel（只在许可和清洗足够时） | 检验真正跨域泛化，不能和 human-only claim 混写 |

核心指标除原任务指标外，还要报告：

$$
\mathrm{generalization\ gap}=\mathrm{metric}_{in\ family}-\mathrm{metric}_{cross\ family}.
$$

对优化型指标（如 TE delta、CAI delta）报告 gap 的绝对差；对约束指标（legal/protein/frame/budget）要求 cross-family 仍为 exact-1。若 cross-family panel 未完成，论文只能写“in-family / matched-split proxy evidence”，不能写“跨家族泛化”或“data scale law”。

对数据和任务的直接要求：

- data scale-up 表不再只列 `record_count`，还要列 `n_gene_clusters`、length bucket、GC/GC3 bucket、motif bucket、source/species bucket。
- T2/T3/T7 必须有 motif-family holdout；T4 必须有 protein-family holdout；T1/T5/T6 必须有 length/complexity holdout。
- frozen foundation model 对比必须使用同一 cross-family split；否则只能报告 feature-transfer trend，不能报告泛化优势。
- eFold 是 RNA structure prediction，不是 mRNA 设计生成 baseline；这里借鉴的是 data diversity / cross-domain evaluation 的方法学，不把其 F1 结构预测结果和 MEF TE/stability 指标直接比较。

## 2. T1 全长 mRNA 从头生成

### 2.1 目标

无条件或弱条件生成合法 full-length mRNA：

$$
p_\theta(x^{5UTR},x^{CDS},x^{3UTR}\mid s),
$$

其中 `s` 可为空，也可包含物种、长度 bucket、表达 bucket 或 protein family 先验。T1 的基础版本是无条件采样合法转录本，用于检验 MEF 是否学到 mRNA 的区域语法与全长分布。

### 2.2 输入输出

| 项 | 定义 |
|---|---|
| 输入 | 可选：物种、人类/鼠、长度 bucket、目标 GC 范围；无条件版本仅给 `BOS`/空状态 |
| 输出 | `5UTR + CDS + 3UTR`，含区域边界与合法 CDS |
| 训练数据 | RefSeq/GENCODE cleaned full-length mRNA，family-disjoint split |
| 生成模式 | empty-growth coupling 为主，corruption/ortholog coupling 为辅 |

### 2.3 指标

合法性：

$$
\mathrm{valid}=\mathbf{1}[\mathrm{AUG}\land \mathrm{stop}\land |CDS|\equiv0\pmod 3\land \neg\mathrm{premature\ stop}].
$$

分布真实性：

- k-mer JS divergence：

$$
JS(P_{gen}^{kmer}\|P_{test}^{kmer}).
$$

- codon usage KL：

$$
KL(P_{gen}^{codon}\|P_{test}^{codon}).
$$

- GC、GC3、长度、MFE、起始区可及性分布的 Wasserstein distance。
- embedding Fréchet distance：

$$
FID_e=\|\mu_g-\mu_t\|_2^2+\mathrm{Tr}(\Sigma_g+\Sigma_t-2(\Sigma_g\Sigma_t)^{1/2}).
$$

多样性与新颖性：

$$
\mathrm{diversity}=\mathbb{E}_{i\ne j}[d_{edit}(x_i,x_j)],
$$

$$
\mathrm{novelty}=\mathbb{E}_{x\sim gen}[\min_{y\in train}d_{edit}(x,y)].
$$

功能代理：

- independent TE/half-life oracle score 分布。
- 起始密码子上下游结构可及性。
- uAUG、Kozak、poly(A) signal 等统计是否落在天然范围。

### 2.4 SOTA 对比

| 基线 | 对比点 | 公平口径 |
|---|---|---|
| 自回归 Transformer LM | 全长生成常规基线 | 同训练数据、同长度 bucket、同参数量或报告参数差异 |
| Masked discrete diffusion | 核心算子消融 | 固定长度/padding 版本与 MEF 变长版本对比 |
| mRNAutilus | 2026 full-length masked diffusion + guidance | 若无法复现，用文献值并注明任务/数据不同 |
| ProMORNA | protein-conditioned full-length generation | 只在 protein-conditioned 变体中比较 |
| GEMORNA | 有 wet-lab 证据的强基线 | 不直接声称生物效果胜出，报告 in-silico 与协议差异 |

### 2.5 Edit Flow 独特优势

T1 的 MEF 优势不是单点合法率，而是生成过程的区域可解释性。每条输出都有编辑轨迹和区域操作预算：UTR 如何增长、CDS 如何保持帧、哪里发生同义替换。这使 T1 能自然衔接 T5-T7，而普通 AR/扩散模型通常只能输出最终序列，难以把“从哪里改、改多少、为什么合法”作为一等对象。

## 3. T2 翻译效率优化

### 3.1 目标

给定 wildtype 5UTR 或完整 mRNA 上下文，生成一个编辑后 5UTR，使独立 oracle 预测的 MRL/TE 提升，同时编辑尽量少：

$$
\max_{x'} f_{TE}^{eval}(x')-f_{TE}^{eval}(x),\quad
\mathrm{s.t.}\ d_{edit}(x,x')\le k,\ x' \in \mathcal{L}_{UTR}.
$$

### 3.2 输入输出

| 项 | 定义 |
|---|---|
| 输入 | wildtype 5UTR，可选 CDS 起始上下文、目标 TE bucket、编辑预算 `k` |
| 输出 | 编辑后 5UTR 与 edit script |
| 数据 | Sample 2019 MPRA random 50 nt、human held-out、可扩展 Cao/Ribo-seq 5UTR TE 数据 |
| 训练 | corruption-refinement coupling，构造 `(low TE UTR, high TE target/condition)` 配对 |

### 3.3 指标

主指标：

$$
\Delta TE=f_{TE}^{eval}(x')-f_{TE}^{eval}(x).
$$

单位编辑收益：

$$
\mathrm{gain/edit}=\frac{\Delta TE}{\max(1,d_{edit}(x,x'))}.
$$

达标率：

$$
\mathrm{Hit@}\tau=\mathbf{1}[f_{TE}^{eval}(x')\ge \tau].
$$

分布与机制指标：

- uAUG 数量变化、Kozak motif 强度、起始区 `[-30,+30]` MFE/可及性。
- 生成 UTR 到 train 最近邻距离，防止背诵高 TE motif。
- 与训练内部 TE 头的 Spearman，只作为诊断，不作为主评测。

### 3.4 SOTA 对比

| 基线 | 对比点 |
|---|---|
| Optimus 5-Prime + genetic algorithm | Sample 2019 原始优化范式 |
| FramePool/任意长度 MRL predictor + search | 长度外推 oracle |
| UTR-LM | 表征学习 + 下游 TE prediction |
| UTailoR | 5UTR 优化专项生成基线 |
| UTRGAN | 5UTR 生成 + predicted TE/MRL 提升 |

公平协议：优化可以使用一个 oracle，但评测必须使用另一个 independent oracle；Sample random-test 与 human held-out 分开报告。

### 3.5 Edit Flow 独特优势

TE 优化往往不是“重新生成一个全新高分 UTR”，而是对现有治疗性 construct 做小幅改造。MEF 的速率场能显式控制编辑预算，并通过 ins/del/sub 组合处理 uAUG 切除、Kozak 邻域微调、局部结构释放。定长 masked diffusion 可以替换 token，但对插入/删除调控元件、最小编辑 Pareto 前沿不自然。

## 4. T3 UTR 设计

### 4.1 目标

给定 CDS 上下文与目标表达/稳定性桶，条件生成 5UTR 或 3UTR：

$$
x^{UTR}\sim p_\theta(x^{UTR}\mid x^{CDS}, y_{target}, L_{target}).
$$

T3 与 T2 的区别：T2 是 refinement，T3 是设计，可以从空 UTR、生物模板或短 seed 开始。

### 4.2 输入输出

| 项 | 定义 |
|---|---|
| 输入 | CDS 起始上下文、目标 TE/表达 bucket、可选长度范围、可选 motif 约束 |
| 输出 | 5UTR 或 3UTR 序列，保持 CDS 不变 |
| 数据 | Sample 2019、GENCODE/RefSeq natural UTR、mRNABench TE/half-life 子任务 |
| 训练 | empty-growth 用于 de novo，corruption-refinement 用于模板改造 |

### 4.3 指标

- 目标达成度：`|f_eval(x)-y_target|` 或 bucket accuracy。
- 合法性：UTR 字符、长度范围、无跨区域编辑。
- 真实性：k-mer JS、motif 统计、GC、MFE、nearest train distance。
- 上下文一致性：结合 CDS 起始区的结构可及性，不只看孤立 UTR。

目标桶可定义为：

$$
\mathrm{bucket}(y)=
\begin{cases}
low,& y<q_{0.33},\\
mid,& q_{0.33}\le y<q_{0.66},\\
high,& y\ge q_{0.66}.
\end{cases}
$$

### 4.4 SOTA 对比

| 基线 | 对比点 |
|---|---|
| UTRGAN | 5UTR de novo 生成 |
| UTR-LM + search | 表征模型 + 采样/优化 |
| Optimus + GA | 固定长度 target expression 设计 |
| AR UTR LM | 生成质量与长度控制 |
| mRNAutilus/GEMORNA UTR 分支 | full-length 设计中的 UTR 生成能力 |

### 4.5 Edit Flow 独特优势

UTR 是长度自由画布，调控功能经常来自 motif 的出现、间距和局部结构。MEF 可以在采样中执行“插入一个 Kozak-like 上下文”“切除 uAUG”“缩短高结构区域”等动作，并把这些动作作为轨迹评测。AR 模型能生成序列，但不能自然保证从模板到设计的最小编辑或局部可控。

## 5. T4 密码子优化

### 5.1 目标

给定蛋白序列或已有 CDS，生成编码同一蛋白的 CDS，提高 CAI/tAI、预测表达、稳定性或降低免疫风险：

$$
\max_{c'_1,\ldots,c'_M} F(c'_1,\ldots,c'_M),
\quad \mathrm{s.t.}\ aa(c'_j)=aa(c_j)\ \forall j.
$$

### 5.2 输入输出

| 项 | 定义 |
|---|---|
| 输入 | protein sequence，或 `(protein, wildtype CDS)` |
| 输出 | optimized CDS，protein identity 必须为 1.0 |
| 数据 | RefSeq/GENCODE protein-CDS pair、HPA high-expression reference、CodonBERT/mRNA-LM benchmark |
| 训练 | CDS codon-level refinement；默认 strict 同义替换 |

### 5.3 指标

硬指标：

$$
\mathrm{protein\ identity}=\mathbf{1}[\mathrm{Translate}(CDS')=\mathrm{protein}].
$$

优化指标：

- CAI/tAI 提升。
- GC、GC3 落在目标区间。
- codon usage KL 到 human high-expression reference。
- codon pair bias、MinMax profile 与 ribosome ramp。
- predicted stability/half-life/TE。
- 起始 30-50 nt MFE 不过低，避免过强结构抑制翻译起始。

综合 Pareto 指标：

$$
\mathrm{HV}(\mathcal{P})=\mathrm{HyperVolume}\{(\Delta CAI,\Delta TE,-|\Delta GC|,-d_{codon})\}.
$$

### 5.4 SOTA 对比

| 基线 | 对比点 |
|---|---|
| CodonBERT / mRNA-LM CDS 分支 | codon 表征与优化 |
| Prot2RNA | protein-conditioned diffusion CDS generation |
| LinearDesign | MFE 与 codon objective 动态规划 |
| DERNA | Pareto CAI/MFE RNA design |
| JCAT/IDT/GenScript 类启发式工具 | 工业常规 codon optimization |
| AR codon LM | 生成式基线 |

注意：Prot2RNA 的公开评审曾指出 CAI/相似度不能证明真实表达提升。本项目必须把表达提升写成 “predicted/oracle-based”，除非后续有湿实验。

### 5.5 Edit Flow 独特优势

MEF 的 CDS 算子支撑集只包含同义密码子，蛋白不变性由构造保证：

$$
q_\theta(\mathrm{sub}(c\to c'))=0\quad \text{if }aa(c')\ne aa(c).
$$

这比“生成后翻译检查并过滤”更强，能显著提高采样有效率。另一个优势是最小编辑密码子优化：对已有临床 construct 只改必要 codon，而不是重新生成整段 CDS。

## 6. T5 最小编辑治疗性改造

### 6.1 目标

给定野生型/患者 mRNA 与功能目标，在最小编辑预算内达成目标：

$$
\min_{x'} d_{edit}(x,x'),\quad
\mathrm{s.t.}\ f_{eval}(x')\ge \tau,\ x'\in\mathcal{L}_{mRNA}.
$$

功能目标可包括：提升 TE 到目标桶、移除降解元件、降低 uAUG、改善起始区可及性、在 CDS 同义约束下优化稳定性。

### 6.2 输入输出

| 项 | 定义 |
|---|---|
| 输入 | wildtype/patient mRNA，目标函数，编辑预算 `k`，允许编辑区域 |
| 输出 | edited mRNA 与最小 edit script |
| 数据 | RefSeq/GENCODE 构造 pair，Sample variant set，ClinVar/UTR variant 可作为 future 扩展 |
| 训练 | corruption-refinement；按目标构造正负编辑对 |

### 6.3 指标

达标率：

$$
\mathrm{Success@}k=\frac{1}{N}\sum_i \mathbf{1}[d(x_i,x_i')\le k\land f(x_i')\ge \tau].
$$

Pareto 前沿：

$$
\mathcal{P}=\{(d_{edit},\Delta f,\mathrm{realism})\}.
$$

其他指标：

- 最小编辑数分布。
- `uplift/edit`。
- 若涉及 CDS，protein identity=1.0。
- 与天然分布距离，防止用极端序列骗 oracle。

### 6.4 SOTA 对比

| 基线 | 对比点 |
|---|---|
| Genetic algorithm with edit penalty | 传统优化 |
| Beam search over edit actions | 小预算可解释基线 |
| Masked diffusion + edit-distance regularization | 定长/掩码方法能否逼近 |
| UTR-specific optimizers | 只在 UTR 子任务比较 |

### 6.5 Edit Flow 独特优势

T5 是 Edit Flow 最自然的任务，因为目标变量就是编辑距离和编辑脚本。MEF 的采样轨迹已经是编辑过程，可直接约束预算、区域和操作类型。定长模型通常需要把 edit distance 作为外部 penalty，生成过程本身并不以“最小改造”为基本单位。

## 7. T6 长度可控设计

### 7.1 目标

把长度作为显式约束或优化目标，例如：

- 最短 5UTR 达到目标 TE。
- 在稳定性约束下缩短 3UTR。
- 生成指定长度范围的 full-length mRNA。

形式化：

$$
\max_{x'} f(x')-\lambda |L(x')-L^\star|,
\quad x'\in\mathcal{L}_{mRNA}.
$$

或：

$$
\min L(x'),\quad \mathrm{s.t.}\ f(x')\ge\tau.
$$

### 7.2 输入输出

| 项 | 定义 |
|---|---|
| 输入 | source mRNA 或 empty state，目标长度 `L*`/长度范围，功能阈值 |
| 输出 | 长度命中的 mRNA 或 UTR |
| 数据 | RefSeq/GENCODE 长度分布，Sample/UTR 数据，构造长度 bucket pair |
| 训练 | empty-growth + length-conditioned refinement |

### 7.3 指标

长度命中：

$$
\mathrm{LenHit}_\delta=\mathbf{1}[|L(x')-L^\star|\le \delta].
$$

控制单调性：

$$
\rho=\mathrm{Spearman}(L^\star,L(x')).
$$

功能保持/达标：

$$
\mathrm{FuncHit}=\mathbf{1}[f(x')\ge\tau].
$$

联合指标：

$$
\mathrm{LCS}=\mathrm{LenHit}_\delta\cdot \mathrm{FuncHit}\cdot \mathrm{valid}.
$$

### 7.4 SOTA 对比

| 基线 | 对比点 |
|---|---|
| AR LM with length token | 生成长度是否可控 |
| Masked diffusion fixed-padding | 是否能真实改变有效长度 |
| GA/heuristic truncation | 最短 UTR 达标 |
| mRNAutilus/ProMORNA | full-length 长度 plausibility |

### 7.5 Edit Flow 独特优势

长度控制的本质是插入与删除的平衡。MEF 直接拥有 `lambda_ins` 与 `lambda_del`，可以在采样时根据 `L-L*` 调整速率，而不是用 padding mask 假装变长。特别是“功能约束下最短 UTR”这种任务，天然需要删除冗余片段并保留关键 motif。

## 8. T7 调控元件移植/切除

### 8.1 目标

向宿主 mRNA 的 UTR 插入功能元件，或切除抑制元件：

- 移植：Kozak-like context、IRES、uORF、stability element、poly(A) signal。
- 切除：uAUG/uORF、ARE、miRNA binding site、G-quadruplex、cryptic splice/polyA signal。

形式化：

$$
\mathrm{Transplant}(x,e,p): x'\ \mathrm{contains}\ e\ \mathrm{near}\ p,
$$

$$
\mathrm{Excision}(x,e): x'\ \mathrm{does\ not\ contain}\ e.
$$

### 8.2 输入输出

| 项 | 定义 |
|---|---|
| 输入 | host mRNA，目标元件 `e`，插入位置范围或切除目标，允许编辑区域 |
| 输出 | 元件植入/切除后的 mRNA 与 edit script |
| 数据 | Rfam、IRESite/IRESbase、miRBase/TargetScan、ARE/polyA 文献库、GENCODE host UTR |
| 训练 | motif-aware corruption：插入/删除真实元件片段，保持区域边界 |

### 8.3 指标

元件成功率：

$$
\mathrm{MotifHit}=\mathbf{1}[\mathrm{Scan}(x',e)=1].
$$

切除成功率：

$$
\mathrm{MotifRemove}=\mathbf{1}[\mathrm{Scan}(x',e)=0].
$$

位置与方向：

$$
\mathrm{PosErr}=|p_{observed}-p_{target}|.
$$

功能与安全：

- independent oracle 的 TE/half-life 变化。
- CDS reading frame 完整。
- off-target motif 新增数量。
- 编辑距离与局部结构变化。

### 8.4 SOTA 对比

| 基线 | 对比点 |
|---|---|
| Motif insertion heuristic + local search | 简单可解释基线 |
| GA with motif constraint | 优化型基线 |
| UTRGAN/UTailoR | 是否能生成含目标 motif 的 UTR |
| Masked diffusion with inpainting | 固定窗口插入/替换能力 |

### 8.5 Edit Flow 独特优势

T7 是变长编辑的标志性任务。移植元件需要插入一段序列，切除元件需要删除一段序列；这不是 token 替换可以自然表达的。MEF 的 CTMC 速率场可以把插入/删除作为连续时间事件，并用区域条件防止操作破坏 CDS。最终输出不仅有序列，还有“哪段元件被插入/切除”的编辑脚本。

## 9. 任务到模型能力的映射

| 能力 | T1 | T2 | T3 | T4 | T5 | T6 | T7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Empty growth | 高 | 低 | 高 | 低 | 低 | 中 | 低 |
| Refinement | 中 | 高 | 中 | 高 | 高 | 高 | 高 |
| UTR nt 插删改 | 高 | 高 | 高 | 低 | 高 | 高 | 高 |
| CDS 同义替换 | 高 | 低 | 低 | 高 | 中 | 中 | 低 |
| CDS 整密码子 indel | 可选 | 否 | 否 | 可选 | 可选 | 可选 | 否 |
| 编辑预算 | 中 | 高 | 中 | 中 | 最高 | 中 | 高 |
| 长度控制 | 中 | 中 | 高 | 低 | 中 | 最高 | 高 |
| 元件级操作 | 中 | 中 | 高 | 低 | 中 | 中 | 最高 |

## 10. 统一实验表设计

每个任务最终论文表建议至少包含：

```text
method, trainable_params, backbone, data_split, oracle_eval,
validity, main_score, realism_score, edit_distance,
10seed_mean, ci95_low, ci95_high, p_value
```

任务主指标：

| 任务 | 主指标 | 必须同时报告 |
|---|---|---|
| T1 | validity + distribution distance + novelty | 最近训练序列距离 |
| T2 | independent oracle ΔTE, gain/edit | random-test 与 human-heldout 分开 |
| T3 | target bucket accuracy / regression error | 真实性与长度分布 |
| T4 | protein identity=1.0, CAI/TE Pareto | GC3/MFE/起始结构 |
| T5 | Success@k, Pareto uplift vs edit | protein identity 与 realism |
| T6 | LenHit + FuncHit | 控制单调性 |
| T7 | MotifHit/MotifRemove | off-target motif 与功能变化 |

## 11. 创新性健康检查

通过标准：

1. T4 中 protein identity 必须由算子掩码构造保证，采样有效率接近 100%。
2. T5/T6/T7 至少一个任务显示定长 masked diffusion 或 AR baseline 难以达到同等编辑预算/长度控制/元件成功率。
3. T2/T3 不只报告 oracle 分数，还报告 human held-out 与分布真实性，避免 oracle hacking。
4. mRNABench 或相关 prediction 任务上报告可训练参数量，支撑 frozen backbone + lightweight head 的参数效率。
5. 所有核心结论有 10 seeds 与 bootstrap CI。

失败信号：

- TE 提升只在训练 oracle 上出现，独立 oracle 不提升。
- 生成序列距离 train 最近邻极近，疑似记忆。
- T4 需要大量后验过滤才能保证 protein identity。
- T5/T6/T7 的优势只来自更大模型或更多采样，而不是编辑文法。
- 全长生成合法率高但 CDS/UTR 分布严重偏离天然分布。

## 12. 可落地 MVP 路线

MVP 不需要一次跑完七个任务。建议顺序：

1. T4：CDS 同义密码子优化。最容易验证 protein identity 与 codon mask。
2. T2：Sample 2019 5UTR TE refinement。最容易接独立 oracle。
3. T6：长度可控 UTR 设计。直接证明变长编辑优势。
4. T7：motif 插入/切除。展示元件级 edit script。
5. T1/T3/T5：在前四项稳定后扩展到全长生成、de novo UTR 和治疗性最小改造。

单 GPU 约束：

- 每个任务先 1k 输入、每输入 16-64 samples。
- 采样步数 50-100 起步，主实验再扩到 200。
- 评测 oracle 与 ViennaRNA/LinearFold 必须本地化，避免评测依赖外网。
- 所有候选先过合法性过滤，但主表报告过滤前有效率。

## 13. 论文叙事建议

推荐标题级主张：

> mRNA-EditFlow frames therapeutic mRNA design as region-conditioned variable-length editing: a codon-locked CDS lattice coupled to a length-free UTR regulatory canvas.

中文对应：

> mRNA-EditFlow 将治疗性 mRNA 设计重构为区域条件下的变长编辑：CDS 是帧锁定密码子格，UTR 是长度自由调控画布。

该主张可被 T1-T7 分层验证：

- T1 证明能生成合法全长 mRNA。
- T2/T3 证明 UTR 画布能调控 TE。
- T4 证明 CDS 密码子格能构造性保持蛋白不变。
- T5/T6/T7 证明变长编辑解锁了定长模型不自然的新任务。

如果后续实验只在 T1 做得一般，但 T2/T4/T5/T6/T7 强，论文仍然成立。MEF 的核心不是做一个更大的 mRNA LM，而是给 mRNA 设计一个正确的编辑文法与可评测任务族。
