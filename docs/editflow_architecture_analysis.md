# mRNA-EditFlow 架构分析与迁移方案

日期：2026-07-11

本文档对应 Task 0.1，只分析原始 Edit Flow 源码与 mRNA 迁移边界，不涉及代码实现。分析对象为 `editflow/flow.py`、`editflow/core/flow_utils.py`、`editflow/utils.py`、`editflow/models/editformer.py`、`editflow/train_mix.py`、`editflow/sample.py`。

核心科学故事是：**mRNA 设计 = 区域条件下的变长编辑；CDS 是帧锁定密码子格，UTR 是长度自由调控画布**。原 Edit Flow 的价值不在于某个 Transformer，而在于把生成问题写成离散序列上的连续时间编辑过程：插入、删除、替换都由速率场驱动，训练监督来自源序列到目标序列的最优编辑对齐。这正好对应 mRNA 的真实设计动作：UTR 需要可插入/切除调控元件，CDS 则必须在密码子格上做帧安全、蛋白不变的同义编辑。

## 1. 原始代码结构

| 文件 | 关键对象 | 研究含义 | mRNA 迁移结论 |
|---|---|---|---|
| `flow.py` | `x2prob`、`sample_p`、`EmptyCoupling`、`GeneratorCoupling`、`CubicScheduler` | 定义耦合、one-hot bridge 与 `kappa(t)` 调度 | 保留 CTMC 与调度器；替换 token 语义与 coupling 来源 |
| `utils.py` | 氨基酸词表、`BOS/PAD/GAP`、`_align_pair`、`opt_align_xs_to_zs`、`rm_gap_tokens`、`apply_ins_del_operations` | 把可变长编辑转成含 GAP 的对齐空间 `z` | DP 对齐必须升级为区域内、CDS 密码子级对齐 |
| `core/flow_utils.py` | `make_hybrid_batch`、`sample_cond_pt`、`make_ut_mask_from_z`、`fill_gap_tokens_with_repeats` | 构造 empty-growth/refinement 混合耦合，生成 ground-truth 编辑场 | 可复用思路；`u_mask` 需要区域条件合法算子掩码 |
| `models/editformer.py` | `EditFlowsTransformer`、时间嵌入、绝对位置嵌入、多头输出 | 预测每位点编辑速率与插入/替换 token 分布 | 保留“单塔 encoder-only + 多头速率输出”，替换为冻结 mRNA 基座 + RoPE + region FiLM |
| `train_mix.py` | 训练循环、bridge 采样、loss | 原始数学目标的实际落点 | 迁移时保留 rate matching 目标，增加区域合法性、辅助结构损失与数值 guard |
| `sample.py` | Euler tau-leaping 式采样 | 以 `lambda * dt` 概率逐位执行删、替、插 | 迁移时必须加入 CDS 帧锁定、长度约束、编辑预算与 guidance |

原项目当前是 AMP 氨基酸序列生成，词表为 20 个氨基酸，加 `BOS=20`、`PAD=21`、`GAP=22`。模型训练时使用 `vocab_model_size=V+2` 作为可预测 token 空间，alignment 空间使用 `V+3` 以包含 `GAP`。mRNA 迁移不能直接继承该词表，需要显式区分 nt token、codon token、特殊 token、区域标签与 CDS 相位。

## 2. CTMC 速率场

设当前序列为

$$
x_t=(x_{t,1},\ldots,x_{t,L_t}),\quad x_{t,i}\in\mathcal{V}.
$$

原模型在每个位置 `i` 输出三类非负速率：

$$
(\lambda_i^{\mathrm{ins}},\lambda_i^{\mathrm{sub}},\lambda_i^{\mathrm{del}})
=\mathrm{softplus}(h_\theta(x_t,t)_i)\in\mathbb{R}_{\ge0}^3,
$$

并输出插入与替换的 token 分布：

$$
p_i^{\mathrm{ins}}(a)=\mathrm{softmax}(g_\theta^{\mathrm{ins}}(h_i))_a,\quad
p_i^{\mathrm{sub}}(a)=\mathrm{softmax}(g_\theta^{\mathrm{sub}}(h_i))_a.
$$

因此连续时间马尔可夫链的原子跳变速率是：

$$
q_\theta(x\rightarrow \mathrm{Ins}(x,i,a)\mid t)=\lambda_i^{\mathrm{ins}}p_i^{\mathrm{ins}}(a),
$$

$$
q_\theta(x\rightarrow \mathrm{Sub}(x,i,a)\mid t)=\lambda_i^{\mathrm{sub}}p_i^{\mathrm{sub}}(a),
$$

$$
q_\theta(x\rightarrow \mathrm{Del}(x,i)\mid t)=\lambda_i^{\mathrm{del}}.
$$

总离开速率为

$$
\Lambda_\theta(x,t)=
\sum_{i=1}^{L_t}\left[
\lambda_i^{\mathrm{del}}
+\lambda_i^{\mathrm{sub}}\sum_{a\in\mathcal{V}}p_i^{\mathrm{sub}}(a)
+\lambda_i^{\mathrm{ins}}\sum_{a\in\mathcal{V}}p_i^{\mathrm{ins}}(a)
\right]
=\sum_i(\lambda_i^{\mathrm{del}}+\lambda_i^{\mathrm{sub}}+\lambda_i^{\mathrm{ins}}).
$$

这解释了 `train_mix.py` 中的速率总量项：`lambda_ins.sum(dim=1)`、`lambda_sub.sum(dim=1)`、`lambda_del.sum(dim=1)` 是 CTMC 对所有可能跳变的积分强度惩罚。若只最大化目标操作概率，模型会把所有速率推高；总量项保证“需要编辑的位置速率高，不需要编辑的位置速率低”。

## 3. 最优对齐 DP 与 `z` 空间

原始代码不直接在不同长度的 `x_0`、`x_1` 上监督，而是先通过 Levenshtein DP 得到含 GAP 的等长对齐：

$$
(x_0,x_1)\mapsto (z_0,z_1),\quad z_0,z_1\in(\mathcal{V}\cup\{\mathtt{GAP}\})^{L_z}.
$$

`utils._align_pair` 的递推为：

$$
D_{i,0}=i,\quad D_{0,j}=j,
$$

$$
D_{i,j}=
\begin{cases}
D_{i-1,j-1}, & x_{0,i}=x_{1,j},\\
1+\min(D_{i-1,j},D_{i,j-1},D_{i-1,j-1}), & x_{0,i}\ne x_{1,j}.
\end{cases}
$$

回溯时：

- match/substitution：`z0[k]=x0[i]`，`z1[k]=x1[j]`
- deletion：`z0[k]=x0[i]`，`z1[k]=GAP`
- insertion：`z0[k]=GAP`，`z1[k]=x1[j]`

复杂度：

$$
T_{\mathrm{DP}}=O(|x_0||x_1|),\quad M_{\mathrm{DP}}=O(|x_0||x_1|),\quad L_z\le |x_0|+|x_1|.
$$

对于 AMP 短序列该复杂度可接受。对于全长 mRNA，若 CDS 可到 1536 nt 甚至 3072 nt，朴素 DP 会成为 CPU 侧瓶颈。mRNA 迁移必须做三点约束：

1. 按区域分块对齐：5UTR、CDS、3UTR 不允许跨区域对齐。
2. CDS 以密码子为单位对齐：长度为 `L_cds/3`，复杂度降为 `O((L_cds/3)^2)`，同时保证帧边界。
3. 对 ortholog coupling 使用 banded DP 或已知同源锚点，复杂度从 `O(mn)` 降为 `O(w\max(m,n))`，其中 `w` 是进化 indel 允许带宽。

## 4. `kappa` 调度与 bridge 采样

`flow.CubicScheduler` 定义：

$$
\kappa(t)=
-2t^3+3t^2+a(t^3-2t^2+t)+b(t^3-t^2),
$$

$$
\kappa'(t)=
-6t^2+6t+a(3t^2-4t+1)+b(3t^2-2t).
$$

边界满足：

$$
\kappa(0)=0,\quad \kappa(1)=1.
$$

`train_mix.py` 实际使用 `CubicScheduler(a=1.0,b=1.0)`。训练中先在 `z` 空间构造条件概率路径：

$$
p_t(z_k\mid z_{0,k},z_{1,k})=(1-\kappa(t))\mathbf{1}[z_k=z_{0,k}]
+\kappa(t)\mathbf{1}[z_k=z_{1,k}],
$$

再调用 `sample_cond_pt` 对每个位置独立采样得到 `z_t`。随后 `rm_gap_tokens(z_t)` 去掉 GAP，得到模型实际输入 `x_t`。

训练损失使用调度系数：

$$
c(t)=\frac{\kappa'(t)}{1-\kappa(t)}.
$$

直觉上，`t` 越接近目标端，尚未完成的编辑越少但越紧迫，监督项权重越高。数值上 `1-\kappa(t)` 在 `t\rightarrow 1` 时趋近 0，mRNA 版本需要对 `t` 做截断，例如 `t\in[\epsilon,1-\epsilon]`，或对 `c(t)` 做上界裁剪，以避免长序列上梯度尖峰。

## 5. Ground-truth 编辑场

`make_ut_mask_from_z(z_t,z_1,vocab_size)` 将对齐空间里的当前位置关系转成操作 one-hot mask。操作集合大小为：

$$
|\mathcal{A}|=2|\mathcal{V}_{model}|+1,
$$

其中前 `V` 个是插入 token，中间 `V` 个是替换 token，最后一个是删除：

$$
u^\star_{k,a}=
\begin{cases}
1, & z_{t,k}=\mathtt{GAP},\ z_{1,k}=a,\quad a\in\mathcal{V}\quad(\mathrm{insert}),\\
1, & z_{t,k}\ne \mathtt{GAP},\ z_{1,k}=a,\ z_{t,k}\ne z_{1,k}\quad(\mathrm{substitute}),\\
1, & z_{t,k}\ne \mathtt{GAP},\ z_{1,k}=\mathtt{GAP}\quad(\mathrm{delete}),\\
0, & \text{otherwise}.
\end{cases}
$$

模型预测在 `x` 空间，而监督在 `z` 空间。`fill_gap_tokens_with_repeats` 用 `z_gap_mask` 的前缀和把 `x` 空间预测重复/映射回 `z` 空间。这个设计是原 Edit Flow 的关键技巧：模型不必显式处理 GAP token，但损失可以对插入位置给出监督。

## 6. Loss 推导

令

$$
\hat{u}_{k,a}=q_\theta(\text{执行操作 }a\text{ at aligned position }k\mid x_t,t)
$$

是映射到 `z` 空间后的预测速率。原实现中：

$$
\hat{u}_{k,a}=
\begin{cases}
\lambda_i^{\mathrm{ins}}p_i^{\mathrm{ins}}(a), & a\in\mathcal{V}_{ins},\\
\lambda_i^{\mathrm{sub}}p_i^{\mathrm{sub}}(a), & a\in\mathcal{V}_{sub},\\
\lambda_i^{\mathrm{del}}, & a=\mathrm{del}.
\end{cases}
$$

`train_mix.py` 中的每类损失可以统一写为：

$$
\mathcal{L}(\theta)=
\mathbb{E}_{(x_0,x_1),t,z_t}
\left[
\sum_{i=1}^{L_x}\sum_{a\in\mathcal{A}_i} q_\theta(a\mid x_t,i,t)
-c(t)\sum_{k=1}^{L_z}\sum_{a\in\mathcal{A}}u^\star_{k,a}\log \hat{u}_{k,a}
\right].
$$

第一项是 CTMC 的总速率积分项，第二项是对齐监督下的目标跳变 log-rate。代码中 `log_uz_cat = torch.clamp(uz_cat.log(), min=-20)` 是防止零速率产生 `-inf`。`valid_z_mask` 与 `valid_x_mask` 分别屏蔽 `z` 与 `x` 空间 PAD，避免 PAD 参与 CE 或总速率项。

从概率建模角度看，这相当于对条件桥的目标 hazard 做 Poisson 过程负对数似然近似：目标操作发生的位置需要提高相应速率，非目标操作由总强度项压低。该目标比普通 token CE 更适合变长生成，因为插入和删除不是后处理，而是概率过程的一等公民。

## 7. Hybrid Coupling

原始 `make_hybrid_batch` 混合两类 coupling：

- Empty coupling：以概率 `empty_prob` 令 `x_0=[]`，目标是从空序列增长到 `x_1`。
- Corruption coupling：对 `x_1` 随机删除、替换、插入，生成 `x_0`，目标是学习局部修复。

若 `empty_prob=0.9`，训练更偏全局生长；若降低该值，模型更偏 refinement。该设计对 mRNA 很重要，因为 T1 全长生成需要 empty-growth，而 T2/T4/T5/T7 都更像从已有序列出发的编辑优化。

mRNA 版本应扩展为三路 coupling：

$$
\pi=\pi_{\mathrm{empty}}+\pi_{\mathrm{corrupt}}+\pi_{\mathrm{ortholog}},\quad
\pi_{\mathrm{empty}}+\pi_{\mathrm{corrupt}}+\pi_{\mathrm{ortholog}}=1.
$$

- `empty-growth`：从空 5UTR、空 CDS scaffold、空 3UTR 或最小 BOS 状态增长到全长 mRNA。
- `corruption-refinement`：UTR 执行 nt 级随机插删改，CDS 只执行同义密码子替换或整密码子 indel。
- `ortholog-coupling`：`x_0` 为其他物种直系同源 mRNA，`x_1` 为目标物种 mRNA；对齐限制在同区域内，CDS 以密码子为单位。

`ortholog-coupling` 是科学上最强的迁移点：真实进化轨迹中 indel 不随机分布，UTR 调控元件的插入/切除和 CDS 同义偏好都受选择压力约束。把该轨迹作为 Edit Flow coupling，可以让变长编辑获得进化先验。

## 8. 原采样过程

`sample.py` 采用离散时间 Euler tau-leaping 近似。设步数为 `S`，`dt=1/S`。每一步：

1. 将当前变长序列 padding 成 batch。
2. 输入模型得到 `lambda_ins`、`lambda_sub`、`lambda_del`、`ins_probs`、`sub_probs`。
3. 对每个样本、每个位置按如下概率执行操作：

$$
P(\mathrm{del}_{i})\approx \lambda_i^{\mathrm{del}}dt,
$$

$$
P(\mathrm{sub}_{i})\approx \lambda_i^{\mathrm{sub}}dt,
$$

$$
P(\mathrm{ins}_{i})\approx \lambda_i^{\mathrm{ins}}dt\cdot s_{\mathrm{ins}},
$$

其中 `s_ins` 对应代码里的 `ins_scale`。删除优先于替换，替换后可再插入。该过程不是精确 Gillespie 采样，而是并行近似；当 `lambda * dt` 不够小时会产生多事件碰撞偏差。

mRNA 采样必须加入以下硬约束：

- CDS strict 模式：禁止 nt 级 ins/del；sub 必须在当前氨基酸的同义密码子集合内。
- CDS `codon_indel` 模式：只允许整密码子插入/删除，长度变化属于 `3Z`，且不引入提前终止。
- UTR 模式：允许 nt 级 ins/del/sub，但需约束 `A/C/G/U`、长度范围、uAUG/Kozak/IRES/miRNA 等 motif 逻辑。
- 预算模式：T5 需要约束累计编辑距离 `d(x_0,x_t)\le k`。
- 长度模式：T6 需要把目标长度或长度惩罚注入采样速率，例如对插入/删除速率做

$$
\lambda_i^{\mathrm{ins}}\leftarrow \lambda_i^{\mathrm{ins}}\exp(-\eta(L_t-L^\star)),\quad
\lambda_i^{\mathrm{del}}\leftarrow \lambda_i^{\mathrm{del}}\exp(\eta(L_t-L^\star)).
$$

## 9. 原实现的工程风险

这些问题不影响 Task 0 文档结论，但迁移时需要在方案中显式规避。

1. `sample_p(pt / temperature)` 对概率整体除以常数，`torch.multinomial` 只看相对权重，因此 temperature 实际无效。mRNA 版本若要温控，应在 logits 上做 `softmax(logits / T)`。
2. `sample.py` 在删除判断发生于 BOS 判断之前，因此 BOS 可能被删除。mRNA 版本应保护边界 token 与区域分隔 token。
3. 原采样中 `lambda * dt` 未显式 clamp 到 `[0,1]`。长序列或 guidance 后速率过大时，需要 `p=1-exp(-lambda*dt)` 或概率裁剪。
4. 原 DP 是 Python list 二维表，长 mRNA 会慢且占内存。应实现按区域缓存、banded DP 或 C++/NumPy 向量化备选。
5. `EditFlowsTransformer` 使用绝对位置 embedding，变长频繁插删下外推不稳。mRNA 版本应使用 RoPE 或相对位置偏置。
6. 原 model token embedding 从零训练，无法利用 mRNA 原生基座。迁移方案应冻结 mRNABERT/Helix-mRNA/Orthrus/LAMAR 等基座，只训练轻量编辑头与 adapter。

## 10. mRNA-EditFlow 的状态、算子与合法性

mRNA 序列定义为区域拼接：

$$
x=(x^{5U},x^{CDS},x^{3U}),\quad
x^{5U},x^{3U}\in\{A,C,G,U\}^{*},
$$

$$
x^{CDS}=(c_1,\ldots,c_M),\quad c_j\in\{A,C,G,U\}^3.
$$

区域条件合法算子集合为：

$$
\mathcal{A}(r)=
\begin{cases}
\{\mathrm{ins}_{nt}(a),\mathrm{del}_{nt},\mathrm{sub}_{nt}(a):a\in\{A,C,G,U\}\}, & r\in\{5UTR,3UTR\},\\
\{\mathrm{sub}_{codon}(c'):c'\in \mathrm{Syn}(\mathrm{aa}(c_j))\}, & r=CDS,\ \text{strict},\\
\mathcal{A}_{strict}\cup\{\mathrm{ins}_{codon}(c),\mathrm{del}_{codon}\}, & r=CDS,\ \text{codon_indel}.
\end{cases}
$$

蛋白不变性定理：

若 CDS strict 模式下所有编辑均满足 `c_j -> c'_j` 且 `aa(c'_j)=aa(c_j)`，并禁止 CDS indel，则

$$
\mathrm{Translate}(x^{CDS}_{out})=\mathrm{Translate}(x^{CDS}_{in}).
$$

该性质是构造性的，不依赖后验筛选。相比定长 masked diffusion 或普通 BERT 重打分，Edit Flow 的优势是把“什么能编辑”写进速率场的支撑集，而不是生成后再过滤非法结果。

## 11. mRNA loss 迁移形式

迁移后速率场应写成区域条件版本：

$$
q_\theta(a\mid x_t,i,t,r_i,\phi_i,c)=
\mathbf{1}[a\in\mathcal{A}(r_i,\phi_i,c)]\,
\tilde{q}_\theta(a\mid h_i,t,r_i,\phi_i,c),
$$

其中 `r_i` 是区域，`\phi_i` 是 CDS 相位或 codon index，`c` 是任务条件。非法操作速率严格为 0，或通过 logits `-inf` mask 实现。主损失保持：

$$
\mathcal{L}_{flow}=
\sum_i\sum_{a\in\mathcal{A}(r_i)}q_\theta(a\mid x_t,i,t)
-c(t)\sum_k\sum_a u^\star_{k,a}\log \hat{q}_\theta(a\mid x_t,k,t).
$$

结构辅助头加入：

$$
\mathcal{L}=
\mathcal{L}_{flow}
+\alpha_{\mathrm{mfe}}\|\hat{y}_{MFE}-y_{MFE}\|_2^2
+\alpha_{\mathrm{acc}}\|\hat{y}_{acc}-y_{acc}\|_2^2
+\alpha_{\mathrm{phase}}\mathcal{L}_{phase},
$$

其中 `y_acc` 可为起始密码子邻域 unpaired probability，`y_MFE` 来自 RNAfold/LinearFold 离线预计算。采样时可用独立 TE oracle 或内部辅助头对速率进行 guidance：

$$
\log q'(a)=\log q(a)+\gamma\nabla_{\Delta_a}\hat{f}_{TE}(x),
$$

但论文评测必须使用训练未参与的独立 oracle，避免“优化哪个 oracle 就用哪个 oracle 评测”的循环。

## 12. 复杂度总览

设 batch 大小为 `B`，输入去 GAP 后最大长度为 `L`，对齐长度为 `Z`，hidden 为 `H`，层数为 `K`。

| 模块 | 时间复杂度 | 内存复杂度 | 单 GPU 约束 |
|---|---:|---:|---|
| 朴素 DP 对齐 | `O(sum_b |x0_b||x1_b|)` | `O(max_b |x0_b||x1_b|)` | 必须按区域/密码子分块，长序列启用 banded DP |
| bridge 采样 | `O(BZ|V|)` one-hot 构造，可优化为索引采样 | `O(BZ|V|)` | mRNA 词表小，但 codon mask 多，避免全 64 codon dense 复制 |
| Transformer 生成头 | `O(K B L^2 H + K B L H^2)` | `O(BL H + KBL^2)` | frozen backbone + 轻量头；长度分桶与梯度累积 |
| 速率与 CE loss | `O(BZ|\mathcal{A}|)` | `O(BZ|\mathcal{A}|)` | 区域合法算子稀疏化，CDS 同义集合平均小于 4 |
| Euler 采样 | `O(S B L (|\mathcal{V}|+1))` | `O(BL)` | 采样步数 50-200；长序列用分段/并行向量化 |

单 GPU MVP 建议：

- 预训练长度分桶：5UTR≤128 nt，CDS≤1536 nt，3UTR≤256 nt。
- 生成头默认 hidden 384，4-8 层，约 5-20M 可训练参数。
- 冻结基座 embedding 离线缓存，训练时只读 NPZ 分片，避免每步大模型前向。
- 每个 batch 限制 `B * L^2` 上界，OOM 时自动降 batch 或缩短 bucket。

## 13. 创新性健康检查

需要避免的弱主张：只说“把 Edit Flow 用到 mRNA”不够。顶刊/顶会级故事必须落在以下可检验差异上：

1. **编辑文法作为归纳偏置**：CDS 与 UTR 的合法操作集合不同，且被写进 CTMC 支撑集。
2. **构造性蛋白不变**：CDS strict 模式由同义密码子掩码保证 identity=100%，不是后验筛选。
3. **变长任务族**：T5 最小编辑、T6 长度可控、T7 元件移植/切除是定长扩散与 BERT 重打分结构上不擅长的任务。
4. **进化感知 coupling**：ortholog alignment 与 Edit Flow 的 DP 监督同构，能把自然进化 indel 作为学习信号。
5. **参数效率**：冻结 mRNA 原生基座，只训练小编辑头与 adapter，和 mRNABench 线性探针/参数量口径对齐。

需要保持克制的主张：v1 不应宣称在全长 mRNA 原始生成所有主指标上碾压 mRNAutilus、GEMORNA 或大规模自回归模型。更稳健的 SOTA 叙事是三腿：新能力轴、TE/UTR 专项、参数效率。

## 14. Task 0 到后续实现的边界

本文档建议后续实现保持以下接口边界：

- `core/mrna_flow_utils.py`：区域感知 DP、三路 coupling、合法算子 mask、同义密码子表。
- `models/mrna_editformer.py`：冻结 backbone embedding 输入、RoPE、region/phase embedding、FiLM、速率头、aux head。
- `sample.py`：统一变长编辑采样器，支持 T1-T7 条件与约束。
- `eval/metrics.py`：把合法性、蛋白 identity、编辑距离、长度命中、motif 成功率作为一等指标。

验证门槛：

1. 随机 CDS 同义编辑后 `protein_identity=1.0`。
2. `codon_indel` 模式下 CDS 长度变化属于 `3Z`，且无移码。
3. UTR 插删改不跨区域边界。
4. DP 对齐输出与编辑距离一致。
5. 采样后所有序列能通过 ATG、终止、in-frame、无提前终止、合法字符检查。
