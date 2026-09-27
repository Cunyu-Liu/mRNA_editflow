# Related Work 笔记：Shen et al. 2025（膜蛋白序列-表达预测）

- **来源**：bioRxiv 2025.09.25.678317（v2, 2025-09-28, CC-BY 4.0）；Europe PMC PPR1092621；团队 Edinburgh (Oyarzún) + Bristol (Curnow/Mulholland)。
- **一句话**：计算设计膜蛋白变体库（12,248 条，单蛋白 113aa）→ 实测 ~2,000 表达量（E. coli GFP 融合）→ 轻量 ML 训练 → 推断 1 万+ 未测（top 预测验证 100%）→ SHAP 定位关键位点 → 模型指导改造获 8 倍纯化产量提升。
- **与 DeltaBench 的关系判定：不撞车（域/对象/贡献形态均异）；四点神似（sequence-to-phenotype、变体库+ML、小数据范式、不泛化命题）**。

## 引用位置建议（论文四处落笔）

1. **Related Work**：作为 sequence-to-phenotype 预测的蛋白侧近邻工作（域差异声明：蛋白表达 E. coli vs mRNA UTR 人源 MPRA）。
2. **Discussion / 发现二**：其引言"models have not been found to generalise beyond very specific training conditions"（引 [5]）与我们双因子结论同声——跨域佐证，防"不泛化已知"审稿攻击；我方增量 = 机理分解（误差相关体制 + 双因子交互 + 任务内剂量响应）。
3. **几何论据**：其"受控组合多样性使轻量模型可用"支持密度/几何决定论（注意：其设计库全量实测表型，与 D16-C 合成增强证伪前提不同，勿混同）。
4. **叙事方法借鉴（不涉及湿实验）**：其"预测→解释→指导改造→验证"闭环 = 我方 §8.4 的计算版对应（polyA 91% → 一阶定位 → COMB +0.35~0.48）；小数据效率卖点与 ERK 1,177 参数贴上界同款。

## 边界（不可学）

湿实验闭环（8 倍验证）超出本项目"无新增湿实验"合同边界；单体系深挖路线与本项目系统归因定位相反。

## 其余相关引用链（从其引言顺藤）

- 其 [3]：sequence-to-expression 模型综述（ML for protein expression）
- 其 [5]：泛化失败的核心引证（写入我方"不泛化是领域共识形成中"的证据链）
