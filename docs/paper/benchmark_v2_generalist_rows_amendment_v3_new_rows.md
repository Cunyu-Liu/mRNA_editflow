# Benchmark v2 通用模型行 Amendment v3：批 2/批 1 骨干 × 三条新评测行（M1/M6/S1）

- **change-id**: benchmark-v2-generalist-rows-amendment-v3-new-rows-20261006
- **落盘日期**: 2026-10-06（发射前冻结；执行触发 = 用户 2026-10-06 指令「发射批 2 扩新行，GPU 空闲即可用」）
- **状态**: DESIGN FROZEN → 点火（本 amendment 落盘即冻结，随后立即执行）
- **上游**: `benchmark_v2_generalist_rows_amendment_v2.md`（§5 批 3 行明文：「全部新行扩满 13 格（含 M1/M6/S1）」）——本 amendment 是该条款的**执行细化**，不是新口径。
- **关系**: 只增行（append-only）；不改动既有 14 族任何读数、不改动任何门槛、不替换任何行。

## 1. 目的与范围

把 **7 个批 2 骨干 + 4 个批 0/1 骨干**（共 11 个已测骨干）扩展到三条新评测行（M1 / M6 / S1），回答：**「换库塌缩」现象（5 个移植族新行全 <0.09）是否在探针协议骨干群体上同样成立？**——这是发现二（监督体制 × 数据几何）在第三组模型群体上的外推检验。

- **模型（11）**：orthrus_4track / orthrus_6track / codonfm_80m / mrnalm_5utr / mrnalm_3utr / calm / lucaone（批 2 七行）+ rinalmo_micro / rinalmo_mega / rinalmo_giga / ernierna（批 0/1 四行）。
- **任务（3 新行 × S1 双臂 = 4 列）**：m1_mrl（2,805 行）/ m6_ndd5utr（800 行）/ s1_stability 5UTR 臂 + 3UTR 臂（5,572 子行）。
- **不做什么**：不微调骨干；不改 5 个移植族的新行读数（已冻结）；不把结果用于改写任何既有判定。

## 2. 口径（冻结；与既有行逐字同款，唯一新决策 = 新行探针的拟合面）

| 项 | 冻结值 | 依据 |
|---|---|---|
| 骨干 | 冻结官方权重，`eval()`，零梯度 | amendment v2 §2 |
| 打分头 | 线性探针（Δh → Δy 回归，seed 20260816，AdamW lr 1e-3 wd 1e-4，100 epoch，source-group 加权 MSE） | te-family 协议（MRL 榜单行同款） |
| **新行探针拟合面（本 amendment 的唯一新决策）** | **同库同任务 TRAIN**：M1 行 = MRL 任务 TRAIN（GSE114002 canonical TRAIN split）；M6 行 = MRL 任务 TRAIN；S1 行 = half_life 对应区域 TRAIN | 与既有 9 任务 IN_STUDY_PROBE 模式一致（每个任务探针在自身 TRAIN 拟合、VALIDATION 评测）；新行无自身 TRAIN → 用**同终点同区域**的既有任务 TRAIN（M1/M6 = MRL；S1 = HL）。选择理由：探针监督信号必须与被评测表面同终点，否则测的是跨终点迁移而非骨干表征的可读性 |
| 评测器 | frozen Task-1 evaluator（K=10），observations 从 projection_rows.jsonl 构造（与矩阵 runner load_new_row 同款字段契约） | benchmark_v2 matrix runner 先例 |
| Δ 口径 | Δ̂ = 探针(候选) − 探针(源)（线性头 bias 相消 = Δh 线性） | te-family 协议 |
| protected TEST | reads = 0（TRAIN 拟合 + 新行全量评测，新行本身非 TEST split） | 项目纪律 |
| CUDA | 硬门：`torch.cuda.is_available()` 失败即停留证；禁 CPU 降级 | 项目纪律 |

> **口径诚实说明**：新行评测面 = NEW_EVAL_ROW 全量（M1 2,805 / M6 800 / S1 5,572）；探针拟合用同终点 TRAIN，**不触碰新行任何标签**——新行读数因此是「纯外推」读数（与 5 个移植族的 frozen-Δ 直接打分不同、与批 0-3 的 IN_STUDY_PROBE 同族）。写作时须照此声明，两种口径不得混写。

## 3. 执行批次（点火即按序）

| 批 | 内容 | GPU 预算 |
|---|---|---|
| A | 7 个批 2 骨干 × 4 列（m1/m6/s1_5utr/s1_3utr） | GPU2（~25GB 空闲）与 GPU4（~24GB 空闲）各一进程，flock 序列化防同卡竞争 |
| B | 4 个批 0/1 骨干 × 4 列 | 同上，随 A 完成情况续发 |
| 冒烟 | 每骨干 × m1_mrl 前 50 行（--smoke-limit 50）验证管线 | <5min |

输出：`experiments/analysis_generalist_rows_v2/batch4_newrows_v1/<model>__<task>/run_detail.json` + 汇总 `batch4_newrows_matrix_v1.json`（reporting-only，逐字拷贝，append-only）。

## 4. 判定与报告规则（零事后裁量）

- **reporting-only**：本批全部读数 = 描述性扩展，**不进入任何预注册判定门**，不改写 §5.6 / 既有矩阵行 / 榜单。
- 措辞：不得声称「14 族新行全 <0.09」（那是指 5 个移植族）；本批结果只写作「11 个探针骨干的新行读数 = X」。
- 若出现意外高分（>0.3）：如实报告 + 登记为 noteworthy（不动门），并在写作时核对是否口径混淆（新行探针 = TRAIN 拟合面同终点，理论读数上限不同于 frozen-Δ 直接打分）。
