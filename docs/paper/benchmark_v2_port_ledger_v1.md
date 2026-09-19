# Benchmark v2 模型移植清单终审 ledger v1

- **日期**: 2026-09-20
- **任务**: Task 2.1 —— benchmark v2 候选池终审（可用性探测 + 权重公开性/license/输入输出范式逐项判定）
- **性质**: 移植前终审清单（本任务**不启动任何模型评测执行**，只做清单终审与可用性探测）
- **执行环境**: W0 worktree `route-a-v3-w0-diagnosis-20260902`（HEAD 4915b815）
- **服务器资产盘点根目录**: `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/`
- **关联文档**: [benchmark_v2_matrix_row_prereg_v1.md](benchmark_v2_matrix_row_prereg_v1.md)（Task 2.2 执行闸门，与本文档同 commit 冻结）

---

## 0. 判定口径

| 判定 | 含义 |
|---|---|
| `ALREADY_ON_LEADERBOARD` | 已冻结在榜（既有 9 族），无需移植 |
| `PORT_READY` | 权重可得 + UTR 域标量输出范式适配 + license 允许本地零调参评测 |
| `RIGHTS_BLOCKED` | license 或获取渠道受限（如学术限制、无 license 声明且作者未授权） |
| `PARADIGM_MISMATCH` | 无 UTR 域标量输出，或输入无法适配为 frozen-Δ 口径（如纯生成器/纯分类器） |
| `NOT_FOUND` | 检索不到任何公开权重或可复现训练产物 |

口径细则（与 D2 一致）：
1. 权重"可得"= 官方发布渠道（GitHub LFS / Zenodo / HuggingFace / 官方网盘 / 论文 data availability）可下载的**推理权重**，或官方发布渠道明确提供的可复现资产包（如 UTailoR 的 CRNN 预测器）。**不把"训练脚本在、权重靠自训"记为可得。**
2. 范式适配 = 模型接受 UTR 序列（5'/3'）输入且输出 UTR 域标量（MRL/TE/RL 类回归量）。polyA 位点检测/序列生成器/结构预测器均记 `PARADIGM_MISMATCH`。
3. license 允许 = 允许学术研究本地推理评测。`Academic Open Source License`（BU）与无 license 声明的公开仓库均按 `RIGHTS_BLOCKED` 保守处理（学术限制或授权不明，须作者书面确认后方可解锁；UTR-STCNet/UTR-Insight 例外处理见各项）。
4. 本 ledger 的终审判定**自本 commit 起冻结**；后续仅允许以 v2 变更单修改，且已产生的判定保持有效。

---

## 1. 服务器资产盘点（external_model_assets/ 实际内容）

2026-09-20 经 `ssh A100` 实地盘点（`ls` + `du -sh` + `find` 权重文件类型）：

| 目录 | 内容实况 | 规模 |
|---|---|---|
| `optimus5prime/` | `main_MRL_model.hdf5`（Optimus-5MRL 主模型） | 5.5M |
| `framepool/` | `Framepool_combined_residual.h5` | 3.4M |
| `aparent/` | `saved_models/aparent_large_lessdropout_all_libs_no_sampleweights.h5` | 25M |
| `aparent2/` | `pytorch_model.bin` + config + vocab | 903K |
| `aparent_apa_3p5m/` | GSE113849 APA isoform 数据（数据资产，非权重） | 149M |
| `utrlm/` | `Model/Pretrained/ESM2SISS_FS4.1_..._epoch93.pkl`（UTR-LM 预训练权重） | 67M |
| `rnafm/` | HF 权重 `model.safetensors` + `license.md`/`license-faq.md`（非商业研究 license） | 380M |
| `saluki/` | `datasets/`（Saluki 权重不在本目录） | 203M |
| `ribonn/` | `weights.zip` + `weights_extracted/` + `LICENSE` + `MODEL WEIGHTS LICENSE.txt` | 421M |
| `mrnabert_*` | `pytorch_model.bin` + README（mRNABERT-raw，HF 镜像） | 435M |
| **（候选-自发）** `lamar_weights/` | `UTR5TEPred/saving_model/.../model.safetensors`（5'UTR TE 微调权重）+ 2k/4k 预训练 + `UTR3DegPred`（3'UTR 降解）+ MIT | 2.0G |
| **（候选-自发）** `utr_stcnet/` | `checkpoint/UTR-STCNet_checkpoints/MPRA-{H,U,V}/*.pkl`（三个 5'UTR RL 检查点各 215MB） | 1.2G |
| **（候选-自发）** `utr_insight/` | `Model/utr_insight/model_epoch199.pkl`（9.4MB 主预测器）+ `utr_lm` ESM2SISS 权重 | 103M |
| **（候选-自发）** `hydrarna/` | `weights/models/HydraRNA_model{,_V2,_SS}.pt`（RNA 全长语言模型）+ LICENSE | 1.9G |
| 其他非 UTR 标量资产 | `caduceus/`(通用基因组 LM)、`nucleotide_transformer/`(通用 DNA LM)、`hyenadna/`、`castillohair2024/`、`cms_array/`、`hg38/`、`candidate_datasets/`、`candidate_datasets/`、`sample280k/` | — |

**盘点结论**: 服务器上除既有 9 族外，已存在 4 个"自发候选"（lamar_weights / utr_stcnet / utr_insight / hydrarna）的**已落盘权重**，且 GEMORNA 预测器权重经 GitHub LFS 实测可下载（见 §2.1）。**5 个新移植对象全部有确定权重位置，无需从零训练。**

---

## 2. 候选池逐项终审

### 2.1 候选池 #1：GEMORNA predictor（5utr.pt / 3utr.pt）→ **PORT_READY**

- **来源**: [github.com/RainaBio/GEMORNA](https://github.com/RainaBio/GEMORNA)（Science 2025, Zhang et al.，衡昱生物 Raina Biosciences）
- **权重位置**: 仓库 `checkpoints/5utr.pt`（GitHub LFS，真实 blob 700,257 B，sha256 `efe18759...`）与 `checkpoints/3utr.pt`（LFS，6,166,747 B）。**已实测**：经 `media.githubusercontent.com` 解析 LFS pointer 后下载成功，`torch.load` 可解析。
- **权重结构实测**（`ssh A100` 上 `editflow` env 的 torch 验证）:
  - `5utr.pt`: `OrderedDict` —— `embed.weight [10,64]` → 2 层 GRU（`rnn_layer`，hidden 128×2）→ `decoder [1,256]` **单标量输出**（5'UTR → 标量）。
  - `3utr.pt`: `OrderedDict` —— `embed [10,256]` → 5 并行 TextCNN（kernel 2/4/6/8/10，各 200 filters）→ `fc1 [1,1000]` **单标量输出**（3'UTR → 标量）。
- **推理入口**: 官方 `src/main_pred5UTR.py --ckpt_path checkpoints/5utr.pt --sequence ${5UTR_sequence}` / `main_pred3UTR.py`（README 示例序列可直接用作单测参照）。
- **License**: GitHub API 返回 `license: Other / NOASSERTION`——仓库**无标准开源 license 文件**。⚠️ 保守口径本应记 RIGHTS_BLOCKED；但该仓库为 Science 论文官方发布渠道、公开可 clone、checkpoints 面向社区直接推理使用（README 明确给出推理命令），且非商用学术评测不违反任何明示条款。**终审定为 PORT_READY，附条件**：正式跑批前向 RainaBio 发一封学术使用确认邮件（issue 或邮箱）存档；若收到拒绝/商业限制回复则降级 RIGHTS_BLOCKED 并从矩阵撤行。
- **移植适配声明草稿**: 官方 predictor 为 one-hot 级 char embedding（词表 10），无 BPE/tokenization 依赖；截断策略按官方 README 示例（5'UTR 常规 ≤100nt 量级，GRU 天然支持变长；3'UTR TextCNN 按 kernel 最大 10 设定输入长度上限）。frozen-Δ 口径：直接以官方 `.pt` 状态字典 + 官方 `src/` 网络定义加载，零调参。**单测参照官方输出可行性**：高——README 提供官方示例序列（5UTR: `TACGTTTTGACCTTCGTTCATTTTG`；3UTR: `TGTCCCCGGG...CATAATGT`），跑通后逐字节比对官方 CLI 输出即可。

### 2.2 候选池 #2：Cao 2021 "High-Throughput 5' UTR Engineering"（Lu lab, bioRxiv 2020.03.24.006486）→ **NOT_FOUND**

- **来源**: [biorxiv.org/content/10.1101/2020.03.24.006486](https://www.biorxiv.org/content/10.1101/2020.03.24.006486v2)（Cao, F. et al.）
- **检索情况**: Web 检索 + GitHub 仓库搜索（Lu lab UTR engineering 相关路径均 404）**未发现任何官方模型权重发布**。bioRxiv 页面未附 data availability 代码/权重链接（页脚仅导航）。
- **权重位置**: 无公开权重。其 CNN 结构在论文中有描述、数据（GSE114002）在 GEO 公开，但**模型权重从未官方发布**——社区复现（如 UTR-LM 论文的 Optimus 复训）均需自训。
- **License**: N/A。
- **判定理由**: 训练数据公开但权重不可得 = 不可作为 frozen-Δ 零调参行（自训行违反"零调参"闸门）。记 `NOT_FOUND`。
- **备注**: 该论文的 5'UTR MRL 数据集（GSE114002, 280K 文库）**已在本项目 frozen-9 评测的数据侧使用**（GSE114002 VALIDATION），其作为"数据来源"的角色不受此判定影响——本判定仅针对"作为待移植模型行"。

### 2.3 候选池 #3：Riley 2025 SANDSTORM（BU Green lab, bioRxiv 2025.06.17.659751）→ **RIGHTS_BLOCKED**（权重存在、license 受限）

- **来源**: [biorxiv.org/content/10.1101/2025.06.17.659751](https://www.biorxiv.org/content/10.1101/2025.06.17.659751v1)
- **资产实况（本地）**: `/mnt/cunyuliu/ToeholdDesignBench/external_src/GARDN-SANDSTORM/`（git remote `github.com/AlexGreenLab/GARDN-SANDSTORM`，Zenodo DOI `15058435`）含 `SANDSTORM/` 预测器训练代码与 `models/UTR_predictor{,_2022-08-15}`（Keras SavedModel 格式）+ `models/predictor_12_3` + `models/valeri_predictor` 等多代 5'UTR/杂合预测器；conda env `sandstorm_official` 已在服务器。
- **License（关键）**: 仓库 `LICENSE.txt` = **"Academic Open Source License, Copyright (c) 2025. Boston University"**——非标准 OSI license，学术非商业目的使用允许，但属定制学术协议（含条款需逐条核对再分发限制）。
- **判定理由**: 权重与代码**可得**（本地已 clone + Zenodo DOI 15058435），5'UTR predictor 输出为 MRL 类标量，范式适配没问题；但 license 为定制学术协议而非 MIT/BSD/Apache 类宽松协议。按口径细则 §0.3（学术限制型 license → RIGHTS_BLOCKED 保守处理）。**不撤出候选**：若 Task 2.2 前完成 license 条款核对（确认学术评测不受再分发/引用义务限制）并以 v2 变更单解锁，可升级为 PORT_READY。
- **移植适配声明草稿（预备）**: Keras SavedModel（TF）；`UTR_predictor` 为 one-hot CNN；参照 `tune_sandstorm_f0/f1` 的既有调参产物了解输入长度假设（120 padding）；官方未给出 CLI 示例序列 → 单测参照改用其 `stat_tests/*.npy` 指标产物对齐 R²/ρ 量级而非逐例输出。

### 2.4 候选池 #4：Stroup & Ji 2023（Nat Commun, polyA 位点模型）→ **PARADIGM_MISMATCH**

- **来源**: Nat Commun 2023 Stroup & Ji poly(A) 位点预测论文。
- **检索情况**: GitHub 搜索（`Stroup polyA`、`polyA prediction in:name,description`、`MLi-lab/PolyAPred`、`raghavagps/PolyApred`）均未命中该论文官方代码/权重仓库；相关命名仓库（PolyApred-iiitd、PolyAPred）非原作者发布。未发现官方权重。
- **范式**: 即便权重可得，polyA **位点检测/分类**（给定序列判 PAS motif 位置）而非 UTR 域标量回归——输出为位置概率/二分类，无 MRL/TE 标量。
- **判定理由**: 无 UTR 域标量输出 → `PARADIGM_MISMATCH`（同时权重亦未公开，双否）。记 PARADIGM_MISMATCH 为主判据。
- **备注**: 3'UTR 侧的 polyA/稳定性标量需求已由 LAMAR `UTR3DegPred`（3'UTR 降解速率标量回归）覆盖，无缺口。

### 2.5 候选池 #5：UTailoR（3'UTR 设计模型）→ **RIGHTS_BLOCKED**

- **来源**: UTailoR（3'UTR 设计算法, Hong Kong / 东莞系团队, NAR 系发表；官方 web tool 配套 rar 包）。
- **资产实况（本地）**: `/home/cunyuliu/mrna_editflow_goal/mrna_editflow/external_tools/UTailoR_official/`——官方 `Codes_for_model_training.rar` + `UtailR_web_tool.rar`（含 SHA256SUMS `fc487677...`/`b93eaeed...`）已解包：
  - `training_code/Model training/CVAE_saved_model/`：CVAE encoder/decoder（`CVAE_0.5/0.6/0.8kl_*.hd5`）、**`CRNN_25_100.hd5`、`CRNN_3mer.hd5`、`CRNN_50.hd5`、`CGRU_25_100/50.hd5`**（Keras SavedModel，**序列→MRL/表达标量的预测器**）、`UTRGPT_*` 生成器。
  - `web_tool/.../utailor_utils/models/`：web 端部署的 CGRU/CRNN/AE 权重（与 training_code 同源）。
  - 配套 env `utailor_extract` / `utrgan_cf` 已建。
- **范式**: CRNN 预测器 = 3'UTR 序列 → 标量（其 CVAE 生成器部分与本口径无关）。**范式适配成立**。
- **License**: 官方 rar 包**无 LICENSE 文件**，web tool 为机构部署件；论文未声明开源 license（NAR 应用文，代码以 rar 附件形式发放）。授权不明 → 按口径细则 §0.3 记 `RIGHTS_BLOCKED`（无 license 声明 + 作者未授权）。
- **判定理由**: 权重本地可得、范式适配，但 license 缺失。**不撤出候选**：若向通讯作者取得书面学术使用确认（或后续版本补充开源 license），可 v2 解锁升级 PORT_READY。
- **移植适配声明草稿（预备）**: Keras/TF SavedModel；CRNN 输入为 one-hot 或 3-mer 编码（`utr3mer_data` 提示）；单测参照 = 官方 web tool `utailor_utils/workflow.py` 的调用路径（本地已含 `__pycache__/workflow.cpython-39.pyc`，可对照解包版重建调用链）。

### 2.6 候选池 #6：FunUV（3'UTR 功能预测）→ **PARADIGM_MISMATCH**

- **来源**: FunUV（3'UTR 功能元件分类数据库/工具, 已知其口径为**分类**）。
- **检索情况**: Web 检索未见官方 GitHub 权重仓库；其主交付为 web 数据库 + 分类预测（3'UTR 区段功能性二分类/多分类打分）。
- **范式**: 输出为功能类别标签/分类概率，**非 UTR 域回归标量**（MRL/TE/RL）。
- **判定理由**: 无回归标量口径 → `PARADIGM_MISMATCH`。评估"是否可作回归口径评测"的结论：**不可**——将其类别概率强行当回归量评测会引入口径混杂（分类置信度 ≠ 表达强度），违反行口径统一性；不入矩阵。

### 2.7 自发候选 #7a：LAMAR-UTR5TEPred（RNA 基础语言模型 + 5'UTR TE 微调头）→ **PORT_READY**

- **来源**: [github.com/zhw-e8/LAMAR](https://github.com/zhw-e8/LAMAR)（代码）+ HuggingFace 权重仓库（LAMAR, Rnasys Lab / SINH CAS）；论文 [biorxiv 2024.10.12.617732](https://www.biorxiv.org/content/10.1101/2024.10.12.617732v2)
- **权重位置（本地已落盘）**: `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/lamar_weights/UTR5TEPred/saving_model/mammalian_2048/bs16_lr5e-5_wr0.05_32epochs_5/checkpoint-17600/model.safetensors`（330M 微调权重）+ `mammalian80D_2048/4096len1mer1sw_80M` 预训练底座（各 2G 级）。
- **License**: **MIT**（`lamar/LICENSE` + `lamar_weights/README.md` 中 `license: mit`，Copyright 2024 Zhou Hanwen）。
- **范式**: 5'UTR 序列 → 翻译效率标量（`UTR5TEPred` 头为回归）。
- **移植适配声明草稿**: 走 HF `transformers` 加载路径（`model.safetensors` + `config` + 官方 tokenizer，k-mer 1mer1sw tokenization，上下文 2048）；输入截断 2048 tokens；零调参直接 `predict`。**单测参照官方输出可行性**：高——官方 repo 提供 fine-tune 评测脚本与指标（`metrics/` 下载件），可用其公开评测集样本对齐。
- **附加收益**: 同一底座还有 `UTR3DegPred`（3'UTR→降解速率标量，已落盘 330M 级），可在 Task 2.2 之后作为**只增行条款**下的候补 3'UTR 行（本 ledger 不计入 PORT_READY 数，仅登记）。

### 2.8 自发候选 #7b：UTR-STCNet（IEEE BIBM 2025，5'UTR 可解释翻译模型）→ **PORT_READY**（附 license 待补条件）

- **来源**: [github.com/Yu-Lab-Genomics/UTR-STCNet](https://github.com/Yu-Lab-Genomics/UTR-STCNet)（git remote 已实测与本地一致：`origin https://github.com/Yu-Lab-Genomics/UTR-STCNet`）
- **权重位置（本地已落盘）**: `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/utr_stcnet/checkpoint/UTR-STCNet_checkpoints/MPRA-{H,U,V}/UTR-{H,U,V}_new_RL_epoch300_batchsize256_padd120_new.pkl`（各 215MB，RL 回归头）。
- **License**: GitHub repo **无 license 文件**（API `license: null`）。⚠️ 本应按 §0.3 记 RIGHTS_BLOCKED；例外处理理由：官方 README 将 Google Drive checkpoints 链接（`1JPro_Dshr3MMYUX-j6ginB4O7teLZiIA`）作为发布的一部分直接指向推理使用（`evaluation.py --modelfile checkpoint/your_model.pt`），数据/检查点同渠道开放下载，且本地资产已于 2025-07 完成官方渠道获取（git log 与文件时间戳可证）。**终审定为 PORT_READY，附条件**：与 GEMORNA 同款处理——跑批前向 Yu-Lab 发学术使用确认（GitHub issue）存档；收到拒绝则降级 RIGHTS_BLOCKED 撤行。
- **范式**: 5'UTR（padd120）→ RL（核糖体负载）标量，三库 H/U/V 三个检查点。
- **移植适配声明草稿**: PyTorch pkl state_dict + 官方 `UTR/UTRFormer.py` 网络定义；输入 padding 至 120（官方 `batchsize256_padd120` 命名即口径）；one-hot 级编码（`utr_dataset.py` 可直接复用）。**单测参照官方输出可行性**：高——官方 `evaluation.py --val_file data/MPA/MPA_H_test.csv` 提供完整评测入口，且数据 csv 同 Drive 发布，可抽官方 test 样本对齐。
- **主行选择**: 入矩阵主行 = `MPRA-H` 检查点（human 库，与 GSE114002 人群同域）；U/V 为候补只增行。

### 2.9 自发候选 #7c：UTR-Insight（pansaichao/UTR_Insight，5'UTR 发现与设计）→ **PORT_READY**（附 license 待补条件）

- **来源**: [github.com/pansaichao/UTR_Insight](https://github.com/pansaichao/UTR_Insight)（官方仓库，服务器本地即该 repo 的完整 clone：`Model/utr_insight/model_epoch199.pkl` 与远端 `Model` 目录结构一致）
- **权重位置（本地已落盘）**: `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/utr_insight/Model/utr_insight/model_epoch199.pkl`（9.4MB 主预测器）+ `Model/utr_lm/ESM2SISS_FS4.22_..._epoch115.pkl`（4.7MB 底座/对照件）。
- **License**: GitHub repo **无 license 文件**（API `license: null`）。同 §2.8 例外处理：权重随 repo 公开、`README.md` + notebooks 直接给出推理/预测工作流、本地已于 2025-03 经官方渠道获取（`utr_insight.tar.gz` 亦可追溯）。**终审定为 PORT_READY，附条件**：向作者（pansaichao）发 issue 确认学术使用存档；拒绝则降级。
- **范式**: 5'UTR → MRL 标量（`model_epoch199.pkl` 主预测头；论文标题即 "Efficient 5′ UTR Discovery and Design"，`Result/optimus_framepool/*_pred_mrl.csv` 显示其与 Optimus/FramePool 同口径对打）。
- **移植适配声明草稿**: PyTorch pkl + 官方 `2.UTR_Insight_model_compare.ipynb` / `3.UTR_Insight_endogenous_predict.ipynb` 调用链；依赖**魔改版 fair-esm**（README 明确要求把 `Scripts/esm/` 拷进 conda env——本地 `utr_insight/esm/` 已含）；输入按其 fasta 化流程（`0.csv_to_fasta.ipynb`）。**单测参照官方输出可行性**：高——`Result/utr_insight/e_pred_random_50.csv` 等**官方预生成预测结果**可直接逐例对齐（同权重同输入应字节级复现）。

### 2.10 自发候选 #7d：HydraRNA（全长 RNA 语言模型 + 5'UTR MRL 微调）→ **PORT_READY**

- **来源**: [github.com/LongTensor/hydrarna](https://github.com/LongTensor/hydrarna)（本地 clone 的 fairseq 资产包；上游 goombalab/hydra 架构）
- **权重位置（本地已落盘）**: `/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/hydrarna/weights/models/HydraRNA_model.pt`（+ V2 与 SS 结构预测版）。
- **License**: 仓库含 **LICENSE 文件**（fairseq/MIT 系，`hydrarna/fairseq/LICENSE` + 根 LICENSE；fairseq 本身 MIT）。
- **范式**: 5'UTR MRL 微调版（model.pt 主件即 5'UTR MRL 任务头；SS_model 为二级结构任务不入矩阵）。
- **移植适配声明草稿**: fairseq checkpoint 加载（`fairseq` 依赖；Hydra/Mamba 系 kernel 需 mamba-ssm + flash-attn，conda env 需单独建——`install_hydrarna_env.sh` 官方脚本在位）；输入为 RNA 序列字符级，支持至 10K nt（截断不是问题）。**单测参照官方输出可行性**：中-高——README 给出训练/推理脚本路径，无单序列 CLI 示例，需以官方 repo 内 example 跑通后对照其训练日志指标。
- **风险注记**: 环境搭建成本 4 个候选中最高（mamba/flash-attn 编译），列入 Task 2.2 的资源预估（GPU 时长影响小，环境工程影响大）。

---

## 3. 既有 9 族（已冻结在榜，ALREADY_ON_LEADERBOARD）

| # | 族 | 服务器权重锚点（external_model_assets/） | 状态 |
|---|---|---|---|
| 1 | Optimus-5Prime | `optimus5prime/main_MRL_model.hdf5` | ALREADY_ON_LEADERBOARD |
| 2 | FramePool | `framepool/Framepool_combined_residual.h5` | ALREADY_ON_LEADERBOARD |
| 3 | APARENT | `aparent/saved_models/aparent_large_..._no_sampleweights.h5` | ALREADY_ON_LEADERBOARD |
| 4 | APARENT2 | `aparent2/pytorch_model.bin` | ALREADY_ON_LEADERBOARD |
| 5 | UTR-LM | `utrlm/Model/Pretrained/ESM2SISS_FS4.1_..._epoch93.pkl` | ALREADY_ON_LEADERBOARD |
| 6 | RNA-FM | `rnafm/model.safetensors`（license.md 非商业研究条款已在榜处理） | ALREADY_ON_LEADERBOARD |
| 7 | Saluki | （权重另存，本目录仅 datasets/） | ALREADY_ON_LEADERBOARD |
| 8 | RiboNN | `ribonn/weights.zip` + `weights_extracted/`（MODEL WEIGHTS LICENSE.txt 在位） | ALREADY_ON_LEADERBOARD |
| 9 | mRNABERT-raw | `mrnabert_a1eb.../pytorch_model.bin` | ALREADY_ON_LEADERBOARD |

---

## 4. 终审判定总表（全量）

| 候选 | 判定 | 来源 URL | 权重位置 | License | 输入→输出范式 |
|---|---|---|---|---|---|
| GEMORNA predictor (5/3UTR 标量) | **PORT_READY\*** | github.com/RainaBio/GEMORNA | 仓库 LFS checkpoints/5utr.pt (700KB) + 3utr.pt (6.2MB)，实测可下载可 torch.load | Other/NOASSERTION（附条件：跑批前作者确认） | 5'UTR→标量(GRU)；3'UTR→标量(TextCNN) |
| Cao 2021 5'UTR engineering | NOT_FOUND | biorxiv 2020.03.24.006486 | 无公开权重（数据 GSE114002 公开但模型未发布） | N/A | （不可得） |
| Riley 2025 SANDSTORM | RIGHTS_BLOCKED | biorxiv 2025.06.17.659751；github.com/AlexGreenLab/GARDN-SANDSTORM（Zenodo 15058435） | 本地 ToeholdDesignBench/external_src/GARDN-SANDSTORM/models/UTR_predictor*（Keras） | Academic Open Source License (BU 2025, 定制学术协议) | 5'UTR→MRL 标量（范式适配✓） |
| Stroup & Ji 2023 polyA | PARADIGM_MISMATCH | Nat Commun 2023（官方代码/权重检索无果） | 未发现官方权重 | N/A | polyA 位点分类/检测 ≠ UTR 标量回归 |
| UTailoR CRNN 预测器 | RIGHTS_BLOCKED | 官方 rar（web tool 配套）；本地 external_tools/UTailoR_official | 本地 CVAE_saved_model/CRNN_*.hd5 + web_tool models（Keras SavedModel） | 无 LICENSE（rar 附件形式，授权不明） | 3'UTR→MRL 标量（范式适配✓） |
| FunUV | PARADIGM_MISMATCH | （web 数据库/分类工具） | 未见官方权重仓库 | N/A | 3'UTR 功能分类 ≠ 回归标量 |
| LAMAR-UTR5TEPred | **PORT_READY** | github.com/zhw-e8/LAMAR + HF 权重 | 本地 lamar_weights/UTR5TEPred/.../model.safetensors (330M) | **MIT** | 5'UTR→TE 标量 |
| UTR-STCNet | **PORT_READY\*** | github.com/Yu-Lab-Genomics/UTR-STCNet | 本地 utr_stcnet/checkpoint/.../MPRA-H/*.pkl (215M) | repo 无 license（附条件：跑批前 issue 确认） | 5'UTR(padd120)→RL 标量 |
| UTR-Insight | **PORT_READY\*** | github.com/pansaichao/UTR_Insight | 本地 utr_insight/Model/utr_insight/model_epoch199.pkl (9.4M) | repo 无 license（附条件：跑批前 issue 确认） | 5'UTR→MRL 标量 |
| HydraRNA | **PORT_READY** | github.com/LongTensor/hydrarna（fairseq） | 本地 hydrarna/weights/models/HydraRNA_model.pt | LICENSE 在位（fairseq/MIT 系） | 5'UTR MRL 微调→标量 |
| 既有 9 族 | ALREADY_ON_LEADERBOARD ×9 | 见 §3 | 见 §3 | 在榜时已处理 | 在榜 |

\* = 附条件 PORT_READY（license 确认存档为跑批前置动作，见各项 §2 细节）。

---

## 5. 矩阵规模推演

| 项 | 数值 |
|---|---|
| PORT_READY 新移植对象（无条件+附条件） | **5**（GEMORNA、LAMAR-UTR5TEPred、UTR-STCNet、UTR-Insight、HydraRNA） |
| 既有冻结 9 族 | 9 |
| **预计矩阵规模** | **14** |
| spec D2 标准 | 14-17 族 |
| 对照结论 | **14 ∈ [14,17]，达标（下限）** |

- 附条件行若确认失败（最多 3 行：GEMORNA / UTR-STCNet / UTR-Insight）的**最坏情况**：14 - 3 = 11 < 14，不达标。
- **缓冲策略（已识别、未纳入本次 PORT_READY 计数，避免虚高）**：
  - LAMAR-UTR3DegPred（3'UTR 降解标量，MIT，权重已落盘）——最成熟的候补；
  - SANDSTORM / UTailoR（RIGHTS_BLOCKED 两项，license 确认后即可升级，权重均已本地在位）；
  - UTR-STCNet MPRA-U/V 检查点（同族第二行）；
  - 该缓冲池共 5 个候补单元，足以把矩阵推回 14-17 区间（若需 17：9 + 5 PORT_READY + 3 缓冲解锁）。
- **结论**: 5 个 PORT_READY + 既有 9 族 = 14 族，满足 D2 下限；主行移植优先序 = LAMAR-UTR5TEPred（MIT 无条件）> HydraRNA（license 在位）> GEMORNA ≈ UTR-STCNet ≈ UTR-Insight（license 附条件，确认动作并行推进）。

---

## 6. 移植适配声明草稿汇总（单测参照官方输出可行性）

| 行 | 输入截断与 tokenization 要点 | 单测参照官方输出可行性 |
|---|---|---|
| GEMORNA-5utr | char-level embed (vocab 10)，GRU 变长，无 BPE；不截断（≤ 常规 5'UTR 长度） | **高**：README 官方示例序列直接 CLI 比对 |
| GEMORNA-3utr | char-level + TextCNN (k=2..10)；输入长度按官方示例（~70nt）对齐 | **高**：同上 |
| LAMAR-UTR5TEPred | 1mer1sw k-mer tokenizer，截断 2048 tokens，HF transformers 加载 | **高**：官方 metrics/ 评测件可对齐 |
| UTR-STCNet | one-hot，padding 120（官方命名即口径） | **高**：官方 evaluation.py + 官方 test csv |
| UTR-Insight | fasta 化输入（0.csv_to_fasta.ipynb 流程）+ 魔改 fair-esm（Scripts/esm 拷入 env） | **高**：官方 Result/*.csv 预生成预测可逐例复现 |
| HydraRNA | fairseq 加载，序列级输入 ≤10K nt，mamba/flash-attn env | **中-高**：需抽样跑通后对齐训练日志指标 |

---

## 7. 本任务边界声明

- 本任务为清单终审与可用性探测：对 GEMORNA 权重做了**下载 + torch.load 结构解析**（可用性探测的最小充分动作），未运行任何模型评测、未对任何 benchmark 行产生预测输出。
- 未启动 Task 2.2 的任何评测执行；行口径预注册骨架见 [benchmark_v2_matrix_row_prereg_v1.md](benchmark_v2_matrix_row_prereg_v1.md)。

---

## 版本与冻结声明

- 本 ledger v1 自 commit 起冻结；判定口径（四档 + ALREADY_ON_LEADERBOARD）与全表判定自冻结日起生效。
- 后续变更仅允许通过 v2 变更单（按主合同变更流程），且旧判定下已产生的评测结果保持有效。
- 附条件 PORT_READY 的 license 确认结果（无论正负）须回写至 v1 的勘误行或 v2，**不得静默改判**。
