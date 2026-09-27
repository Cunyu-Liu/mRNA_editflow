# Benchmark v2 通用模型行扩充 Amendment v2（**DESIGN FROZEN — 不执行**）

- **change-id**: benchmark-v2-generalist-rows-amendment-v2-20260928
- **落盘日期**: 2026-09-28
- **状态**: **DESIGN FROZEN（设计冻结，未点火、未下载、未占用 GPU）**；本文件冻结的是**接入方案与判定规则**，不产出任何数字。
- **执行触发条件（唯一）**: 集群出现持续可用显存（单卡 ≥12GB 自由）且主线（M1/polyA 收割）不冲突时，由人工或 watcher 按 §5 批次发射。**在收到点火指令前，本 amendment 不产生任何计算。**
- **上游**: `benchmark_v2_port_ledger_v1.md`（既有 5 族移植终审）+ `benchmark_v2_matrix_row_prereg_v1.md`（矩阵行闸门）+ `benchmark_v2_matrix_v1` 冻结矩阵（65 格）
- **关系**: 本 amendment **只增行**；不改动既有 14 族任何读数、不改动任何门槛、不替换任何行。

---

## 1. 目的与范围

把 8 个候选模型（用户 2026-09-28 指定）纳入 DeltaBench 的「通用/领域模型」行，用于回答一个具体问题：**换更强的通用序列模型骨干，能不能拿到 Δ 能力？**——这是发现二（监督同类性 × 密度）的直接压力测试。

- **模型（8）**：RiNALMo（micro/mega/giga 三规模）、ERNIE-RNA、Orthrus、CodonFM、CaLM、AIDO.RNA、LucaOne、mRNA-LM。
- **任务范围（分批，见 §5）**：**先 3 个头条任务** = MRL（GSE114002）/ polyA（GSE269595）/ MPRAU（ENCSR854RUF）；批 3 再扩满 13 格（含 M1/M6/S1 三新行）。
- **不做什么**：不做任何微调（不含 LoRA/全参微调）；不做工程指标承诺；不把文献分数搬进榜。

## 2. 口径（冻结：逐字复用既有行，不重定义）

| 项 | 冻结值 | 依据 |
|---|---|---|
| 骨干 | **冻结**（官方权重，`eval()`，无梯度） | 与既有 RNA-FM / UTR-LM 行同款 |
| 打分头 | **线性探针**（任务 TRAIN 上拟合；量级锚：RNA-FM 行 = 640+1 参数、UTR-LM 行 = 128+1 参数） | 既有行实测声明 |
| α/epoch 选择 | 由**任务 VALIDATION** 的 source-group 加权 MSE 选择（HPO_VALIDATION_ONLY 协议） | 既有行同款 |
| 评测器 | frozen Task-1 evaluator（K=10），VALIDATION only | 项目冻结评测器 |
| Δ 口径 | Δ̂ = 探针(候选序列) − 探针(源序列)；指标 = task macro Spearman（MPRAU 主口径 = variant pair-mean，n=2,008） | 与矩阵行口径一致 |
| 代码路径 | `scripts/route_a_v3/run_route2_frozen_delta_te_family_v1.py` → `run_route2_frozen_delta_full_coverage_v1.py`：**只新增“嵌入提取”模型分支**，探针/评测/统计代码零修改 | 见 §4 端口验证 |
| protected TEST | **reads = 0**（仅 TRAIN 拟合 + VALIDATION 评测） | 项目纪律 |
| CUDA | 硬门：`torch.cuda.is_available()` 失败即停留证；禁 CPU 静默降级 | 项目纪律 |

> **口径诚实说明（写作时须照抄）**：本行不是 “zero-shot”。骨干零调参，但**线性头在任务 TRAIN 上拟合**——这正是既有 RNA-FM / UTR-LM 行的口径；把它写成 “零样本” 属过度声明。

## 3. 模型清单、权重来源与适配风险（实地盘点 + 联网核实，全部带 locator）

| 模型 | 官方渠道（locator） | 服务器现状（实测） | 权重/格式 | License（风险） | 输出范式 | 显存/适配难度 |
|---|---|---|---|---|---|---|
| **RiNALMo-micro / mega / giga** | HF `multimolecule/rinalmo-{micro,mega,giga}` | ✅ 已落盘：`hf_home/models/multimolecule--rinalmo-{micro,mega,giga}`（256M / 1.2G / 2.5G）；另 `rna-jepa/weights/rinalmo-giga/` | safetensors | 待逐份核对（HF 模型卡） | embedding（无标量头）→ 用线性探针 | 低（`multimolecule` 0.2.x 原生；与 RNA-FM 同 `RnaTokenizer` 家族） |
| **ERNIE-RNA** | HF `multimolecule/ernierna` | ✅ 已落盘：`hf_home/models/multimolecule--ernierna`（659M） | safetensors/bin | 待核对 | embedding → 线性探针 | 低（同上；768 维 / 12 层 / 86M 单卡可跑） |
| **Orthrus**（4-track / 6-track） | GitHub `bowang-lab/Orthrus`；HF `antichronology/orthrus-*` | ❌ 未落盘 | F32 safetensors（4-track ≈10.2M） | MIT（低风险） | Mamba/SSM `representation()`；对比学习版无 token 预测头 | **中**：需 CUDA + `mamba-ssm`（服务器已装 2.2.4）；`trust_remote_code` |
| **CodonFM**（80M/600M/1B） | GitHub `NVIDIA-Digital-Bio/CodonFM`；HF `nvidia/NV-CodonFM-Encodon-*`；NGC | ❌ 未落盘 | PyTorch ckpt（格式待核实） | **NVIDIA Open Model License（须审阅，可能限制再分发/商用）** | codon tokenizer（≈2046）+ CLS embedding；可选回归头 | 中-高：验证码/下载条款 + 1B 显存 |
| **CaLM** | GitHub `oxpig/CaLM`；权重 `next.opig.stats.ox.ac.uk/...calm_weights.pkl` | ❌ 未落盘 | pkl（pickle，需自写 codon 预处理） | **未标 license（须作者确认）** | codon MLM embedding | 中-高：非 HF 原生加载路径 |
| **AIDO.RNA（1.6B）** | HF `genbio-ai/AIDO.RNA-1.6B`；`genbio-ai/modelgenerator` | ⚠️ **残缺**：`hf_home/models/genbio-ai--AIDO.RNA-1.6B`（仅 `pytorch_model-00001-of-00002.bin.part`）；架构代码已在 `/mnt/cunyuliu/gbrna_pkg/modeling_rnabert.py`；同族 GB.RNA-1.6B 完整（13G） | bin（分片） | HF 标 other/NOASSERTION（须确认） | embedding + 可选回归头 | 中：需续传 + 1.6B 显存（≥24GB 建议） |
| **LucaOne（1.8B）** | GitHub `LucaOne/LucaOne`；HF `LucaGroup/lucaone`；Zenodo `10.5281/zenodo.15171943`；FTP | ❌ 仅 conda env（`envs/lucaone`，torch 2.5.1 / transformers 4.26） | 多 step checkpoint | Apache-2.0（低风险） | 39-token（gene+protein）embedding | 中-高：下载渠道多（FTP/Zenodo）+ 1.8B 显存 |
| **mRNA-LM** | GitHub `Sanofi-Public/mRNA-LM`；NAR 2025 `10.1093/nar/gkaf044`；Zenodo `10.5281/zenodo.14606043` | ❌ 未落盘 | 三段权重 zip（CodonBERT / 5UTR / 3UTR）+ CLIP 整合 | 待核实（Sanofi 公共仓库） | 分段 BERT + CLIP 整合；embedding/微调头 | 中-高：需按官方整合路径拼装 |

**来源纪律**：本表 “官方渠道” 列来自联网检索（各 GitHub/HF/Zenodo/NAR 页面）；“服务器现状” 列来自 2026-09-28 只读实地盘点。**未落盘项禁止在表格中写成可得**；`License 待核实/须审阅` 的模型在其 license 明确前**不得进入对外榜**（只能内部诊断）。

## 4. 端口验证（先做，未通过不得进入批 1）

新增模型分支必须满足以下**端口验证**，否则整体不执行：

1. **回归验证（必做）**：用新代码路径重跑既有 **RNA-FM 行**于 GSE114002（MRL），与入档值逐位比对（既有 te-family 端口验证的先例：UTR-LM 0.1107267878538859 精确一致、RNA-FM |Δ| = 2.4e-6）。**容差固定 |Δ| ≤ 1e-5**，超差即视为端口污染，停止并排查。
2. **官方示例单测**：每个新模型用其官方 README 示例序列（或官方 notebook 输入）跑一次前向，确认能加载、能出 embedding、维度与 config 一致。
3. **适配声明落盘**：每个模型写清 tokenizer/词表、T→U 或 codon 切分、池化方式（non-special token mean-pool 或 CLS）、长度切分/截断策略、批次上限；写入 `frozen_delta_results.json` 的 adapters 段。
4. **权重指纹**：所有新下载权重记录 `sha256` + 来源 URL + 下载日期（防来源漂移）。

## 5. 执行批次（点火后按序；每批独立可交付）

| 批次 | 内容 | 前置 | 预计资源 |
|---|---|---|---|
| **批 0** | 端口验证（§4.1 + §4.2） | 无 | 单卡 <30min |
| **批 1** | 已有权重 4 行：RiNALMo-micro / mega / giga + ERNIE-RNA × 3 头条任务（MRL / polyA / MPRAU） | 批 0 通过 | 4 行 × 3 格；micro/mega/ERNIE 单卡可跑，giga 建议整卡（650.9M） |
| **批 2** | 需获取 6 行：Orthrus / CodonFM / CaLM / AIDO.RNA / LucaOne / mRNA-LM × 3 头条任务（license 已明确者优先） | 批 0 通过 + 权重落盘 + license 结论 | 1B 级模型需整卡；建议 2-3 张卡并行 |
| **批 3** | 全部新行扩满 13 格（含 M1/M6/S1） | 批 1/2 完成且 3 格读数有信息量 | 按格位增量 |

**资源与占用规则**：单卡 ≥12GB 自由显存即可发射；多卡并行时逐卡独立进程；发射沿用 `launch_*` 的幂等 + flock 序列化 + 每尝试独立 log 模式；**禁止**因显存不足降级到 CPU。

## 6. 判定与报告规则（零事后裁量）

1. **只增行**：新行一律 append；不修改、不替换既有 14 族任何读数。
2. **三任务先批次只作内部信号**：批 1/2 的 3 格读数**不得**写成最终科学结论（smoke/proxy 纪律）；只有批 3 满 13 格后、且与既有行同表同口径，方可进入论文表格。
3. **不可得者如实登记**：权重拿不到 / license 不允许 / 范式不符 → 记为 `NOT_AVAILABLE` / `RIGHTS_BLOCKED` / `PARADIGM_MISMATCH`，**不以文献分数补位**（与既有 5 个不可测候选同一处理）。
4. **负结果照登**：若这 8 个模型在 Δ 上全线接近 0（预期大概率如此，按发现二），这本身就是“更强骨干≠Δ 能力”的正面证据，如实写入。
5. **不 peak-picking**：规模变体（micro/mega/giga）全报，不得只报最好的一档；也不得用“最好的那档”替换模型行。
6. **统计**：新行与既有行对比使用既有 Holm 家族框架的**追加行**（原 4 行不动）；新家族的多重比较另立家族，报告 raw p 与 Holm（家族内）并注明家族定义。

## 7. 风险与回退

| 风险 | 触发信号 | 回退动作 |
|---|---|---|
| License 不允许 | CodonFM（NVIDIA Open Model License）/ AIDO.RNA（NOASSERTION）/ CaLM（未标） | 不进对外榜；内部诊断行加 `INTERNAL_ONLY` 标记；必要时联系作者留证 |
| 权重来源失效 | Sanofi CDN / OPIG / FTP / Zenodo 直链 404 | 立即停止该模型，登记 `NOT_AVAILABLE` + 失败证据 |
| 架构不兼容 | Mamba 无 CUDA 扩展 / 非 HF pickle 加载失败 | 该模型退化为 `PORT_FAILURE`，其余模型照跑（矩阵缺格机制已有） |
| 显存不足 | 1B 级模型 OOM | 降低批大小（不改协议）；仍失败则该规模行记 `GPU_CAPACITY` 并保留证据；**不降级到 CPU** |
| 端口污染 | §4.1 回归验证超差 | 全批停止，先修端口 |

## 8. 交付物

1. 新行读数 JSON（每模型每格 + 逐 seed/探针维度）+ 与既有行的同表对比。
2. 每模型适配声明（§4.3）+ 权重指纹（§4.4）。
3. 端口验证证据（§4.1 逐位比对日志）。
4. 论文侧：矩阵表扩行 + §3 一句“骨干规模/架构不是 Δ 能力的充分条件（新增 X 族证据）”；PPT 侧：替换本页状态行为实测行。
5. journal 批次记录（含失败项与资源账）。

## 9. 纪律自检

- [x] 本体为 **DESIGN FROZEN**: 未下载、未点火、未占 GPU、未读 TEST
- [x] 口径逐字复用既有行，不新开口径；只增行不删行
- [x] 端口验证（回归 RNA-FM 逐位）为硬前置
- [x] 不可得/license 受限项如实登记，不用文献分数补位
- [x] 三任务批次不写成最终结论；规模变体全报（禁 peak-picking）
- [ ] 点火前需人工指令（用户 2026-09-28：显卡紧张，先设计不跑）
