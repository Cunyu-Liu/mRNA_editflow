# 预注册 v1：delta 差分有效性分析 — oracle Δh-probe 对照臂（Task 1.0.2）

- 日期：2026-09-19（执行前落盘）
- worktree：route_a_v3_w0_diagnosis_20260902
- 目的：以"oracle 差分监督"对照臂隔离**监督形式**变量，判定 frozen LM 表征差分 Δh 是否承载性质差分 Δy 的可读出信号。
- 主产物（规划）：`/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/experiments/analysis_delta_validity_v1/oracle_probe/results_oracle_probe.json` + 汇总 md。

## 1. 对照臂定义（按既有代码事实申报）

- **既有 frozen-Δ LM 行**（对照组）：frozen encoder + 线性 probe。实现源：
  `runs/development_hpo/utrlm_lr1e3_wd1e4_replay_gpu5_v1`（`run_route2_utrlm_baseline_v1.py`）与
  `runs/development_hpo/external_lr1e3_wd1e4_replay_gpu5_v1`（`run_route2_external_prediction_baselines_v1.py`，multimolecule RNA-FM 行），
  任务泛化版 `run_route2_frozen_delta_te_family_v1.py` / `run_route2_frozen_delta_full_coverage_v1.py`。
- **oracle 臂**：同一 frozen encoder（官方权重零调参），probe 改为在任务 TRAIN split 上以 **(Δh, Δy)** 训练：
  特征 h(cand) − h(src)（表征差），目标 Δy = direction_normalized_delta（性质差），VALIDATION 评测。
- 输入适配（照抄既有声明）：UTR-LM = 官方 SISS ckpt，BOS ([cls]) token、layer 6 表征，rotary 无硬长度限制，length-sorted 批 ≤128 seq / ≤16384 tokens；
  RNA-FM = multimolecule RnaTokenizer（T→U），fp32 forward，non-special token mean-pool，>1000 nt chunking（本两任务 ≤164 nt，不触发），批 ≤32 seq / ≤8192 tokens。

### 1.1 监督形式口径核实（预注册前审计发现，如实申报）

预注册前的代码审计发现：**既有行 probe 的代码实现本身就是 Δh→Δy 回归**（`train_probe`/`_train_multimolecule_rnafm_probe`/`fit_probe` 中
特征均为 `embeddings[candidate] − embeddings[source]`、目标均为 `direction_normalized_delta`、z-score 统计取自 Δh），而
te-family/full-coverage 脚本 docstring 中"probe 以绝对标签 y 训练、预测时 Δ̂ = ŷ(cand) − ŷ(src)"的表述与代码不一致。
线性读出下两种形式**数学等价**（bias 相消），数值结果不变。因此本 oracle 臂按**代码事实**实现（同一管线、显式声明 Δh→Δy 监督），
预期与既有行同 seed 完全复现（MRL 行曾端到端复现：UTR-LM 0.1107267878538859 精确一致，RNA-FM |Δ| 2.4e-6）。
本对照实际检验的是：差分监督形式在既有协议+多种子下能否兑现入档行之外的增益，并把 docstring/代码不一致登记为 protocol note
（结论读法见 §3——两臂不可分本身即有效结果）。

## 2. 执行对象与入档对照值

| 任务 | 族 | TRAIN n | VALIDATION n | 既有 frozen-Δ 入档 ρ（VALIDATION task_macro_spearman） |
|---|---|---|---|---|
| GSE114002 MRL | UTR-LM | 2,443 | 730 | 0.1107267878538859 |
| GSE114002 MRL | RNA-FM | 2,443 | 730 | 0.13693731732817227 |
| GSE269595 polyA | UTR-LM | 25,710 | 2,628 | 0.7490152634646079 |
| GSE269595 polyA | RNA-FM | 25,710 | 2,628 | 0.7114084807476264 |

入档值来源：MRL = development_hpo 两个 replay run 的 validation_evaluation.json；polyA = analysis_frozen_delta_full_coverage_20260904
（同口径 frozen-Δ 行主战场）。mRNABERT 外部对照行**非必需**，本预注册不纳入（如实记录，不硬凑）。

## 3. 判定规则（预注册，先于执行）

对每个 (模型族, 任务) 行计算 oracle ρ（VALIDATION，Task-1 评估器口径：任务内 Spearman on Δy 排序，K=10），与同族同任务既有 frozen-Δ 入档值对照。**判定优先级：先 (c) 后 (a)/(b)**（(c) 覆盖 (a)）：

- **(c) NO_SIGNAL**：oracle ρ ≤ 0.10 —— 表征不含 delta 信号（"表征差分 ≠ 性质差分"的直接证据）。
- **(a) HEAD/EQUIVALENT**（仅当 oracle ρ > 0.10）：oracle − frozenΔ ≤ +0.05 —— 差分监督无增益（瓶颈在表征，或两臂等价；结合 §1.1，等价是预期结果之一，其本身支持"监督形式不是差分有效性的增益变量"）。
- **(b) REPRESENTATION_SIGNAL**（仅当 oracle ρ > 0.10）：oracle − frozenΔ > +0.05 —— 表征含 delta 信息但绝对监督头未兑现（在 §1.1 事实下，此分支意味着既有行未达该管线可兑现水平，需回溯既有行协议差异）。

阈值理由：**+0.05 ≈ 项目 bootstrap CI 半宽量级**（差值小于 CI 半宽不作增益解读）；**ρ ≤ 0.10 ≈ 内靶 control 水平**（低于内靶对照的信号不作有效信号解读）。

## 4. probe 协议（与既有 LM probe 同款）

- 线性 head：UTR-LM 129 参数（128+1）/ RNA-FM 641 参数（640+1）。
- 特征 z-score（Δh 的 TRAIN 拟合池统计，std clamp 1e-6）；AdamW lr 1e-3 / wd 1e-4；100 full-batch epochs；
  source-group-equal 加权 MSE；epoch 选择 = 任务 VALIDATION source-group 加权 MSE（HPO_VALIDATION_ONLY，与既有行同款）。
- 多 seed：**3 seeds（20260816 / 20260902 / 20260919）**（probe 训练为秒-分钟级，全跑）；主读数 = 每 seed oracle ρ + 均值±范围；
  判定按主读数（均值）执行并登记 3 seed 一致性。若某行遇阻则降级单 seed 并在产物中声明。
- **分列声明**：oracle 行为 TRAIN-fit probe 口径；zero-shot frozen 行（frozen_delta zero-shot 类）与本臂分列，不覆盖既有行、不修改既有产物。

## 5. 纪律与遇阻登记

- protected TEST reads = 0；只用 TRAIN/VALIDATION；不修改 /mnt 既有产物；新产物仅写
  `experiments/analysis_delta_validity_v1/oracle_probe/`。
- CUDA 留证（cpu_fallback=false；沿用"禁止 CUDA_VISIBLE_DEVICES 重映射"的项目惯例，MIG 切片以 UUID 选择）。
- 若某族 embedding 提取遇阻（如 port 不含 embedding 钩子），该行登记 BLOCKED 原因，不硬凑。

## 6. 产物结构（规划）

`results_oracle_probe.json`：逐 (族,任务) 行 = oracle ρ（每 seed + 均值±范围）+ 既有 frozen-Δ 入档值 + 三分支判定 + n（TRAIN/VALIDATION）+ seed + CUDA 证据 + 协议元数据；汇总 md 一份。
