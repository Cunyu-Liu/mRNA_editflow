# V8 Stage 1 预注册：联合外部大库预微调（三臂 S / H / M）

- 预注册版本：`route2_v8_stage1_prereg_v1`（2026-09-04）
- 依据：SPECS_CRITIC_V6/spec.md「V8 攻坚线」增补节（已批准）；Phase 0 直调双臂终态结论
- 实现代码：本 worktree `core/route2_v8_hybrid_backbone_v1.py` / `core/route2_v8_joint_library_v1.py` / `scripts/route_a_v3/run_route2_v8_stage1_joint_prefinetune_v1.py`
- 纪律：本预注册冻结判定门与预算；冒烟（max-steps 小跑）不构成任何终态判定；无 peak-picking（H2 红线）；TEST 分区全程不碰，仅用 VALIDATION。

## 1. 背景与动机

Phase 0 直调双臂（`route_a_v3_mrnabert_directft_diag_20260904`）已终态：

- mRNABERT **原始预训练权重直调全面失败**：MRL −0.119 / MPRAU −0.091 / polyA 0.7901（arm A，raw-init LoRA 直调）。
  原始权重对 delta 判别是**负面资产**——Stage 1 预微调（外部大库监督）是必需品，不是可选项。
- 臂 B（冻结 embedding + 轻预测器）稳定优于臂 A（继续训 LoRA）——Stage 2 需保留"冻结表征"对照路径。

V8 Stage 1 的科学问题：**把多个外部大库（MRL 280K + polyA APA）联合预微调进同一 mRNABERT 主干，
能否形成不弱于单域预微调、且天然多域可迁移的联合先验？CNN motif stem（H 臂）是否额外有益？**

## 2. 三臂定义（同库 / 同预算 / 同 seed 20260903）

| 臂 | 架构 | 训练库 | 预算 | seed | 状态 |
|---|---|---|---|---|---|
| **S**（纯联合） | mRNABERT 12L×768d 原始 init + masked-mean-pool + 域条件 readout + linear head | MRL 280K + polyA APA（域均衡采样） | 见 §6 | 20260903 | 本 runner `--arch s` |
| **H**（混合） | S + Optimus 式 CNN motif stem（见 §3） | 同 S | 同 S | 20260903 | 本 runner `--arch h` |
| **M**（单域对照） | Route A 280K fullft_v2（MRL-only，全参数微调） | MRL 280K only | 6 ep × 677,608 行 = 31,752 步 | 20260903/04/05 | **已终态，复用引用，不重跑** |

M 臂引用数值（GSE114002 VALIDATION frozen-Δ task_macro_spearman，`experiments/analysis_fullft_v2_adjudication_20260903/`）：
单 seed 0.3198 / 0.2873 / 0.3157（seed 20260903/04/05），**3-seed 均值 0.3076（主对照）**，3-seed ensemble 0.3158（次级参考）。

polyA 域单域基线：2026-09-03 的 APA-only 预微调（`run_route2_mrnabert_apa_3p5m_prefinetune_v1.py`）仅存截断日志
（库加载/审计完成：2,740,320 行、flagged=0、target mean 0.318 / std 7.131），**无落盘 checkpoint/指标**。
→ 注册处置：polyA-only 基线由本同一 runner `--arch s --libraries polya` 补齐（配方与 S 完全一致，仅库不同），
在 Stage 1 正式发射时一并跑（作为第 4 个运行，不改变三臂判定结构）。

## 3. 架构细节（S 与 H）

- **主干加载**：与 Route A 参考完全一致（`AutoConfig` + `AutoModel.from_config` + 手工剥 `bert.` 前缀 +
  `flash_attn_qkvpacked_func = None`；`AutoModel.from_pretrained` 与自定义 ALiBi 栈不兼容）。
- **域条件注入（S/H 共用）**：域 one-hot embedding（3 域 × 768d，normal(0, 0.02) init）**加到 pooled 表征**
  （masked-mean-pool 之后、linear head 之前）。
  选择理由（预注册）：(a) 不扰动预训练编码器输入分布——Phase 0 已证 raw-init 主干脆弱；
  (b) 主干保持域无关，联合先验成型于共享主干，域 token 只选择读出方向；
  (c) S/H 代码路径完全一致，zero-shot 应用时域 id 恒已知。弃选方案：加到首 token（会改 CLS 输入分布，收益不明确）。
- **CNN motif stem（仅 H）**：Optimus 式两段 1D conv，核苷酸分辨率输入、逐位置输出：
  `one-hot[B,L,4] → Conv1d(4→96, k=8) → GELU → MaxPool(2) → Conv1d(96→128, k=6) → GELU → 最近邻上采样回 L →
  Linear(128→512) → GELU → Linear(512→768)`，**共 537,056 参数（≈0.5M 预算内）**。
  长度精确保持（TF-SAME 非对称补零）；非核苷酸位置（CLS/SEP/PAD/UNK/N）stem 输出严格置零。
- **融合方式（预注册声明：加法残差注入）**：stem 输出投影到 768d 后**与 word embedding 逐位置相加**，
  再走标准 `BertEmbeddings` 后处理（token-type 相加 + LayerNorm + dropout，经 `inputs_embeds` 入口）。
  弃选方案：concat 后投影（改变编码器输入宽度，需改动钉死的 bert_layers 实现，可靠性差）。
- **零初始化投影**：stem 末层 Linear 权重/偏置零初始化 ⇒ **H 在 step 0 与 S 函数完全等价**
  （单测 `test_arm_h_equals_arm_s_at_init` 固定该性质）。stem 必须在训练中"挣得"贡献，消除初始化混杂。
- 读出头：masked-mean-pool（+域 embedding）→ Linear(768→1)。

## 4. 联合数据管线（`core/route2_v8_joint_library_v1.py`）

统一记录格式：`(sequence [+cell_context_id] → standardized activity, domain_label, 泄漏 flag)`。

| 域 | domain_id | 数据 | 行数（clean） | 活性标准化 |
|---|---|---|---|---|
| mrl | 0 | Sample-2019 280K（两重复按 UTR 合并取均值 rl） | 677,608 | z-score（clean 库上） |
| polya | 1 | APARENT APA GSE113849 isoform 表（total_count_vs_distal ≥ 10；p clip 1e-4；log2 odds） | 2,740,320 | z-score（clean 库上） |
| cms | 2 | **STUB**：ENCODE CMS array 未下载（等用户浏览器中转）；接口/CSV schema（`sequence,activity[,cell_context]`）与 domain 槽位已冻结，数据落盘即插入，不阻塞其余域 | — | — |

- **泄漏审计 flag 随行**：每行携带 `flag_vs_gse114002`、`flag_vs_gse269595`，规则与 W0 参考脚本逐一相同
  （3-block pigeonhole：17bp 精确块碰撞 → 逐位比较 ≤2 mismatch 判 flag）：
  GSE114002 用连续三分块（覆盖整条 50-mer），GSE269595 用首/中/尾块；**两套审计对全部库行交叉执行**
  （跨域保护）。任一 flag 为真的行不入训练（MRL 280K 已审计 flagged=0；polyA 亦为 0——均对各自 benchmark；
  交叉审计结果以运行时 `leakage_audit.json` 落盘为准）。
- **采样**：域均衡——每个 batch 按活跃域均分配额（batch=128、2 域 → 64/64；余数按 batch 序轮转防偏置），
  每域独立洗牌循环。一个 epoch := ceil(联合库总行数 / batch) 步。
- tokenizer：mRNABERT 词表逐核苷酸（A/T/C/G/N = id 5–9，启动时 `verify_vocab_alignment` 强校验）；
  polyA 186bp → 188 token（截断上限 512 不触发）。

## 5. 训练配方（沿用 280K fullft_v2 预微调配方）

全参数微调；MSE（域内 z-scored 活性）；AdamW lr 2e-5、weight decay 1e-4；batch 128；
cosine 退火至 10%（5% 线性 warmup）；BF16 autocast；seed 20260903（对齐 Route A / M 臂主 seed）；
默认 epochs 2（`--epochs 2`，按 V8 攻坚线注册默认）。每 epoch 存 checkpoint
（`stage1_{arch}_epoch{N}.pt`，含域标准化常数与审计摘要）+ 分域 loss 曲线
（`training_losses.jsonl` 每 50 步、`epoch_domain_loss.jsonl` 每 epoch）。

## 6. 预算核算（诚实对账，注册为最终口径）

S/H 默认（epochs 2、batch 128、库 mrl+polya，clean 计）：

- 每 epoch 步数 = ceil((677,608 + 2,740,320)/128) = **26,703**；两 epoch 总更新 **53,406 步**。
- 域内曝光（每 epoch 每域 1,709,792 行抽取）：MRL 侧 5.04 遍 / polyA 侧 1.25 遍；两 epoch 即 MRL 5.04 遍。
- 对照 M 臂：31,752 步、MRL 侧 6.00 遍。

**不等价声明**：联合库下无法与单域臂做到预算严格相等。注册口径：总更新数 S/H 比 M 多 68%（+21,654 步），
MRL 侧曝光少 16%（5.04 vs 6.00 遍）。若判定门争议落在该差额内，按 §8 的平局规则处置，不追加预算。
polyA-only 基线臂：epochs 6（与 M 臂同更新数量级：ceil(2,740,320/128)×6 ≈ 128,454 步；
polyA 域单域对照以"同配方同库全预算"为口径，不与联合臂逐位对齐）。

## 7. zero-shot 评估协议（与 Route A 各域 benchmark VALIDATION 同口径）

- 主判据：**FINAL-EPOCH-FIXED**（末 epoch 固定，无逐 epoch 择优；逐 epoch 指标仅诊断落盘）。
- MRL 域：GSE114002 **VALIDATION**（730 条）；frozen-Δ：prediction = f(candidate, domain=mrl) − f(source, domain=mrl)；
  冻结 Task-1 评估器 K=10（task_macro_spearman / top_1 / ndcg@10）。
- polyA 域：GSE269595 **VALIDATION**（2,628 条）；同 frozen-Δ、同评估器、domain=polya。
- 评分 BF16 autocast、batch 256、model.eval()。**TEST 分区不碰。**

## 8. 判定门（Stage 1 → Stage 2 发射条件）

1. **非破坏门（每域）**：联合臂 zero-shot ≥ 单域臂 × 90%。
   - MRL 域：S 与 H 各自 ≥ 0.9 × 0.3076 = **0.2768**（M 臂 3-seed 均值口径；ensemble 0.3158 仅作参考记录）。
   - polyA 域：S 与 H 各自 ≥ 0.9 × polyA-only 基线（§2 补齐运行）。
2. **S vs H 裁决**：MRL 域 zero-shot task_macro_spearman 上，**若 S ≥ H − 0.02 则选 S**（简单优先，
   0.02 为注册的实质差异容忍带）；否则选 H。polyA 域差异作为次级证据记录，不单独裁决。
3. **Stage 1 成功定义**：通过非破坏门的臂 ≥ 1 个且完成 S/H 裁决 → 该臂晋级 Stage 2（任务域 delta 微调 +
   冻结 embedding 对照复用 Phase 0 臂 B 结论）。
4. **失败处置**：若 S 与 H 均未过非破坏门 → 判"联合先验注入失败"，Stage 2 不发射；V8 攻坚线回退至
   单域预微调 + 逐域路由（记录归档，不追加预算重试）。
5. 平局/边界处置：任何判据落在注册差额带（预算不等价 §6 或 0.02 容忍带）内 → 按上面保守规则
   （选 S / 判失败）处置，不追加 seed 重跑；3-seed 扩展仅作为未注册的探索性选项留档。

## 9. 冒烟协议（非终态）

GPU5（不触碰 GPU1-4 进程）；S/H 各 `--max-steps 200`（scheduler 以 200 步为总步数缩放）；
验证全链：库加载 → 交叉泄漏审计 → 域均衡采样 → 前向/loss/反传 → checkpoint 落盘 → 分域 loss 曲线 →
zero-shot 评估路径。冒烟产物标记 `smoke`（non-terminal），不进入任何判定。

## 10. 已知限制

- polyA 单域基线待补齐（§2）；CMS 域 stub 待 ENCODE 数据落盘（§4）；
- 预算不等价（§6，已注册口径）；
- S/H 首发单 seed（20260903），与 M 臂主 seed 对齐；seed 扩展为未注册选项；
- Saluki TFRecords（P2 降解域）未纳入 Stage 1（Zenodo 中转暂缓）。
