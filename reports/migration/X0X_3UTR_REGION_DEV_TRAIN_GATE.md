# X0-X 3′UTR RegionAdapter Development Training — mRNA-XEditFlow v1.1

- **Migration Goal:** `GOAL-XEDITFLOW-MIGRATION-01`
- **Prior terminal state:** `BLOCKED_WITH_EVIDENCE`（sealed-final GSE246381 保留，CDS-B1 序列阻塞）
- **Phase:** X0-X — 3′UTR transfer（**PURE DEVELOPMENT TRAINING ONLY**；formal X0-X gate NOT triggered）
- **Gate outcome:** `X0X_3UTR_REGION_DEV_TRAINED`（development-only；无 measured cross-region transfer claim，无 sealed 访问，冻结 5′ 模型未触碰）
- **UTC:** 2026-08-07
- **Worktree branch:** `xeditflow-migration-20260806T024650Z`
- **New module:** `scripts/x0x/train_3u_region.py`

---

## 1. FACTS_FROM_REPO
- 迁移终态为 `BLOCKED_WITH_EVIDENCE`；用户选择"先做无争议开发再定"（sealed final 暂不消费）。
- 3U-A1 在 `artifacts/b0x/effect_dataset.jsonl` 中：总 44,347 行，其中 **42,962 行 delta-defined**（1,385 行 `delta=None`/`source_anchor_unavailable`），7 个 study（ENCSR854RUF/GSE186455/GSE200304/GSE232571/GSE232572/GSE261709/GSE298114），7 个 3′ endpoint。
- 每个 3U-A1 source 仅一个 measured candidate → **无 measured search neighborhood**，故本开发训练不计算 ranking headline，只做 3′UTR delta-effect transfer 评测。
- `scripts/x0x/region.py` 提供 `RegionAdapter`（5′/3′ 独立 effect heads + 3′ 专属 endpoint embedding，架构级隔离）。
- `scripts/m4_sparse/` 提供编码约定（ACGU / MAX_SEQ_LEN=100 / 12-dim edit features）与 `EffectDataset`。

## 2. FACTS_FROM_CONTRACTS
- 合同 §16（X0-X）：3′UTR 独立 endpoint heads、3′ mechanism adapter/coupling、study/context transfer、不混 5′ MRL 与 3′ stability；"只有 5′primary 模型、threshold 和 sealed 结果冻结后执行" **formal gate**。
- 合同 §16：不得用 secondary 结果反向调整 5′ sealed 模型 → 本开发训练只训 3′ RegionAdapter，不触碰 F0-X base flow / M4 critic。
- 合同 §10.2 effect metrics（Spearman / sign accuracy / top-decile enrichment）；§5.2 数据角色表（3U 各数据集为 A1/A2 transfer）。

## 3. INFERENCES
- 因为 formal X0-X gate 被 sealed 结果冻结阻塞，正确的推进方式是**纯开发训练**：在 3U-A1 上训练 `RegionAdapter`（backbone + 3′ adapter），验证：(a) 独立 endpoint head 结构约束成立；(b) 3′UTR delta-effect 的 S4 leave-one-study-out transfer 表现可被诚实记录。**不**据此声称正式跨区域迁移结果。
- 从序列模型在 S4 transfer 下接近机会水平是 B0-X/M4 已确立的模式；3U-A1 开发结果与之一致，属预期内发现，不是工程失败。
- `GSE298114` study 全部 delta 为正（pos_frac=1.000），单符号 study 下 sign_acc 无判别信息（0.0），故单列并排除于可比较 macro。

## 4. UNKNOWN_OR_BLOCKED
- **Formal X0-X transfer gate:** BLOCKED（需 frozen sealed 结果，E0-X 决策保留 GSE246381）。
- **Measured CDS B1:** DORMANT_BLOCKED_ON_SEQUENCE（GSE207584 无 per-variant 同义序列）。
- **真实跨区域迁移增益（5′→3′）:** 未评估（formal gate 未触发；本训练只建 3′ 开发底座）。

## 5. FILES_READ
- `scripts/x0x/region.py`、`scripts/x0x/codon.py`、`scripts/m4_sparse/{model,dataset,config,train}.py`、`scripts/b0x/build_effect_dataset.py`、`scripts/b0x/run_effect_baselines.py`、`artifacts/b0x/effect_dataset.jsonl`、`reports/migration/FINAL_MIGRATION_REPORT.md`。

## 6. FILES_CHANGED
- `scripts/x0x/train_3u_region.py`（new）：3U-A1 开发训练脚本。`SparseBackbone`（复用 M4 编码器结构，输出 context z）+ `RegionModel`（backbone + RegionAdapter 组合）+ `RegionAdapter`（3′ effect heads）。S4 leave-one-study-out over 7 3U studies；仅用 delta-defined 行；评测 Spearman / sign_acc / top-10% enrichment；输出 per-fold `.pt` + 结果 JSON。
- `artifacts/x0x/region_3u_dev/`（new）：7 个 fold checkpoint + `region_3u_dev_results.json`。

## 7. COMMANDS_RUN
- 服务器 editflow env（GPU `cuda:3`，非 owned GPU 4）：
  ```
  python scripts/x0x/train_3u_region.py --gpu cuda:3 --out artifacts/x0x/region_3u_dev
  ```
- 修复过程：初始版本误含 1,385 个 `delta=None` 行 → `EffectDataset` 产生 NaN 标签 → `dev_mse=Infinity`、`sign_acc=0`。修复为只保留 delta-defined 行后重跑，数值正常。

## 8. TEST_RESULTS
- 迁移全量 `tests/migration/`：**224/224 PASS**（未回归；本开发训练不新增单测文件，区域适配器结构约束已由 `test_x0x.py` 覆盖）。
- 开发训练自检：`independent_heads_all=True`（7/7 fold），`dev_mse` 有限（0.06–0.28），7/7 fold checkpoint 生成。

## 9. DATA_COUNTS_AND_DENOMINATORS
- 3U-A1 delta-defined: **42,962 行 / 7 studies**（每个 fold train≈35.8–37.4k，test=held-out study）。
- 每 fold n_test：ENCSR854RUF 11,934 · GSE186455 621 · GSE200304 5,826 · GSE232571 14,092 · GSE232572 9,066 · GSE261709 749 · GSE298114 365。
- **无 measured ranking headline**（每 source 单 candidate）；仅 delta-effect transfer。

## 10. REUSE_DECISIONS
- M4 编码器结构（stem/self-attn/cross-attn/edit MLP/context）：`REUSE_WITH_ADAPTER`（SparseBackbone 复用结构，输出 context z，未复用冻结 effect heads）。
- `scripts/x0x/region.py` RegionAdapter：`REUSE_AS_IS`（独立 3′ heads）。
- `scripts/m4_sparse/dataset.py` EffectDataset/编码：`REUSE_AS_IS`。
- 3′UTR 机制 adapter：`REBUILD`（开发底座，非旧 alignment 主线）。

## 11. GATE_STATUS
- `X0X_3UTR_REGION_DEV_TRAINED` — **development-only**。formal X0-X transfer gate NOT triggered（blocked on frozen sealed results）。无 measured 跨区域 transfer claim。

## 12. CLAIMS_UNLOCKED
- 3′UTR `RegionAdapter` 可训练于 3U-A1（S4，独立 3′ heads），结构约束 7/7 成立（`predicted/internal proxy` 限定的开发结果）。
- 诚实记录：3′UTR 从序列 delta effect 在 S4 transfer 下接近机会水平（macro Spearman 0.029，macro sign_acc 0.499 剔除单符号 study 后），与 B0-X/M4 的 5′UTR 模式一致。

## 13. CLAIMS_STILL_PROHIBITED
- 任何 5′→3′/CDS 跨区域 transfer 结果（formal gate 未触发）。
- 任何 measured CDS B1 claim（DORMANT_BLOCKED_ON_SEQUENCE）。
- 任何 GSE246381 sealed label 访问（保留，一次性未消耗）。
- 任何 "improves TE/stability/expression" 无 `predicted/internal proxy` qualifier。

## 14. NEXT_PHASE_INPUTS
- Formal X0-X transfer gate 仍 blocked on frozen sealed 结果（用户决策是否消费 GSE246381 一次性访问）。
- CDS-B1 进入 B1 需外部 per-variant 同义序列表。
- 3′UTR adapter 已在 3U-A1 上完成开发训练，formal gate 解除后可接正式跨区域评测。

## 15. COMMIT_SHA
- 见分支 `xeditflow-migration-20260806T024650Z` git log（本开发训练代码 + 产物提交）。

## 16. MANIFEST_AND_HASHES
- `scripts/x0x/train_3u_region.py`（new）。
- `artifacts/x0x/region_3u_dev/region_3u_dev_results.json` + 7 fold `.pt`（保留于 worktree；`artifacts/` 按仓库约定不入 git）。
