# Gate P 揭盲预注册 v1（2026-10-07 · 执行前冻结）

- **change-id**: gate-p-test-unblinding-prereg-v1-20261007
- **状态**: FROZEN（本文件落盘即冻结；执行触发 = 用户 2026-10-07 指令「一次性读取 18,292 行 TEST → receipt → 回填 → 一致性核对」）
- **上游**: Gate P 决策（用户 2026-10-06 拍板「全文 v2 完成后揭盲」）+ W2 全部完成（journal 140：接线/样式/availability/references 全绿）+ gate_p_precheck_v1.md v1.2。

## 1. 揭盲范围（一次性，不可重复）

- **表面**: `route2_development_frozen_v1/development_manifest.jsonl` 中 split=TEST 的 **18,292 行**（9 个 Development 任务的分层：MPRAU 12,048 / polyA 2,628 / TE200304 1,615 / MRL 726 / HL 503+402 / REFALT 274 / PLUMAGE-TE 48 / PLUMAGE-RNA 48）。
- **一次性读取语义**: 本次读取后，TEST 不得再用于任何调参/选型/阈值决策（history-dependent decisions prohibited，gate_p_precheck §2 条款原文）。揭盲输出 = append-only。

## 2. 评测对象与口径（冻结）

- **模型（1 个，headline）**: **冻结 V5 主行 checkpoint**（`xeditcritic_v5/v5_screen_seed_20260907_runner_1113cd2c.../v5_full/final_pass_8_checkpoint.pt`，即 polyA 0.8219 主行血统；MULTI-TASK V4-FULL，state_dict + vocabs + target_scaler + capacity 全部从 checkpoint 原样加载）。
  - **为什么只评 V5**：TEST 揭盲的目的是验证「主结果（polyA 91.3% 完成度 + 家族领先）在 held-out TEST 上成立」——这是论文的主张所在。矩阵 5 移植族/11 探针骨干是 VALIDATION 表象刻画（frozen-Δ 描述性），其 TEST 外推不在本门范围内（矩阵 prereg 从未注册 TEST 评测）；把范围扩到 14 族会把一次性揭盲变成多模型竞赛，违反「揭盲后不得选型」纪律。
- **推理口径（与 VALIDATION 主行完全同款）**: checkpoint 的 model_state_dict 原样重建 `_build_model_v4_full` → vocabs/target_scaler 从 checkpoint dict 原样恢复（TRAIN 拟合的 scaler 不重拟合）→ bottom-six chunk cache 对 TEST 序列**在线构建**（`FrozenMRNABERTBottomSixEncoderV4.encode_sequences`，与训练同 chunk 语义）→ `XEditCriticDatasetV4` + `XEditCriticCollatorV4` + `_forward_bf16`（BF16 autocast，A100 硬门）→ `v4_evaluate` 同款循环（`evaluation_index_batches_v4`）。
- **指标**: `validation_metrics` 同款（task-macro Spearman 为主 + 逐任务表 + standardized MAE）；**TEST 列直接进 bottomline 主表**（与 VALIDATION 列并排，不替换 VALIDATION 列）。
- **标签读取（一次性）**: canonical `direction_normalized_delta` 字段按 manifest TEST record_id 集合读取（这就是「揭盲」动作本身——读取即计数 1 次，receipt 记录）。

## 3. 与 VALIDATION 一致性核对规则（冻结，零事后裁量）

| 核对项 | VALIDATION 基准 | 一致性判据 |
|---|---|---|
| C1 polyA 主张 | V5 VAL Spearman 0.8219（91.3% 完成度） | TEST polyA Spearman 落在 [0.8219 − 0.05, 0.8219 + 0.05] → **双集确认**（polyA 主张升级为 TEST-confirmed）；区间外 → 如实报告偏离（写 limitation，不改 VALIDATION 主张） |
| C2 家族排序 | polyA 为 V5 最强任务（0.8219 vs 其余 ≤0.32） | TEST 上 polyA 仍为 V5 最强任务 → 排序确认；否则如实报告 |
| C3 任务带结构 | 低密度任务带（HL≈0 / TE≈0.06 / MRL≈0.13-0.20） | TEST Spearman 与 VAL 差 |Δ|≤0.08 的任务 ≥6/9 → 带结构确认；否则逐任务如实报告 |
| C4 整体 | macro 0.167 | TEST macro 与 VAL macro 差 |Δ|≤0.05 → 宏观一致 |

- **所有判据的方向均为「确认 or 如实报告偏离」，没有 FAIL 重训分支**——模型已冻结，揭盲是验证不是竞赛。
- 双集一致 → 论文相应主张加注 "TEST-confirmed (one-shot unblinding, receipt on file)"；不一致 → 写入 §9.3 limitation + 主结果保持 VALIDATION 口径（读者看到两列）。

## 4. Receipt schema（append-only，落盘即不可改）

`unblinding_receipt_v1.json`（输出目录内）:
```
{
  "schema_version": "route_a_v3_gate_p_unblinding_receipt.v1",
  "executed_utc": <ISO8601>,
  "test_record_count": 18292,
  "test_manifest_sha256": <development_manifest.jsonl 的 sha256>,
  "test_record_id_set_sha256": <18,292 个 canonical_record_id 排序后 join 的 sha256>,
  "test_label_read_events": 1,          // 一次性语义的机器计数
  "models_evaluated": ["V5_frozen_main_row"],
  "checkpoint_sha256": <final_pass_8_checkpoint.pt 的 sha256>,
  "predictions_path": <TEST predictions.jsonl>,
  "metrics_path": <unblinding_metrics_v1.json>,
  "prereg": "docs/paper/gate_p_unblinding_prereg_v1.md",
  "prior_test_reads_before_this_event": 0,
  "gate_p_decision": "user 2026-10-06: after full-paper v2 (satisfied 2026-10-07, journal 140)"
}
```

## 5. 执行顺序（原子性）

1. 冻结本 prereg（本 commit）→ 2. 构建 TEST cache（outcome-free：只编码序列，不读标签）→ 3. 模型推理（零标签接触）写 predictions.jsonl → 4. **一次性读取 TEST 标签**（manifest id 集合 ∩ canonical）+ 评测 + receipt 落盘 → 5. 一致性核对表输出 → 6. 主结果表 TEST 列回填（论文 + 底账）。第 4 步之前任何中断不构成揭盲（标签未读）。

## 6. 禁止事项

- 揭盲后不得用 TEST 读数调任何东西（含措辞选择性加强）；
- 不得重跑（若推理崩溃在标签读取前，可重启——标签未读；若崩溃在读取后，用已写 predictions 完成评测，不得重新推理）；
- 5 移植族/探针骨干的 TEST 评测**不在本门**（未预注册）；如未来需要，须新 amendment 且承认 TEST 已被 V5 读取过的事实。
