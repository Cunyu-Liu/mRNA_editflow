# E0-X sign_accuracy Selective-Prediction Exploration — mRNA-XEditFlow v1.1

- **Migration Goal:** `GOAL-XEDITFLOW-MIGRATION-01`
- **Prior terminal state:** `BLOCKED_WITH_EVIDENCE`
- **Phase:** E0-X — effect-gate sign_accuracy 修复探索（**DEV ONLY；sealed 未访问；冻结阈值未改**）
- **Outcome:** `E0X_SIGN_AMENDMENT_EXPLORED_AND_REJECTED`
- **UTC:** 2026-08-07
- **Worktree branch:** `xeditflow-migration-20260806T024650Z`

---

## 1. 目的
ordinary internal test（`E0X_PREREG_20260807`）因 `macro_sign_accuracy 0.510 < 0.60` 判 NO_GO。用户要求避免 NO_GO，并询问能否以合同 §T5 的**选择性预测（abstain + coverage-risk）**做 pre-unblinding 修订，合法提升 sign accuracy，从而在不浪费一次性 GSE246381 sealed final 的前提下转 GO。

## 2. 方法（DEV ONLY，诚实验证）
- 复用**冻结 delta critic**（`artifacts/m4_sparse/delta/model_5U-A1__*.pt`）、**同一 S4 5U-A1 folds**、**同一 test rows**，与 ordinary internal test 完全一致。
- 实现 [selective_sign_dev.py](file:///Users/liucunyu/Documents/all_code/ZJU/mRNA_editflow/scripts/e0x/selective_sign_dev.py)：对两种诚实置信度源做 sign_acc×coverage frontier：
  1. **预测方差** `exp(logvar)`（heteroscedastic head 输出）——低方差 = 高置信；
  2. **预测幅度** `|pred|`——高幅度 = 高置信。
- coverage grid：{0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 1.0}。
- 无 sealed 访问；无阈值修改；无 GO/NO-GO verdict（仅证据收集）。

## 3. 结果
| 置信度源 | Spearman(置信度, 符号正确性) | frontier 结论 |
|---|---|---|
| logvar（方差） | GSE114002: 0.0155 · GSE217518: 0.0038 | 无判别力 |
| \|pred\|（幅度） | GSE114002: −0.021 · GSE217518: −0.010 | 无判别力 |

- **selective sign_accuracy 在所有 coverage 下被封顶 ~0.50–0.51**（远低于 0.60），且 coverage 越低并未单调上升（cov=0.25 反而略低于 cov=1.0 部分情形）。
- **根因**：冻结 critic 对"符号是否正确"未学到任何可用的置信度信号——与该问题从序列模型的类别先验封顶（~0.52，已由重训证明）一致。**选择性预测在现有模型上不是合法可行路径。**

## 4. 决策（用户确认）
**不接受修订、接受 NO_GO。** 具体：
- 不修订预注册（sign_accuracy 保持 primary gate 指标，0.60 阈值不改）。
- 接受 E0-X ordinary internal test NO_GO 为诚实结果。
- 保持终态 `BLOCKED_WITH_EVIDENCE`。
- **不消费一次性 GSE246381 sealed final**（避免在已知 NO_GO 的冻结协议上浪费机会）。
- 无伪造 PASS；无事后调阈值。

## 5. 审计记录
- 决策已追加至 `docs/execution/xeditflow_migration_decision_log.yaml` → **XE-DEC-008**，status=`EXPLORED_AND_REJECTED_AMENDMENT_20260807`。
- 脚本 `scripts/e0x/selective_sign_dev.py` + 结果 `artifacts/e0x/selective_dev/e0x_selective_sign_frontier.json` 保留为证据。

## 6. 下一步
- 终态保持 `BLOCKED_WITH_EVIDENCE`。
- 若未来出现具备 sign-置信度判别力的更强模型/新数据，可在那时重新评估 selective 修订（仍须 pre-unblinding）。
- GSE246381 sealed final 一次性访问继续保留，由用户决定是否在更有利条件下消费。
