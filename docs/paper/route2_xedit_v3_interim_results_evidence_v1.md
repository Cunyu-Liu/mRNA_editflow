# XEdit V3 interim Development evidence

Status date: 2026-08-23. This note is an observed-results companion to the prospectively frozen protocol. It does not amend the architecture, seeds, thresholds, task set, baselines, selection order, or authorization boundary in the protocol. Development TEST and new final Evaluation outcomes remain unread.

## Interpretation boundary

The Critic screen is still incomplete, whereas the SetFlow screen is terminal NO-GO. C0 and C1 are not selectable Critic arms, and C2 plus the remaining preregistered critic controls are incomplete. For SetFlow, F0 is a read-only historical reference and F1 is an objective diagnostic that cannot be selected. Both selectable arms have completed: F2 failed its frozen arm gate on unique-candidate rate, and F3 failed recovery, top-k recovery, and unique-candidate rate. Neither arm can enter confirmation.

## Critic diagnostic evidence

| Arm | Status | Task-macro Spearman | Margin over C0 | Standardized MAE | Positive tasks | Frozen interpretation |
|---|---|---:|---:|---:|---:|---|
| C0 endpoint-aware raw CNN | Terminal | 0.1108180590 | 0 | 1.9924297611 | 8/9 | Matched same-information baseline only. Ranking is weak and MAE exceeds 1.70. |
| C1 C0 + global mRNABERT mean residual | Terminal | 0.1386460633 | +0.0278280043 | 1.9004665151 | 8/9 | Global residual gives a small diagnostic improvement, but C1 is nonselectable and misses the 0.25 Spearman and 1.70 MAE thresholds. No retraining is authorized. |
| C2 frozen edit-site token critic | Full arm terminal; controls running | 0.1042656112 | -0.0065524478 | 1.9705208102 | 7/9 | The full arm fails Spearman, margin, MAE and positive-task thresholds and cannot become eligible. Preregistered controls continue to terminal for complete evidence. |

C1 therefore supports only the narrow observation that the legacy global pretrained residual contains some Development signal. C2 supplies no edit-site ranking advantage: it underperforms C0 by 0.0066 Spearman, exceeds the 1.70 MAE ceiling, and has only seven positive tasks. Its full model does beat the source-only control, whose task-macro Spearman is -0.0013, but this cannot rescue the failed primary criteria. C2 is ineligible; C3 remains the only architecture that could pass the Critic screen after current-HEAD synchronization.

Primary repository evidence: `audits/route_a_v3_route2_xeditcritic_v3_c1_terminal_c2_launch_v1.json` and `audits/route_a_v3_route2_xeditcritic_v3_c2_full_source_terminal_controls_running_v1.json`. The authoritative full artifacts remain under the Route 2 `/mnt` experiment root and the central attempt ledger.

## SetFlow diagnostic evidence

| Arm | Status | Common set-NLL | Relative NLL improvement over F0 | Recovery | Top-k recovery | Unique rate | Correctness | Frozen interpretation |
|---|---|---:|---:|---:|---:|---:|---|---|
| F0 terminal Base Flow V2 replay | Frozen reference | 5.3979076352 | 0 | 0.203 | 0.098 | 0.883 | Historical G0 ready | Read-only comparator; not retrained. |
| F1 small trunk + set-marginal objective | Terminal diagnostic | 5.4724267445 | -0.0138051842 | 0.2691732136 | 0.2131454877 | 0.4192269921 | Legality 1.0; budget, replay and numerical failures 0 | Recovery and top-k improve, but NLL worsens and diversity collapses. F1 is nonselectable and terminal. |
| F2 eight-block hybrid, width 384 | Terminal; arm gate failed | 2.0680908164 | +0.6168717666 | 0.2924616536 | 0.1682782203 | 0.6793630752 | Legality 1.0; budget, replay and numerical failures 0 | NLL, recovery, top-k and G0 checks pass, but unique rate misses 0.90. F2 is terminal and cannot enter confirmation. |
| F3 twelve-block hybrid, width 512 | Terminal; arm gate failed | 2.0504294127 | +0.6201436647 | 0.1939768051 | 0.1048712807 | 0.6374508979 | Legality 1.0; budget, replay and numerical failures 0 | NLL and G0 checks pass, but recovery, top-k and unique rate all miss their frozen thresholds. F3 cannot enter confirmation. |

F1 supplies useful mechanism evidence: an order-invariant set target can improve measured-candidate recovery even when the small legacy trunk remains severely mode-concentrated. F2 shows that the larger hybrid trunk can strongly improve common set-NLL and clear the recovery/top-k thresholds, but its source-macro unique rate remains only 0.6794. F3 training selected pass 3 after five completed passes and reached a common set-NLL of 2.0504, a 62.01% improvement over F0, but its generation recovery fell to 0.1940 and its unique rate to 0.6375. Thus increased capacity improved likelihood without producing the frozen recovery/diversity profile. Because neither selectable arm passed, the SetFlow V3 screen is terminal `XEDITSETFLOW_V3_SCREEN_NO_GO`; no confirmation seed, retraining, threshold reduction, or architecture substitution is authorized under the frozen protocol.

An outcome-free post-terminal decomposition shows that F2's diversity failure is not confined to the one-edit budget. Mean source-level unique rates for B1, B3, and B5 are 0.4832, 0.7134, and 0.8066, respectively; the fractions of sources reaching 0.90 are 0%, 22.0%, and 35.8%. B1 is the most concentrated regime, but every edit budget remains below the frozen diversity target. The appropriate interpretation is therefore that F2 improves distributional fit and measured recovery without resolving candidate concentration across the full budget range.

The same outcome-free diagnostic by evaluable study/endpoint domain gives mean unique rates of 0.4359 for GSE269595/poly(A)-usage, 0.6581 for GSE114002/mean-ribosome-load, 0.7063 for ENCSR854RUF/allelic-skew, and 0.8221 for GSE217518/RNA-half-life. Every domain remains below 0.90, so the failure is not attributable to a single study. Study and endpoint are one-to-one confounded in this validation cohort and must not be interpreted as independently estimated endpoint-semantic effects. GSE114002 contributes 652 of 891 sources, so the overall source-macro value is also composition-sensitive; domain-resolved values should accompany the aggregate benchmark result.

Primary repository evidence: `audits/route_a_v3_route2_xeditsetflow_v3_f1_terminal_diagnostic_v1.json`, `audits/route_a_v3_route2_xeditsetflow_v3_f1_training_terminal_20260823_075027.json`, `audits/route_a_v3_route2_xeditsetflow_v3_f2_terminal_f3_health_v1.json`, `audits/route_a_v3_route2_xeditsetflow_v3_f2_diversity_diagnostic_v1.json`, `audits/route_a_v3_route2_xeditsetflow_v3_f2_diversity_domain_diagnostic_v1.json`, `audits/route_a_v3_route2_xeditsetflow_v3_f3_training_terminal_validation_launch_v1.json`, `audits/route_a_v3_route2_xeditsetflow_v3_f3_terminal_screen_no_go_v1.json`, and `docs/execution/route_a_v3_route2_training_attempt_table_20260817.md`.

## Current claim state

No V3 model-performance claim is established. The project remains not ready for submission, critic guidance remains unauthorized, and the external Evaluation remains locked. A strong critic claim still requires a selectable C2/C3 screen pass, exact three-seed confirmation, atomic Development TEST, all-Development refit, and LOSO readiness. The frozen SetFlow method-repair round cannot support the requested generator claim because no selectable base-flow arm passed screen; consequently three-seed Flow confirmation and soft-value SMC are not authorized in this round.

This interim evidence should be replaced by a terminal screen table only after every preregistered screen artifact required by its gate is present. Failed results remain terminal evidence; thresholds, tasks, seeds, and baselines are not changed in response to these observations.
