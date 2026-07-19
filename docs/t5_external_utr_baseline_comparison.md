# T5 External 5'UTR Baseline Comparison

- Claim policy: This artifact supports a descriptive T5 5'UTR comparison under a shared offline proxy oracle. The MEF local-search row is oracle-guided rather than model-only, and UTRGAN candidates are de novo rather than source-conditioned. Do not use this table to claim MEF or UTRGAN superiority.
- Descriptive table ready: `True`; model-only head-to-head ready: `False`; MEF superiority claim ready: `False`
- Paired per-record inference ready: `False`. UTRGAN generates an unconditional batch and the adapter assigns candidates to ordered source rows; this is not source-conditioned pairing. Per-row deltas are descriptive only.

| Method | Status | n | TE | delta TE | uAUG | Kozak | Access | UTR edit | UTR length delta | exact native | unique | CDS fixed | 3UTR fixed | protein exact-1 | seed p | signal | sec/seq |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| native_source | `measured_reference` | 1024 | 0.791036 | 0.000000 | 0.547852 | 0.601562 | 0.579590 | 0.000000 | 0.000000 | 1.000000 | 0.779297 | 1.000000 | 1.000000 | 1.000000 | NA | NA | NA |
| MEF_full_length_mo_pareto_top64 | `measured_internal_model_context` | 1024 | 0.800309 | 0.009273 | 0.511719 | 0.597607 | 0.633556 | NA | NA | NA | NA | 1.000000 | NA | 1.000000 | NA | NA | NA |
| MEF_region_adapter_utr5only_top64 | `measured_internal_model_10seed_utr5only` | 1024 | 0.786258 | -0.004779 | 0.551855 | 0.592041 | 0.618237 | 2.787109 | 0.000000 | 0.046289 | 0.999023 | 1.000000 | 1.000000 | 1.000000 | 0.004498 | significant_negative | NA |
| MEF_pure_utr_teacher_utr5only_top64 | `measured_internal_model_10seed_utr5only` | 1024 | 0.794220 | 0.003184 | 0.575293 | 0.596875 | 0.628355 | 2.796289 | 0.000000 | 0.046582 | 0.999023 | 1.000000 | 1.000000 | 1.000000 | 0.004498 | significant_positive | NA |
| MEF_full_then_utr_teacher_utr5only_top64 | `measured_internal_model_10seed_utr5only` | 1024 | 0.794013 | 0.002976 | 0.566406 | 0.599463 | 0.628053 | 2.747559 | 0.000000 | 0.046387 | 0.999023 | 1.000000 | 1.000000 | 1.000000 | 0.004498 | significant_positive | NA |
| MEF_pure_utr_teacher_budget5_utailor_strict_25_100nt | `measured_internal_model_10seed_utailor_protocol_subset_budget5` | 315 | 0.840999 | 0.007774 | 0.292063 | 0.633016 | 0.612496 | 4.796190 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.004498 | significant_positive | NA |
| MEF_utr5_constrained_local_search_budget3 | `measured_internal_oracle_guided_ceiling` | 1024 | 0.874059 | 0.083022 | 0.036133 | 0.693848 | 0.649969 | 2.937500 | 0.495117 | 0.000000 | 0.770508 | 1.000000 | 1.000000 | 1.000000 | NA | NA | 0.098567 |
| UTailoR_official_strict_25_100nt | `measured_external_protocol_subset` | 315 | 0.869331 | 0.036105 | 0.069841 | 0.704762 | 0.576329 | 4.419048 | 0.000000 | 0.012698 | 0.892063 | 1.000000 | 1.000000 | 1.000000 | NA | NA | 0.047395 |
| UTRGAN_official_budgeted_10_steps | `measured_external_budgeted` | 1024 | 0.819757 | 0.028721 | 0.375977 | 0.534668 | 0.631947 | 70.998047 | 4.804688 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | NA | NA | 0.013841 |
| UTRGAN_official_paper_default_10000_steps | `measured_external_paper_default` | 1024 | 0.831717 | 0.040681 | 0.380859 | 0.571289 | 0.652633 | 72.485352 | 8.088867 | 0.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | NA | NA | 2.732809 |

## Descriptive Distribution Differences

| Metric | n | MEF local search | UTRGAN | MEF - UTRGAN | paired p |
|---|---:|---:|---:|---:|---:|
| `te_proxy` | 1024 | 0.874059 | 0.819757 | 0.054302 | NA |
| `te_proxy_delta_vs_native` | 1024 | 0.083022 | 0.028721 | 0.054302 | NA |
| `uaug_count` | 1024 | 0.036133 | 0.375977 | -0.339844 | NA |
| `kozak_score` | 1024 | 0.693848 | 0.534668 | 0.159180 | NA |
| `start_accessibility_proxy` | 1024 | 0.649969 | 0.631947 | 0.018022 | NA |
| `utr_edit_distance_vs_native` | 1024 | 2.937500 | 70.998047 | -68.060547 | NA |
| `normalized_utr_edit_distance_vs_native` | 1024 | 0.094979 | 0.626328 | -0.531349 | NA |
| `utr_length_delta` | 1024 | 0.495117 | 4.804688 | -4.309570 | NA |

## UTRGAN Paper-Default vs Budgeted Protocol

| Metric | n | paper 10000 | budgeted 10 | paper - budgeted | paired p |
|---|---:|---:|---:|---:|---:|
| `te_proxy` | 1024 | 0.831717 | 0.819757 | 0.011960 | NA |
| `te_proxy_delta_vs_native` | 1024 | 0.040681 | 0.028721 | 0.011960 | NA |
| `uaug_count` | 1024 | 0.380859 | 0.375977 | 0.004883 | NA |
| `kozak_score` | 1024 | 0.571289 | 0.534668 | 0.036621 | NA |
| `start_accessibility_proxy` | 1024 | 0.652633 | 0.631947 | 0.020687 | NA |
| `utr_edit_distance_vs_native` | 1024 | 72.485352 | 70.998047 | 1.487305 | NA |
| `normalized_utr_edit_distance_vs_native` | 1024 | 0.630322 | 0.626328 | 0.003994 | NA |
| `utr_length_delta` | 1024 | 8.088867 | 4.804688 | 3.284180 | NA |

Both UTRGAN protocols are stochastic de novo batches rather than source-conditioned outputs; this section is descriptive and does not report paired inference.

## MEF 5'UTR-Only Model Ablations

| Comparison | Run | Baseline | n seeds | run | baseline | delta | 95% CI | paired p | signal |
|---|---|---|---:|---:|---:|---:|---|---:|---|
| `pure_utr_teacher_vs_region_adapter` | MEF_pure_utr_teacher_utr5only_top64 | MEF_region_adapter_utr5only_top64 | 10 | 0.003184 | -0.004779 | 0.007962 | [0.007356, 0.008634] | 0.004498 | `significant_positive` |
| `sequential_utr_teacher_vs_region_adapter` | MEF_full_then_utr_teacher_utr5only_top64 | MEF_region_adapter_utr5only_top64 | 10 | 0.002976 | -0.004779 | 0.007755 | [0.007383, 0.008110] | 0.004498 | `significant_positive` |
| `pure_vs_sequential_utr_teacher` | MEF_pure_utr_teacher_utr5only_top64 | MEF_full_then_utr_teacher_utr5only_top64 | 10 | 0.003184 | 0.002976 | 0.000207 | [-0.000397, 0.000856] | 0.567716 | `positive_not_significant` |

## MEF vs UTailoR Strict 25-100 nt Subset

| Comparison | n records | n model seeds | MEF delta TE | UTailoR delta TE | MEF - UTailoR | 95% CI | paired p | signal | MEF edits | UTailoR edits | TE/edit MEF/UTailoR | hard budget | budget matched | alignment |
|---|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---|---:|---:|---|
| `region_adapter_vs_utailor_strict_subset` | 315 | 10 | -0.008876 | 0.036105 | -0.044981 | [-0.045950, -0.044153] | 0.004498 | `significant_negative` | 2.919048 | 4.419048 | -0.003041/0.008170 | NA | `False` | `unmatched` |
| `pure_utr_teacher_vs_utailor_strict_subset` | 315 | 10 | 0.004661 | 0.036105 | -0.031444 | [-0.032264, -0.030695] | 0.004498 | `significant_negative` | 2.931746 | 4.419048 | 0.001590/0.008170 | NA | `False` | `unmatched` |
| `sequential_utr_teacher_vs_utailor_strict_subset` | 315 | 10 | 0.003939 | 0.036105 | -0.032166 | [-0.032715, -0.031474] | 0.004498 | `significant_negative` | 2.899048 | 4.419048 | 0.001359/0.008170 | NA | `False` | `unmatched` |
| `pure_utr_teacher_budget5_vs_utailor_strict_subset` | 315 | 10 | 0.007774 | 0.036105 | -0.028332 | [-0.029271, -0.027301] | 0.004498 | `significant_negative` | 4.796190 | 4.419048 | 0.001621/0.008170 | 5 | `False` | `closer_hard_budget_5_vs_unbounded_external_observed_mean` |

## Remaining Gate

The MEF hard-budget-5 UTailoR strict-subset run narrows the edit-effort mismatch but is not an exact budget match because official UTailoR is not hard-capped per record. UTRGAN paper-default is complete; paired inference remains invalid because UTRGAN is not source-conditioned.
