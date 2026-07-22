# RL final evaluation

Status: `incomplete_preflight`.

No method comparison is claimed until all required final-evaluation evidence is available.

## Evidence gates

- `required_methods_unavailable:stage_b_stop,stage_b_dagger,grpo_without_kl,grpo_with_kl,grpo_with_kl_editflow_replay,random_safe_editing`
- `required_ablations_unavailable:no_stop,no_cycle_prevention,raw_intensity_softmax,log_intensity_softmax,single_step_teacher,dagger_teacher,scalar_reward,vector_reward,fixed_preference,random_preference,no_uncertainty_penalty,no_edit_cost,no_kl,no_editflow_replay`
- `final_result_matrix_incomplete:stage_a_editflow_only:family_disjoint_test,stage_a_editflow_only:ood_family_test,stage_a_editflow_only:length_shift_test,stage_a_editflow_only:gc_shift_test,stage_b_single_objective_ranker:family_disjoint_test,stage_b_single_objective_ranker:ood_family_test,stage_b_single_objective_ranker:length_shift_test,stage_b_single_objective_ranker:gc_shift_test,stage_b_vector_reward_ranker:family_disjoint_test,stage_b_vector_reward_ranker:ood_family_test,stage_b_vector_reward_ranker:length_shift_test,stage_b_vector_reward_ranker:gc_shift_test,stage_b_stop:family_disjoint_test,stage_b_stop:ood_family_test,stage_b_stop:length_shift_test,stage_b_stop:gc_shift_test,stage_b_dagger:family_disjoint_test,stage_b_dagger:ood_family_test,stage_b_dagger:length_shift_test,stage_b_dagger:gc_shift_test,grpo_without_kl:family_disjoint_test,grpo_without_kl:ood_family_test,grpo_without_kl:length_shift_test,grpo_without_kl:gc_shift_test,grpo_with_kl:family_disjoint_test,grpo_with_kl:ood_family_test,grpo_with_kl:length_shift_test,grpo_with_kl:gc_shift_test,grpo_with_kl_editflow_replay:family_disjoint_test,grpo_with_kl_editflow_replay:ood_family_test,grpo_with_kl_editflow_replay:length_shift_test,grpo_with_kl_editflow_replay:gc_shift_test,oracle_guided_local_search:family_disjoint_test,oracle_guided_local_search:ood_family_test,oracle_guided_local_search:length_shift_test,oracle_guided_local_search:gc_shift_test,random_safe_editing:family_disjoint_test,random_safe_editing:ood_family_test,random_safe_editing:length_shift_test,random_safe_editing:gc_shift_test,external_baselines:family_disjoint_test,external_baselines:ood_family_test,external_baselines:length_shift_test,external_baselines:gc_shift_test`

## Split and Oracle status

| item | status | manifest_path | family_disjoint | paper_eligible | reason |
| --- | --- | --- | --- | --- | --- |
| family_disjoint_train | available | /home/cunyuliu/mrna_editflow_goal/mrna_editflow/benchmark/dev/gencode_family_leakage_protocol/split_manifest.json | True | False |  |
| family_disjoint_validation | available | /home/cunyuliu/mrna_editflow_goal/mrna_editflow/benchmark/dev/gencode_family_leakage_protocol/split_manifest.json | True | False |  |
| family_disjoint_test | available | /home/cunyuliu/mrna_editflow_goal/mrna_editflow/benchmark/dev/gencode_family_leakage_protocol/split_manifest.json | True | False |  |
| ood_family_test | available | /home/cunyuliu/mrna_editflow_goal/mrna_editflow/benchmark/dev/rl_stage6_assets/splits/ood_family_test/split_manifest.json | True | False |  |
| length_shift_test | available | /home/cunyuliu/mrna_editflow_goal/mrna_editflow/benchmark/dev/rl_stage6_assets/splits/length_shift_test/split_manifest.json | True | False |  |
| gc_shift_test | available | /home/cunyuliu/mrna_editflow_goal/mrna_editflow/benchmark/dev/rl_stage6_assets/splits/gc_shift_test/split_manifest.json | True | False |  |
| species_shift_test | missing |  |  |  | manifest_path_not_supplied |

| item | status | oracle_type | independent | manifest_sha256 | reason |
| --- | --- | --- | --- | --- | --- |
| training | available | p1_04_crossfit_training_teacher_predictions | False | b296dc7b1428f13a2b8cfd07204b103f0f3fba2deb5ad21755e3b9ca5246b4d0 |  |
| heldout | available | gbt_regressor | True | fb66f3bc2a860255c8adaba8b5fd5c1f754331c831316b9a8c1412346a53a3b1 |  |
| alternative | available | alternative_heuristic_local_translation_oracle | False | 3b44854ddbf5756a155646ea255dae073625ffc81c8b363c0db077e069c09d6b |  |
| public_experimental | missing |  |  |  | oracle_manifest_not_supplied |


## Comparator status

| item | status | source_kind | source_path | source_hash | reason |
| --- | --- | --- | --- | --- | --- |
| stage_a_editflow_only | available | checkpoint | /home/cunyuliu/mrna_editflow_goal/mrna_editflow/ckpts/stage_a_public_full_1k/stage_a_best.pt | 074d0dd9adf152ab6a27f2cd850c3c0226f8c42d62cea6adf4747ba3fdf5b0e8 |  |
| stage_b_single_objective_ranker | available | checkpoint | /home/cunyuliu/mrna_editflow_goal/mrna_editflow/ckpts/proposal_ranker_t5_mo_te_only_head256/proposal_ranker_best.pt | 2e0dc8340997281d08154f4dbfe7f3fd7b30c5b4f1cc67d1bc977e23715c7030 |  |
| stage_b_vector_reward_ranker | available | checkpoint | /home/cunyuliu/mrna_editflow_goal/mrna_editflow/ckpts/proposal_ranker_t5_mo_pareto_head256/proposal_ranker_best.pt | 360d9d3b732eec8f15ac066b535e024dfee0e7380c7d9e6cdeb175b646be567e |  |
| stage_b_stop | missing |  |  |  | artifact_not_found |
| stage_b_dagger | missing |  |  |  | artifact_not_found |
| grpo_without_kl | missing |  |  |  | artifact_not_found |
| grpo_with_kl | missing |  |  |  | artifact_not_found |
| grpo_with_kl_editflow_replay | missing |  |  |  | artifact_not_found |
| oracle_guided_local_search | available | artifact | /home/cunyuliu/mrna_editflow_goal/mrna_editflow/benchmark/utr_local_search_head1024.json | 9df0671b4a3bbc5152d7616495064830db6a6c523dbad03e25588a2490af338a |  |
| random_safe_editing | missing |  |  |  | artifact_not_found |
| external_baselines | available | artifact | /home/cunyuliu/mrna_editflow_goal/mrna_editflow/docs/external_sota_real_run_audit.json | 9bcc75d455eb38877e7458785ebe45fad81d96f876101646ef60861b953236ab |  |


## Required per-result provenance

Each admissible row must record: `dataset_hash`, `split_manifest`, `checkpoint_hash`, `code_commit`, `seed`, `hardware`, `runtime`, `oracle_metadata`, `decoder_type`, and `reward_schema_version`. Missing fields reject the result rather than receiving an inferred value.

## Scientific interpretation boundary

This report records heuristic-proxy outputs only as model/Oracle scores. It does not claim measured translation efficiency, half-life, or experimental effects. A result generated on the training Oracle or train-role teacher data is rejected as a final comparison.
