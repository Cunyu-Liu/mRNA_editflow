# Remote Execution Status

Last refreshed: 2026-07-16T11:16:39+08:00

## Server

- Host: `cunyuliu@36.137.135.49`
- Remote root: `/home/cunyuliu/mrna_editflow_goal/mrna_editflow`
- Resource gate: `MAX_LOADAVG=80`
- Current load average: `131.68 / 136.41 / 131.50`

## Active Processes

```text
    PID STAT     ELAPSED CMD
```

## Dynamic MEF Processes

```text
1493953 S       20:16:38 bash scripts/run_stage_a_a100_max_train.sh
1493985 S       20:16:38 bash scripts/run_stage_a_a100_max_train.sh
1494013 S       20:16:38 bash scripts/run_stage_a_a100_max_train.sh
1495455 Rl      20:16:32 /home/cunyuliu/miniconda3/envs/editflow/bin/python3.10 -m mrna_editflow.train_backbone --config /home/cunyuliu/mrna_editflow_goal/mrna_editflow/configs/stage_a_full_a100_max.json --records-jsonl /home/cunyuliu/mrna_editflow_goal/mrna_editflow/data/processed/gencode_human_transcripts.records.jsonl --steps 100000 --save-dir /home/cunyuliu/mrna_editflow_goal/mrna_editflow/ckpts/stage_a_full_a100_max_gencode_100k_seed1 --profile-path /home/cunyuliu/mrna_editflow_goal/mrna_editflow/logs/stage_a_full_a100_max_gencode_100k_seed1.profile.jsonl --device cuda --seed 1
1495549 Rl      20:16:31 /home/cunyuliu/miniconda3/envs/editflow/bin/python3.10 -m mrna_editflow.train_backbone --config /home/cunyuliu/mrna_editflow_goal/mrna_editflow/configs/stage_a_full_a100_max.json --records-jsonl /home/cunyuliu/mrna_editflow_goal/mrna_editflow/data/processed/gencode_human_transcripts.records.jsonl --steps 100000 --save-dir /home/cunyuliu/mrna_editflow_goal/mrna_editflow/ckpts/stage_a_full_a100_max_gencode_100k_seed2 --profile-path /home/cunyuliu/mrna_editflow_goal/mrna_editflow/logs/stage_a_full_a100_max_gencode_100k_seed2.profile.jsonl --device cuda --seed 2
1495551 Rl      20:16:31 /home/cunyuliu/miniconda3/envs/editflow/bin/python3.10 -m mrna_editflow.train_backbone --config /home/cunyuliu/mrna_editflow_goal/mrna_editflow/configs/stage_a_full_a100_max.json --records-jsonl /home/cunyuliu/mrna_editflow_goal/mrna_editflow/data/processed/gencode_human_transcripts.records.jsonl --steps 100000 --save-dir /home/cunyuliu/mrna_editflow_goal/mrna_editflow/ckpts/stage_a_full_a100_max_gencode_100k_seed0 --profile-path /home/cunyuliu/mrna_editflow_goal/mrna_editflow/logs/stage_a_full_a100_max_gencode_100k_seed0.profile.jsonl --device cuda --seed 0
1498387 S       20:16:14 bash scripts/run_stage_a_a100_max_train.sh
1499316 Sl      20:16:08 /home/cunyuliu/miniconda3/envs/editflow/bin/python3.10 -m mrna_editflow.train_backbone --config /home/cunyuliu/mrna_editflow_goal/mrna_editflow/configs/stage_a_full_a100_max.json --records-jsonl /home/cunyuliu/mrna_editflow_goal/mrna_editflow/data/processed/gencode_human_transcripts.records.jsonl --steps 100000 --save-dir /home/cunyuliu/mrna_editflow_goal/mrna_editflow/ckpts/stage_a_full_a100_max_gencode_100k_seed5 --profile-path /home/cunyuliu/mrna_editflow_goal/mrna_editflow/logs/stage_a_full_a100_max_gencode_100k_seed5.profile.jsonl --device cuda --seed 5
1658490 Rl      19:56:36 /home/cunyuliu/miniconda3/envs/editflow/bin/python3.10 -m mrna_editflow.train_backbone --config /home/cunyuliu/mrna_editflow_goal/mrna_editflow/configs/stage_a_mig_tiny_gencode.json --records-jsonl /home/cunyuliu/mrna_editflow_goal/mrna_editflow/data/processed/gencode_human_transcripts.records.jsonl --steps 20000 --save-dir /home/cunyuliu/mrna_editflow_goal/mrna_editflow/ckpts/stage_a_mig_tiny_gencode_20k_seed6 --profile-path /home/cunyuliu/mrna_editflow_goal/mrna_editflow/logs/stage_a_mig_tiny_gencode_20k_seed6.profile.jsonl --device cuda --seed 6
1658570 Rl      19:56:36 /home/cunyuliu/miniconda3/envs/editflow/bin/python3.10 -m mrna_editflow.train_backbone --config /home/cunyuliu/mrna_editflow_goal/mrna_editflow/configs/stage_a_mig_tiny_gencode.json --records-jsonl /home/cunyuliu/mrna_editflow_goal/mrna_editflow/data/processed/gencode_human_transcripts.records.jsonl --steps 20000 --save-dir /home/cunyuliu/mrna_editflow_goal/mrna_editflow/ckpts/stage_a_mig_tiny_gencode_20k_seed7 --profile-path /home/cunyuliu/mrna_editflow_goal/mrna_editflow/logs/stage_a_mig_tiny_gencode_20k_seed7.profile.jsonl --device cuda --seed 7
1715460 S       19:49:59 bash scripts/run_refseq_family_leakage_audit.sh
2331626 S     1-03:43:54 bash scripts/run_stage_a_scalelaw_sweep.sh
3516421 S     1-01:38:01 bash scripts/run_downstream_predictor_protocol.sh
3723262 S     1-01:16:52 bash scripts/run_stage_a_downstream_eval_queue.sh
3723448 S     1-01:16:51 bash scripts/watch_p3_readiness.sh
```

## T6 Head1024 Progress

| target length delta | status | seed dirs | eval summaries | sha256/path |
|---:|---|---:|---:|---|
| -30 | summary | 0 | 0 | `25e1dc1ffa2342bbb9ff4855fcab7b9cf7e34c4d154f010b61e2f9ba20aedee6` |
| -15 | summary | 0 | 0 | `27018371be440771f3b62b32a5384d760865417f830405b878984166a84e0f5b` |
| +0 | summary | 10 | 10 | `59a06fe26e7d11a81d73abcb3e09be8e0ac03a4601017554f6cba514241d1cbb` |
| +15 | summary | 10 | 10 | `990b2259e9d96efa001fe29ade07ac90618135a13036e36e9cbca70bad9b4f94` |
| +30 | summary | 0 | 0 | `686eed840026722b0865990b9b3ffb15ed0022d1b310f3446e2fe747880433b5` |

## T6 Head1024 Shards

| target length delta | status | seed dirs | eval summaries | sha256/path |
|---:|---|---:|---:|---|
| -15 | summary | 5 | 5 | `d623a765b55b53520569a64886cae0e97b34eac721141724c7b6cf372f9a6a72` |
| -30 | summary | 5 | 5 | `ac22be72e499fd72d5e0e1401d8c500366702f30cc99f44bae1b112c05a04f6a` |
| +15 | superseded | 4 | 3 | `benchmark/t6_shards/head1024_stagea10k_len_pos15_shard_20260715_022534_seeds5_9_top64` |
| +30 | superseded | 1 | 0 | `benchmark/t6_shards/head1024_stagea10k_len_pos30_extra_20260715_042931_seeds5_9_top64` |
| +30 | superseded | 1 | 0 | `benchmark/t6_shards/head1024_stagea10k_len_pos30_extra_20260715_043956_seeds5_9_gpu6_top64` |
| +30 | superseded | 2 | 1 | `benchmark/t6_shards/head1024_stagea10k_len_pos30_shard_20260715_022534_seeds5_9_top64` |
| +30 | superseded | 3 | 2 | `benchmark/t6_shards/head1024_stagea10k_len_pos30_targeted_20260715_0450_seed8_first_top64` |
| +30 | superseded | 4 | 3 | `benchmark/t6_shards/head1024_stagea10k_len_pos30_targeted_20260715_0450_seed9_first_top64` |
| +30 | superseded | 5 | 4 | `benchmark/t6_shards/head1024_stagea10k_len_pos30_targeted_20260715_0529_seed8_only_top64` |
| +30 | superseded | 5 | 4 | `benchmark/t6_shards/head1024_stagea10k_len_pos30_targeted_20260715_0530_seed8_only_gpu4_top64` |

## T6 Head1024 Merge Coverage

| target length delta | eval seeds | completed seeds | missing seeds | complete |
|---:|---:|---|---|---|
| -30 | 10 | `000,001,002,003,004,005,006,007,008,009` | `-` | `True` |
| -15 | 10 | `000,001,002,003,004,005,006,007,008,009` | `-` | `True` |
| +0 | 10 | `000,001,002,003,004,005,006,007,008,009` | `-` | `True` |
| +15 | 10 | `000,001,002,003,004,005,006,007,008,009` | `-` | `True` |
| +30 | 10 | `000,001,002,003,004,005,006,007,008,009` | `-` | `True` |

## Gate Evidence

Latest watcher tail:

```text
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| region_adapter_utr5_top64 | True | True | True | not_positive | not_positive | not_positive | not_positive | not_positive |
| region_adapter_cds_top64 | True | True | True | not_positive | not_positive | not_positive | not_positive | not_positive |
| region_adapter_utr3_top64 | True | True | True | not_positive | not_positive | not_positive | not_positive | not_positive |
| region_adapter_all_top64 | True | True | True | not_positive | not_positive | not_positive | not_positive | not_positive |

## Missing Artifacts

| path |
|---|
[2026-07-14T16:31:43+08:00] REFRESH SOTA gap report
{"json_path": "/home/cunyuliu/mrna_editflow_goal/mrna_editflow/docs/sota_gap_report.json", "markdown_path": "/home/cunyuliu/mrna_editflow_goal/mrna_editflow/docs/sota_gap_report.md"}
```

GC sweep queue tail:

```text
[2026-07-14T16:43:47+08:00] loadavg 92.36 >= 80; waiting 120s (waited 720s)
[2026-07-14T16:45:47+08:00] loadavg 87.26 >= 80; waiting 120s (waited 840s)
[2026-07-14T16:47:47+08:00] loadavg 88.46 >= 80; waiting 120s (waited 960s)
[2026-07-14T16:49:47+08:00] loadavg 87.73 >= 80; waiting 120s (waited 1080s)
[2026-07-14T16:51:47+08:00] loadavg 79.85 < 80; proceeding
[2026-07-14T16:51:47+08:00] RUN protein-conditioned CDS GC sweep (SLICE=head256)
{"out_json": "/home/cunyuliu/mrna_editflow_goal/mrna_editflow/benchmark/protein_conditioned_cds_gc_sweep_head256.summary.json", "out_jsonl": "/home/cunyuliu/mrna_editflow_goal/mrna_editflow/benchmark/protein_conditioned_cds_gc_sweep_head256.jsonl", "out_md": "/home/cunyuliu/mrna_editflow_goal/mrna_editflow/benchmark/protein_conditioned_cds_gc_sweep_head256.md", "pareto_front_gc_weights": [8.0, 16.0, 4.0, 2.0, 1.0, 0.0, 0.1, 0.5]}
[2026-07-14T17:40:01+08:00] GC SWEEP COMPLETE -> /home/cunyuliu/mrna_editflow_goal/mrna_editflow/benchmark/protein_conditioned_cds_gc_sweep_head256.summary.json
```

## Artifact Status

| artifact | status | sha256 |
|---|---|---|
| `benchmark/region_adapter_vs_hardneg_v2_top64_head256.json` | present | `4aff158937fbe7bc44854354604cfa5005b34a1182a15704ccd007659010aaf3` |
| `benchmark/region_adapter_vs_mo_grpo_top64_head256.json` | present | `8281d88de6ed8f6eeff0bc3604783d068942b3b6e8d89955b4462571c29119f7` |
| `benchmark/region_adapter_vs_mo_scalar_top64_head256.json` | present | `c0ae07348ada35424b24c6cfccc5ed5b200aea70ca73ca31c788a27907f723ff` |
| `benchmark/region_adapter_vs_mo_pareto_top64_head256.json` | present | `f82c3c6bc814cc0cf6500f15821abbe386bb87f746ca3de64f01fd45765250ca` |
| `benchmark/region_adapter_vs_mo_te_only_top64_head256.json` | present | `0dc13c7a6c14f589e31b38e82a0f7c9fd9034fe3d993172fe829fa9d2b70f11c` |
| `benchmark/region_adapter_decision_report_head256.json` | present | `333993473903ba9b4ca660c2766311a9a06b9d4b52d5d3f69531cba1b62a797a` |
| `benchmark/region_adapter_result_audit_head256.json` | present | `0c08196194a31434d14f8dd3b73255f657b954efa0acd4fa552c7b5e4075164c` |
| `benchmark/protein_conditioned_cds_gc_sweep_head256.summary.json` | present | `22819209bd8ad96f3b48a26835d86b84079f5f2692bd4bf8647391283b948976` |
| `benchmark/protein_conditioned_cds_gc_sweep_head256.audit.json` | present | `5cddc2620ffdd0e99ae0a042ea0ac2814f3762a10719635ac47a9ce31afc4edc` |
| `benchmark/protein_conditioned_codon_metrics_head256.json` | present | `846aac9f434fc083483a657df90f5767ef3efbc6509920b37a5de2d4825322dd` |
| `benchmark/protein_conditioned_codon_metrics_head256.md` | present | `6d3836d39c3f431033dc7c80d0d63e515dfcdc7fda078a1e7047eacdd76fcd04` |
| `benchmark/protein_conditioned_t4_head1024/status.json` | present | `9e850d35813c7ab071802d0133834bfd6134416004454839b9bf9323691f6911` |
| `benchmark/protein_conditioned_t4_head1024/status.md` | present | `0f388d18802b8bc9e5a0e06870dfc8394bb788e341bfeace6b1d961f41ff5089` |
| `benchmark/protein_conditioned_t4_head1024/progress.jsonl` | present | `8df35e2811ed736d6155c492e0be84e47126fbeda8cb1a174d3a58b30b987685` |
| `benchmark/codon_lattice_dp_head1024.json` | present | `deffe04d7f5cdf562e12a40f18c78d042fe7233da4db312310ff565f0d9ccf92` |
| `benchmark/protein_conditioned_cds_head1024.summary.json` | present | `69d192a5fc3b8c5fc90b10c2bf318080f824a78347a1930fc988edc7cbab2c60` |
| `benchmark/protein_conditioned_codon_metrics_head1024.json` | present | `33e202c365440773f7d9c2da95b6ab099878940f35ef708c80290c4ed856923f` |
| `benchmark/protein_conditioned_codon_metrics_head1024.md` | present | `4dadec7e7d8e94181780f1b3a4f603d0276ef2d75eaee2808bd372de3aedad3a` |
| `benchmark/protein_conditioned_cds_gc_sweep_head1024.summary.json` | present | `49a272ec31a47393b366128f884f72b2fae98769c33e6b4d4e483494e1b1f0ca` |
| `benchmark/protein_conditioned_cds_gc_sweep_head1024.audit.json` | present | `b284919b1e2f7af0e5c826164a8eee01bccb54c114b08169ae17ec07971395ad` |
| `benchmark/t4_protein_identity_cai_gc_report_head1024.json` | present | `5053b2b05b3ab86959609d054f42c33bd931d6879a02e12cc5adc358e0316f93` |
| `benchmark/t4_protein_identity_cai_gc_report_head1024.md` | present | `8f8e1fc7991d0ac0d2d555cd1f78874f9b30ff5dae22dd2924b77e8027c5f136` |
| `benchmark/multiseed_t5_public_head256_mo_te_only_top64/multiseed_summary.json` | present | `9562b67198f437a9bfccec627fb0e7e33f889d43b2d1798b62d6fefac0c0da67` |
| `benchmark/multiseed_t5_public_head256_mo_scalar_top64/multiseed_summary.json` | present | `acb36743de8182a20cd3ea89e82ad85a10c0c35f2c50c778d07685979528e8a8` |
| `benchmark/multiseed_t5_public_head256_mo_pareto_top64/multiseed_summary.json` | present | `629df09bb9e854e63694d32367de9ba4a500e0f934ac9fd8f4a59649e6f221c3` |
| `benchmark/multiseed_t5_public_head256_mo_grpo_top64/multiseed_summary.json` | present | `018e419c7293751ded1a4b1e274d0ba421067ae191f168c1e245b4f3ac4f3b99` |
| `benchmark/multiseed_t5_public_head256_hardneg_v2_top64/multiseed_summary.json` | present | `7c9f3f6c1446c3b94136049fbe7df2b24d30a6d4841514595733ff07e0845a91` |
| `benchmark/multiseed_t5_public_head1024_mo_te_only_top64/multiseed_summary.json` | present | `48d9fb8c533e2b3d14976e2da0cff673f6d740c96da3edb35be5e896ff6eca3d` |
| `benchmark/multiseed_t5_public_head1024_mo_scalar_top64/multiseed_summary.json` | present | `4a16d84c7f4f19d24ec929847d429a54673499b61b3ae2d25e1a9771c6131563` |
| `benchmark/multiseed_t5_public_head1024_mo_pareto_top64/multiseed_summary.json` | present | `2e421dad80576c7e6d612b264c629e915f72a819ad57cf7a9156a0f6264e771c` |
| `benchmark/multiseed_t5_public_head1024_mo_grpo_top64/multiseed_summary.json` | present | `116b136d0182b5da69ab1a548864109508b27de1839b388a152ee840bcb1bc3c` |
| `benchmark/multiseed_t5_public_head1024_hardneg_v2_top64/multiseed_summary.json` | present | `d6a6895dce7ff29f3014282524022d150e7b7e53bcab3f75da2da3e5b61015d4` |
| `benchmark/frozen_backbone_protocol_head256/summary.json` | present | `dbfd16a9cfca4be485fad91c5aabdb69935195e9a158b13878a517db208d7dbc` |
| `benchmark/frozen_backbone_protocol_head256/leakage.json` | present | `65a65a23af0a1a188101ab35dafd39e531457abddaef7fd72252ff01e4f00136` |
| `benchmark/t1_t7_evidence_status_head256.json` | present | `0b7db37f9e98777791a614e46721e48195a43d50900e4736bb467b089a688b48` |
| `benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k.json` | present | `3986800747603f34347eda905362d5e0e47abacefb7360ee3801a40e7d262112` |
| `benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k.md` | present | `941c1987c1d1585347456999d9e1d64c8b5961c8ffd0454e39c8b582b2dfd779` |
| `benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k_figures/base_composition_full.svg` | present | `40d1d72f063c9b6acc1b7f75ae82374f8bc8c57a630fe1188c6fc806d4586abd` |
| `benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k_figures/base_composition_five_utr.svg` | present | `5abbe92275b96aa0954b1822158775406f28c542ec1b10ff9e4db37883284db8` |
| `benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k_figures/base_composition_cds.svg` | present | `14be50311abc42bcfd246128d99df4dda7b031a56f6be61030f6d371fafd032c` |
| `benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k_figures/base_composition_three_utr.svg` | present | `7320419cddcebad32490122b6c3027c3cb9c412ae50aea9e63fe688dcfc4d09d` |
| `benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k_figures/length_histogram.svg` | present | `c2d35fb99c46a4db8644103f942c679e2f681015299b86a1297efed4135bbf71` |
| `benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k_figures/gc_histogram.svg` | present | `f4ea2b66350b04c21ce31a5a8722bf5ed839c105fda7c87b7c431cc6da318066` |
| `benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k_figures/kmer_top_delta.svg` | present | `1c9224d3054845ed2964adce89236ce8c5fd383f0218a9094f60222ff039fe3e` |
| `benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k_figures/codon_pair_top_delta.svg` | present | `f8b31c93911fb688888afe4d22c59cefa67df43d6f8484425ca92458e42bf09e` |
| `benchmark/external_sota/input_pack_t5_head1024/summary.json` | present | `7f5ff13ab725cdf582e60ae7359f02a94b7fdc7cf56dd6511bea57b8a8680ae0` |
| `benchmark/external_sota/input_pack_t5_head1024/table.md` | present | `0e3a40a4ae0e646762d3071244168612bad2a149b3935f5c39521a1fe83794b0` |
| `benchmark/external_sota/input_pack_t5_head1024/cds_protein_inputs.jsonl` | present | `77e9d89aa5736fe995bfbbbd7f314456d315f73017a8d3d45c7c5745699d60c4` |
| `benchmark/external_sota/input_pack_t5_head1024/utr5_inputs.jsonl` | present | `8ba1a21906a542a98584fefb451856aa5a9807065d7a9173ef8c6c690013dec1` |
| `benchmark/external_sota/input_pack_t5_head1024/metric_schema.json` | present | `b7aba4cc8cc4c90191ea2fe7fc3fa4afeb1b673347d1558e84551219f92b3b51` |
| `benchmark/edit_budget_curve_report_head256_head1024.json` | present | `8b4b0d25ddba3d82a50a741162dc6b42cd0d2fd2c8a34da3b04e1862a2f60aa2` |
| `benchmark/t6_length_curve_report_head256_head1024.json` | present | `3ec3a66af81c8260ee8f6d6734abebe010943afebf9f22801e56d8f3978c0306` |
| `benchmark/region_adapter_decision_report_head1024.json` | present | `b3d491dac43d8d37eb63df316bf237a981c70bc5826ab8a2dfcb5d49207ac632` |
| `benchmark/region_adapter_result_audit_head1024.json` | present | `3d6bde0738900985be0e230d2066b8c835295bf11f25dd1ba7c95f21a7ac13ca` |
| `docs/multiobjective_scaleup_claim_audit_head256_head1024.json` | present | `6606015eea613ea471f01542d22f14594f47500b80a688dafdc177712b49dc61` |
| `docs/sota_readiness_audit_head256.json` | present | `9b84794b22a4a975ae5df6f21f7d0b33a676ccc71411d778109fe925d11238e3` |
| `docs/sota_gap_report.json` | present | `4baf16efeec33fc0930759193b7524f54c30ac0e569cd19be3174e889ff8d9ac` |
| `configs/stage_a_full_a100_max.json` | present | `de40bb3ff0b9d497f25fd67de85d4708644d6f3588f1864551fec16914a789cb` |
| `configs/stage_a_mig_tiny_gencode.json` | present | `c278f228c9524fe1ee73cd338970feb309d2889977c8a0752d2509619be370da` |
| `scripts/run_stage_a_a100_max_train.sh` | present | `4c25d01a209695cd3b36629f2dea403e48b9f63d95cdb0a7ee84c3fd046a687d` |
| `scripts/run_after_stage_a_a100_max.sh` | present | `15caa6feb7297065227140fea4643b9030c80ac3cba24775be8599849f7376b5` |
| `scripts/run_protein_conditioned_t4_slice.sh` | present | `a2eb16caff4e3167ffba3a6a48b3e61961604267226b31d234991e35d8bfa47a` |
| `docs/reproduce_full_training_eval_commands.md` | present | `1e9aa31144e2f2d3483a4b832d4ae53c1eae046f32423b7c1f28605ad9faa96d` |
| `benchmark/stage_a_full_a100_max_gencode_100k_seed0/status.json` | present | `72ec09b013d50160813f39e1e21ad8e01f9aefbb3fd496e8b5fa31ca9fe4dd65` |
| `benchmark/stage_a_full_a100_max_gencode_100k_seed0/status.md` | present | `aab386d7020f5a062e20998341281d8704c64cab08a02200c8400c8fddad7289` |
| `benchmark/stage_a_full_a100_max_gencode_100k_seed0/progress.jsonl` | present | `07140f1c5991098c2cf59d918dfd9a83d411b08a326cc1ac98acd57b449e6b52` |
| `benchmark/stage_a_full_a100_max_gencode_100k_seed0_posteval/status.json` | present | `00feaaba59ba6fd6626431a0686bc56f3220b4f96928cb416e86466129b7dacc` |
| `benchmark/stage_a_full_a100_max_gencode_100k_seed0_posteval/status.md` | present | `ee34dd7242d57b67154a0339fcf4fccac52af286ac99c1741b754f5c37dec381` |
| `benchmark/stage_a_full_a100_max_gencode_100k_seed0_posteval/progress.jsonl` | present | `b9a14421a57c7ff6dbd03cc3c99d0f87a1a0e8f16f37d31a48858fac9baaee34` |
| `logs/stage_a_full_a100_max_gencode_100k_seed0.metadata.json` | present | `5608e2e886bb49653704b45e3a88edd12abe964dca4c5a789583cccd2d10b2f1` |
| `logs/stage_a_full_a100_max_gencode_100k_seed0.profile.jsonl` | present | `9dd1083e1cd8a13136dfddfedd5493ad48ab05b345b2035e7ef7ac7334c159c4` |
| `logs/stage_a_full_a100_max_gencode_100k_seed0.train.log` | present | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `docs/data_scaleup_readiness.json` | present | `87404f1631e97c90e4ffcd24496b6c84d67e115c3906de9cf8a8c30735ab6b11` |
| `docs/data_scaleup_readiness.md` | present | `cc8ebd99a4e43372b9e85c261b7c8695745d652fc4511774be5275eaf03986f9` |
| `docs/dataset_manifest_audit.json` | present | `9ad5e3ba87feee7f49522bf50ec2390f8570676b81723b12b2b1e3478e10c7aa` |
| `docs/dataset_manifest_audit.md` | present | `f74f444dd3295f4062125ff0d369591b9f93d50ffa802e6fdb03f5c69122d016` |
| `benchmark/mpra_te_predictor_protocol_smoke/report.json` | present | `630c4306c0e3e7b20dd726bcd095ee23bd157de8c1bf902a3f2ca607e2b9097e` |
| `benchmark/mpra_te_predictor_protocol_smoke/report.md` | present | `5f56989d427e49596286629b6ab8cbd66bd8488a2e859375363bf79d13cab9e0` |
| `benchmark/mpra_te_predictor_protocol_smoke/predictions.jsonl` | present | `8b4be17d306dddef1bcde1df756b9e92fd574be0158b6619cfc47114853d9922` |
| `benchmark/stability_predictor_protocol_smoke/report.json` | present | `e0e049ac59d31296ec8268bc06ef94bf9750a5bba955dc46de71d178e9d92a5d` |
| `benchmark/stability_predictor_protocol_smoke/report.md` | present | `f7521c71974c616b649de06591448156819f3c7e5797e31790d6c0507aeddb48` |
| `benchmark/stability_predictor_protocol_smoke/predictions.jsonl` | present | `2408c242f4f1b07552a62458873cc02b428b8f60c2f5fc04974886332f7da6cc` |
| `benchmark/family_leakage_protocol_smoke/report.json` | present | `19596de4c36c63352a843b1572d8e555dbf5925189c6da6ac5eb50eb47d7b78e` |
| `benchmark/family_leakage_protocol_smoke/report.md` | present | `8b8709bc1355819093add36220f9c29ff7c926f4edc9a382e8d7cdbd1f96b1d4` |
| `benchmark/family_leakage_protocol_smoke/splits/train.idx` | present | `ff6834a9a698641c8c1fd9f8bcaa02bcb5ba8e73f501430e976acede1f6e40ca` |
| `benchmark/family_leakage_protocol_smoke/splits/val.idx` | present | `d5d2fdb4807b10dcd5585a3ce0a54019a044dce7c7abf66cf56a0655fd64dd53` |
| `benchmark/family_leakage_protocol_smoke/splits/test.idx` | present | `cada028cd941649ee75e1ea490649abd9c69235d8360e5480b8d5b882cb00dc5` |
| `benchmark/gencode_family_leakage_protocol/status.json` | present | `d6514bb234dae00ea8f802e7a5ca5022a3b4791f08ab6c4e2acd82d2f42ff4ff` |
| `benchmark/gencode_family_leakage_protocol/status.md` | present | `dcae32089ee24a84ce1f90fdecf09db9b024d2a1304fd3ffd7908f4307aeed84` |
| `benchmark/gencode_family_leakage_protocol/progress.jsonl` | present | `9f53883f46a2457f40f63dd33f35f65470d650431297c85fb17f36f2b4b95a9e` |
| `docs/paper_table1_sota_landscape.json` | present | `8ffa98ec4195e0b9925c43db69ddf8d1288fedb9730532a2ae21284233ae850b` |
| `docs/paper_table1_sota_landscape.md` | present | `47aebcff4727f7cd417738f872931c986d26cfe33ee5a4f6d99362142881a9d5` |
| `docs/paper_table2_t1_t7_main_results.json` | present | `656350944bc4331541283389fbddfe593d64bf5095301656760ddbeef2634e47` |
| `docs/paper_table2_t1_t7_main_results.md` | present | `f0b81180e3352eea15b8dfe05654017acb38ffae2fd889ed3e82387b1b8f00b0` |
| `docs/paper_table3_external_baseline_readiness.json` | present | `8838deff59836db361068a846e6bb816fb8b26424ed583b5ae12f082b6207dc0` |
| `docs/paper_table3_external_baseline_readiness.md` | present | `1784dfba69400a9f2674615fdae55913fd5b0fce8fcfeccc57601f7f0686d509` |
| `docs/paper_table4_architecture_ablation.json` | present | `45e1e22e709497f783fa9501afae984f660b05ab3badd9fb6264607657d9135f` |
| `docs/paper_table4_architecture_ablation.md` | present | `c077c52499fca60faaf6760d846126a480810b2ceff628b55ff0b243efbf29c5` |
| `docs/paper_table5_scale_law_readiness.json` | present | `8c4f17de881d59e18ef841f5dc14f5de64400e8b60590a807dcdab0ea58b47de` |
| `docs/paper_table5_scale_law_readiness.md` | present | `3ba4034ede4f7dd2a01b415b8a9c2bd3bec97af5ac62506e5e1008cfbb2f4292` |
| `docs/paper_figure1_full_length_edit_flow.json` | present | `c0ac82ff9f6132e0c9fbc80fd2bd16c2789d1e4721c38cc6810f363e970adaf4` |
| `docs/paper_figure1_full_length_edit_flow.md` | present | `c17f3703f0267d0acdc38bca1bddd3dbdeca616eff8d081175ea0e91d660ac9b` |
| `docs/paper_figure2_cascade_recall_precision.json` | present | `42c6e871f1e4dd20ac3695e8a3d3239fa48aa6d221e0994475293bb74d11d9eb` |
| `docs/paper_figure2_cascade_recall_precision.md` | present | `9a4f572a92a97d857d8150d504b6e6b84d6de11c2f3d8ce1adcc888d5ebc88a2` |
| `docs/paper_figure3_oracle_gap_closure.json` | present | `39d1255d17f00a0dbba2aec26476c59a92f856f6c2d067b3170c3d3036f950f9` |
| `docs/paper_figure3_oracle_gap_closure.md` | present | `d23d82585bea1c7525b5c546250627812cb4b9f5bb9aaeb4f68b4d4297e032f0` |
| `benchmark/gencode_family_leakage_protocol/report.json` | present | `862f54901f7fbf8bb7225a43a22d520e7bd265eb13af91edf856d137a25f12d2` |
| `benchmark/gencode_family_leakage_protocol/report.md` | present | `fd4adeefc2bec7ca47ff0fd43b837077c62b3cedca2350fa55a9c5efd6b3f05f` |
| `benchmark/gencode_family_leakage_protocol/splits/test.idx` | present | `1fbc5aa435a7cd743222f32000296614328ff2f704da367e632f6d001c057be3` |
| `benchmark/gencode_family_leakage_protocol/splits/train.idx` | present | `08bc473c3cef3d738d5de4d501ee8791b4a7d954223d8b8901ed0c3806c8ccd1` |
| `benchmark/gencode_family_leakage_protocol/splits/val.idx` | present | `4b618f3849bc98f92a6a244f1477aff1e77b0693ff1d6b254389956066750f52` |
| `logs/gencode_family_readiness.done.json` | present | `c9541e7f3bd1733a927fb6e8b2354369d9c02356505f769739b0ac9bdc2bf795` |
| `benchmark/stage_a_scalelaw_downstream/progress.jsonl` | present | `f2d11661ef71fe3c95a23047ac092ffb961f3ed72a49a8cd091a89239b276f81` |
| `benchmark/stage_a_scalelaw_downstream/status.json` | present | `450e5acb00f4303e3109f3b1777a1497993bad43402fa26352f0b9538549bde3` |
| `benchmark/stage_a_scalelaw_downstream/status.md` | present | `72717b6d547cffccc29c9ad2b94597fdcef7b727b93a6c818fd44cca9e5ec6c5` |
| `benchmark/stage_a_scalelaw_p3p4_20260715_0708/plan.json` | present | `13d71fd1bc414020a0a29a1d75956ee77ac73a6a5ef13ae5704fe6031a1dde5a` |
| `benchmark/stage_a_scalelaw_p3p4_20260715_0708/plan.tsv` | present | `22be60ecf9d316058216c09edc450d12ef2d80325cbd5afc361497cd76c94b8f` |
| `benchmark/stage_a_scalelaw_p3p4_20260715_0708/progress.jsonl` | present | `efec75105eaddcd456e53e811b1aa3c936d92adb23ff472f2abdf4bf2cda7202` |
| `benchmark/stage_a_scalelaw_p3p4_20260715_0714/plan.json` | present | `58cc154f42ecac90d758aef1f4060f3bd19eecae6b724553d52dd635fa77d28c` |
| `benchmark/stage_a_scalelaw_p3p4_20260715_0714/plan.tsv` | present | `f0cf0c19f34af4b3c660dac1c3328192023fd303104147d2268f054462c2f2fc` |
| `benchmark/stage_a_scalelaw_p3p4_20260715_0714/progress.jsonl` | present | `0611706dce659636e11c84fd510e18d52e0af13bc203b66884ec74a7cb9eea11` |
| `benchmark/stage_a_scalelaw_p3p4_20260715_0714/status.json` | present | `5df7b7020acf949028e71bfb13686af167458822aa02f667cddb27e1dea682ce` |
| `benchmark/stage_a_scalelaw_p3p4_20260715_0714/status.md` | present | `4d2e3b5a63fb1f242f8bf02f30f454dd9f8f3f47dbc4475881a5b9ee9fecae3c` |
| `benchmark/refseq_public_build_20260715_0743/progress.jsonl` | present | `625dd232f77d96579d5918ea21904f202d4a5993de3bae47ffa649fb4d84731d` |
| `benchmark/refseq_public_build_20260715_0743/status.json` | present | `4fb3b07cf2fc8838752449bf48e21882e77b74055d250471f8b985f5a180cfaf` |
| `benchmark/refseq_public_build_20260715_0743/status.md` | present | `e1699b769d008c757a885039718cd551acd94d2d1ac2d1be1804414ab10acb8d` |
| `data/raw/human.1.rna.gbff.gz` | present | `6de3a3a93e0d413db027f73cb44c8c59620436677f915fa8b916eed2d64c4f63` |

## Current Readiness

- Region eval started: `True`
- Target artifacts present/missing: `127/0`
- Readiness artifact present: `True`

## Policy

Do not lower `MAX_LOADAVG=80`. Wait for the watcher to pass the load gate, then run `scripts/harvest_sota_artifacts.sh` and inspect the unified readiness report before making any positive SOTA claim.
