# GSE200304 standard development critic result — 2026-08-15

## Outcome

The owner-directed standard development run completed on full non-MIG GPU 3.
It reused the frozen 6,547-member dataset, split, architecture, optimizer,
batch size, evaluator and four baselines.  The previous out-of-memory failure
was therefore a 5 GiB MIG allocation problem, not evidence that the model could
not run on a full A100.

The completed run is a negative predictive result:

- source-group equal-weight test Spearman: `0.001882286575573072`;
- test MAE: `0.13537058487266992`;
- best nonconstant baseline Spearman: `0.017701033937807364`;
- best baseline MAE: `0.13361905984653222`;
- nominal 90% interval coverage: `0.9087665647298675`;
- parameter updates: `72`;
- checkpoint count: `1`;
- peak allocated CUDA memory: `20,994,054,144` bytes of
  `42,404,806,656` visible bytes.

The calibrated predictions were nearly constant: prediction standard
deviation was `0.0034092143915147284`, versus observed-effect standard
deviation `0.20740485975685122`.  Test Pearson correlation was approximately
`0.02415`.  Retaining fewer examples by predicted uncertainty increased MAE,
so the current uncertainty ranking did not provide useful abstention.

## Data geometry diagnosis

The 6,547 records contain 6,544 source groups: 6,541 groups have one candidate
and only 3 groups have two candidates.  The mean test biological standard error
is about 63% of the test effect standard deviation.  This dataset therefore
supports a named-dataset source-relative effect benchmark, but almost no
within-source multi-candidate ranking geometry.

## Decision

Do not launch blind seed/model retries.  The next scientific task is to decide
whether a better endpoint or feature representation has a defensible signal on
the frozen split, and whether the project wants effect prediction or true
within-source ranking.  The result does not establish a biological, cross-study,
true-A2, guidance-superiority, A6 learned or A7 claim.  Portfolio counts remain
`1/1/0/6547`; sealed access remains zero.

Remote aggregate output:

`/mnt/cunyuliu/mrna_xeditflow_routea_v3/runs/A1/A1_DATA_QUALIFICATION_20260810T032128P0800_fd722d5/GSE200304_CRITIC_DEVELOPMENT_GPU3_177d83c/GSE200304_SOURCE_RELATIVE_CRITIC_G1_AGGREGATE_RESULT.json`
