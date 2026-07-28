# H1–H8 hypothesis-to-data requirement matrix

Contract: `utr_editflow_goal_v2`

Question: `RQ-UTR-EDITFLOW-V2`

Frozen for D0: 2026-07-28

This matrix is a qualification record, not a declaration that any candidate is
already paper-ready. An endpoint pair identifies a source and a measured
candidate; it does not identify a unique observed edit trajectory.

| Hypothesis | Minimum supervision | Ideal supervision | Non-substitutable fields | Current candidate path | Insufficient-data / alternative path | Alternative cannot support |
|---|---|---|---|---|---|---|
| H1 edit-process modelling | exact source/candidate sequences, endpoint, source-aware split, recoverable canonical edit script | repeated candidates per source with measured variable-length SUB/INS/DEL endpoints, raw counts and replicates | source identity, candidate identity, edit legality, assay/context, split group | paired natural variants from GSE149487, GSE217518, GSE200304; MPRAu substitution plus measured deletion subset; dense endpoints from GSE145046 | ground single edits with measured pairs; train variable-length prior on absolute libraries; treat constructed multi-step paths as latent and report ambiguity | observed biological transition path; insertion efficacy; broad UTR causal process |
| H2 architecture is not replaceable | matched-budget ablations on the same frozen splits and edit vocabulary | measured support for every SUB/INS/DEL/multi-step component in both UTR regions | matched compute, seeds, selection rule, legality masks, source/time/region/target conditions | all qualified paired datasets plus synthetic grammar tests for unsupported actions | if INS or multi-step coverage is absent, test software correctness separately and limit biological ablations to supported actions | biological value of an unsupported action or component |
| H3 constructive hard validity | complete legality metadata and stepwise constraint checking for every generated action | dataset-provided anchor/forbidden-position/budget/length rules with adversarial cases | exact sequence, UTR region, allowed positions, anchor and length policies, edit budget | contract grammar plus exact sequences from qualified UTR candidates | synthetic property tests and exhaustive small-state checks where biological metadata are absent | biological benefit; functional superiority |
| H4 conditional controllability | explicit region, assay/context, endpoint and direction labels with source-aware holdout | multiple contexts and target strengths for the same source family with replicates | condition values, source key, endpoint semantics, negative/permutation controls | MPRAu multi-cell-line context; GSE145046 multiple endpoint assays; 5′/3′ region comparison | assess only conditions genuinely recorded; permutation controls; mark unavailable target-strength axes | control over unmeasured conditions; universal dose response |
| H5 generative advantage over search | measured candidate sets, frozen oracle/critic, matched query/candidate/compute budget | dense measured landscape with multiple valid candidates per source and independent measured test | budget accounting, source/candidate grouping, measured labels, frozen selection and critic | GSE145046 dense fixed-scaffold landscape; MPRAu source families where recoverable; external paired sets for recovery | report search comparison only on supported landscapes; keep prediction-only candidates out of headline evidence | superiority on variable-length insertion or open-ended multi-step biology |
| H6 cross-source and cross-study transfer | exact source/gene/study/context groups and study-aware split | multiple independent studies per region with compatible endpoints | provenance, source family, study, exposure status, assay/context | GSE114002/GSE149487/GSE145046 for 5′UTR and GSE217518/GSE200304/MPRAu for 3′UTR | leave-one-study/source-family-out; retain negative transfer; do not use random-pair headline split | generalization beyond represented studies, endpoints or contexts |
| H7 foundation-model value | frozen checkpoint identity/license/exposure ledger and matched from-scratch control | multiple compatible foundations with pretraining-overlap audit and sample-efficiency curves | checkpoint hash, license, exposure ledger, trainable-parameter/compute match | ENCSR854RUF raw reads remain an observational pretraining candidate only; future foundation audit is required | reuse-first comparison with frozen/adapter/partial-unfreeze controls after D1/B0 manifests freeze | de novo training justification; causal value of observational ENCODE reads; absence of pretraining leakage |
| H8 shared and region-specific structure | both 5′UTR and 3′UTR measured sets, consistent edit representation, region-specific endpoint/constraint metadata | matched endpoint families and balanced source groups across regions | UTR region, endpoint semantics, motif/anchor/length policy, study groups | qualified 5′ candidates (GSE114002, GSE149487, GSE145046) and 3′ candidates (GSE217518, GSE200304, MPRAu) | compare joint, fully shared and fully separate models only after region-specific manifests freeze; report endpoint mismatch | biological equivalence of 5′ and 3′ UTR; CDS/full-length transfer |

## D0 qualification verdict

- Every hypothesis has a candidate verification path and a fail-closed
  insufficient-data path.
- Measured single-nucleotide UTR pairs are available.
- MPRAu provides measured 5-bp deletion tiling for a subset, but this does not
  establish general deletion, insertion, or multi-step coverage.
- No verified source-paired UTR insertion dataset was found in the D0 search.
- Dense or variable-length absolute libraries may support priors or constrained
  landscape studies; they do not automatically become source-paired edit data.
- If later D1 reconstruction cannot recover exact sources, candidates and
  split groups, the affected biological claim remains blocked rather than being
  replaced by a predictor-only claim.
