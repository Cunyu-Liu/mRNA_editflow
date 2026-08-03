# mRNA-EditFlow UTR Benchmark-First — v3.1 authoritative contract (C3 copy)

```
contract_id:  utr_editflow_goal_v3.1_benchmark_first
contract_version: "3.1"
schema_version: "3.1"
status: AUTHORITATIVE_REVISED_CONTRACT_APPROVED_NOT_ACTIVATED
```

## Authoritative source

This file is the C3 authoritative copy of the v3.1 contract. The original
authoritative draft is preserved at the path recorded below and is the read
source for §5 (schemas), §5.7 (TaskRegistry/SplitRegistry), and §14.3 (C3).

- Authoritative source file: `2026-08-03/ssh-p-22-cunyuliu-36-137-2/mrna_contract_v3_draft.md`
- Authoritative source SHA256: `35dd4bf27a3c7d574ab777f5d858ad1b13dcb9273bdb4961e4c30a1a94bf8759`

## Frozen scope (C3)

- C3 is definition-only: it freezes contract copies, configs, docs, schemas,
  tests and scripts. It performs **no data access**.
- The 21-schema filename set, Task/Split registries, task→split allowlist,
  grouping-atom rule, activation-calibration rule, diagnostic registry expected
  set and sealed cohort set are frozen with the hashes recorded in
  `configs/utr_editflow_contract_v3_1.yaml` and asserted by
  `tests/contracts/test_utr_editflow_v3_1_contract.py`.
- C3 must not PASS until the failing tests on stale active artifacts are
  converted to PASS, and `C3_STATUS.json` / `C3_MANIFEST.json` /
  `C3_SHA256SUMS` are emitted to the run root.

## Key frozen constants (verbatim from §5.7 and §5.1)

| object | SHA256 |
|---|---|
| 21-schema filename set | `d2e5ddaef3665214007422638df3cc6b0357747aad3911efd4f29319647b1762` |
| 12-task ID set | `b0b43cb76f39b32009e3a6ef8ae6d05395d61bf7baa7480743587e6772447207` |
| task semantic descriptor set | `8f42ef044d8de1a26b9b587587c2de99c6068f67f37e269e226e143333245ba3` |
| 10-split ID set | `b8c6fb2718875862da500c949481d04db08d1d21f94e3d13da49e3ace64ff487` |
| split semantic descriptor set | `c8a6c82a9a1ab687ef2c3cb912ed96aae26c73a0662b0ae0911040c37e8ef1fa` |
| task→split allowlist | `02b25e4717e4a7192b658d5e69cdbb198e5b696b3ea520b7a0a887fcf89097ab` |
| grouping-atom rule | `bd8395ab0ec23d98d7c1b717e7fcb0bdd3df6d18002985624cd9eb41f8bd7983` |
| activation-calibration rule | `b2652abda7a2dbb7001e7fb655db9b6ac19f2b8f80fbc65362dc1236fd9781e9` |
| diagnostic registry expected set | `f25c0adc643f38ff26c5e08bf07e4175a4e2571eaae939d61daa91fc6f2aabb2` |
| sealed cohort set | `275774a99cbe46ccd3084747f7a6efa4ac9af04ed841b2932c318f3682f07df0` |

## GSE246381 truth lock (four axes, §1.1)

- `project_sequence_analytic_exposure = NONE_CONFIRMED`
- `project_sequence_analytic_use_types = [NONE_CONFIRMED]`
- `project_label_analytic_exposure = NONE_CONFIRMED`
- `project_label_analytic_use_types = [NONE_CONFIRMED]`
- `pipeline_sequence_materialization = PRESENT`
- `pipeline_label_materialization = PRESENT`
- `foundation_requirement = REQUIRED_FM0_A`
- `foundation_audit_status = DEFERRED_TO_FM0_A`
- `future_role = SEALED_EXTERNAL_FINAL_CANDIDATE`
- E4 / historically-exposed interpretation permanently revoked.