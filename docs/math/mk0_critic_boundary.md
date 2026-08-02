# MK0 critic, conditioning and evaluator boundary

## Hard boundary

The MK0 rate field and primary sampler must generate legal trajectories with no critic object, critic score or critic query. Hard legality is a deterministic constraint kernel and cannot be weakened or changed by guidance, reranking or an evaluator.

`math_kernel_v1` contains no critic reward, reinforcement-learning term, final-evaluator term or unregistered conditioning regularizer. Its only loss is `L_EF + alpha_stop L_stop`, with `alpha_cond=alpha_reg=0`.

## Role matrix

| Role | May enter rate field? | May affect sampling preference? | May affect hard legality? | Allowed in MK0? | Final-evaluation independence |
|---|---:|---:|---:|---:|---|
| Declared region/context/target condition | yes | yes | no | yes | inference-visible task input, not an evaluator |
| Base generator/rate head | yes | yes | no | yes | evaluated, not used as final oracle |
| Frozen functional critic guidance | future FC0 | future FC0 | no | no | log every query and shared provenance |
| Post-generation reranker | future FC0 | future FC0 | no | no | selected on development only |
| Final evaluator `E_final` | no | no | no | isolation audit only | never queried by generator, sampler or selector |

Future FC0 may compare classifier-free condition, frozen critic guidance and post-generation reranking. It must select one development-primary route once, freeze it, and only then evaluate final data. Choosing the best route after final labels is prohibited.

## No-critic interface test

The base sampler is instantiated without a critic parameter, callback, score cache or environment variable. It must complete legal trajectories using only the rate field, hard constraints and seeded randomness. A default dummy critic is not evidence of independence; the interface must be absent or explicitly `None`, with zero queries.

## Final-evaluator isolation

The audit scans:

- generator and sampler function signatures;
- model feature dictionaries and serialized states;
- query and network logs;
- checkpoint metadata; and
- selection/ranking code paths used in MK0.

The required result is `final_evaluator_used_for_guidance: false` with zero queries. Final-evaluator names, handles, scores, gradients and cached features are prohibited from rate computation and candidate selection.

## Append-only query log for future roles

Any future guidance or selection query uses an append-only record containing timestamp, run ID, candidate hash, model identity/revision/hash, role, input-feature class, response hash, caller, and whether the query influenced generation or selection. Missing entries fail role-governance checks; logs are never rewritten to hide a query.

When one model is reused across teacher, conditioning, guidance, selection or evaluation, the report lists shared training data, weights and features and explicitly lowers the independence claim. Renaming the same model does not create independent evidence.

## Evidence boundary

MK0 critic-boundary PASS establishes only that the mathematical base process can run without a critic and that the final evaluator was isolated in this run. It does not validate a future critic, select a guidance route or support a functional-improvement claim.
