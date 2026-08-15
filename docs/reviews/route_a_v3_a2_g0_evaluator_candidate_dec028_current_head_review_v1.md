# Route A V3 A2 G0 evaluator candidate — DEC028 current-HEAD review v1

Review base: `865b6941ce86822b775da396cef47c4ea9fb985c`  
Authority context: `V3-DEC-028`, runtime `A1-EVT-061`  
Verdict: `PASS_G0_SYNTHETIC_INTERFACE_ONLY_PARTIAL_NOT_ACTIVE`

The A2 G0 candidate was forward-ported without semantic changes and rechecked
against the DEC028 single-study operational context. It remains a synthetic-only,
non-authoritative interface. Its focused suite passes 26 tests.

The reviewed behavior is limited to outcome-blind source-group and known-duplicate
connected components, component-disjoint aggregate split summaries,
direction-normalized `candidate - source` endpoint metadata, biological-replicate
standard-error rules, missing/nonfinite exclusion without zero imputation, and the
predeclared planning calculation whose first joint power/precision pass is 156
post-dedup independent source groups. It reads no project or dataset rows and
does not emit member keys or split assignments.

The evaluator receives neither A6 learner/guide output nor model-selection output.
A2 and A6 reviews remain independent and cannot serve as evidence for each other.

This review does not freeze real membership, a real split, or a final salt; run a
real evaluator; establish NDCG, recall, regret, search or measured-neighborhood
claims; execute formal power; qualify A2; change `1/1/0/6547`; or authorize
training, CUDA, model selection, A7, or a later phase.
