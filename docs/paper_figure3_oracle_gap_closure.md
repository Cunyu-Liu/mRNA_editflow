# Paper Figure 3: Oracle-Gap Closure Curve

- Claim policy: Figure 3 may show offline candidate-pool oracle-gap closure for matched proposal-ranking diagnostics. The oracle is a proxy upper bound within the generated candidate pool; it is not wet-lab validation, an external SOTA result, or proof of unconstrained de novo design.
- Ready for oracle-gap figure draft: `True`; ready for oracle/SOTA claim: `False`; ready for wet-lab claim: `False`
- Best point: `Cascade hard-negative v2` closure `0.21671`; negative closure present: `True`; oracle gap fully closed: `False`

## Caption

Figure 3. Offline oracle-gap closure on the matched head64 proposal pool. The source-to-oracle gap is defined as mean_oracle_top_TE - mean_source_TE; each ranker closes (mean_model_top_TE - mean_source_TE) / gap. The best current point is Cascade hard-negative v2 with closure fraction 0.21671, so most of the proxy oracle gap remains open. Negative closure for the Stage A base is retained rather than hidden.

## Curve Points

| Order | Ranker | Model top TE | Source TE | Oracle top TE | Delta vs source | Closure fraction | Residual gap | Recall fraction |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | Stage A base | 0.78953 | 0.79539 | 0.82764 | -0.00586 | -0.18176 | 0.03812 | 0.03279 |
| 1 | Previous TE-ranker | 0.79779 | 0.79539 | 0.82764 | +0.00240 | 0.07453 | 0.02985 | 0.42623 |
| 2 | UTR-teacher ranker | 0.79552 | 0.79539 | 0.82764 | +0.00013 | 0.00406 | 0.03212 | 0.70492 |
| 3 | Direct hybrid teacher | 0.79608 | 0.79539 | 0.82764 | +0.00069 | 0.02147 | 0.03156 | 0.47541 |
| 4 | Full-then-UTR sequential | 0.79967 | 0.79539 | 0.82764 | +0.00428 | 0.13260 | 0.02798 | 0.40984 |
| 5 | Source-aware hybrid | 0.79676 | 0.79539 | 0.82764 | +0.00137 | 0.04256 | 0.03088 | 0.75410 |
| 6 | Cascade hard-negative v2 | 0.80238 | 0.79539 | 0.82764 | +0.00699 | 0.21671 | 0.02526 | 0.73770 |

## Vega-Lite Spec

```json
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "data": {
    "values": [
      {
        "closure_fraction": -0.18175987331259125,
        "label": "Stage A base",
        "mean_model_top_te": 0.7895265141590562,
        "oracle_best_in_model_top_k_fraction": 0.03278688524590164,
        "order": 0
      },
      {
        "closure_fraction": 0.07452941470733146,
        "label": "Previous TE-ranker",
        "mean_model_top_te": 0.7977926880282098,
        "oracle_best_in_model_top_k_fraction": 0.4262295081967213,
        "order": 1
      },
      {
        "closure_fraction": 0.004062719090048743,
        "label": "UTR-teacher ranker",
        "mean_model_top_te": 0.7955199049460238,
        "oracle_best_in_model_top_k_fraction": 0.7049180327868853,
        "order": 2
      },
      {
        "closure_fraction": 0.021472407892380016,
        "label": "Direct hybrid teacher",
        "mean_model_top_te": 0.7960814247651304,
        "oracle_best_in_model_top_k_fraction": 0.47540983606557374,
        "order": 3
      },
      {
        "closure_fraction": 0.1326001046358848,
        "label": "Full-then-UTR sequential",
        "mean_model_top_te": 0.7996656590889972,
        "oracle_best_in_model_top_k_fraction": 0.4098360655737705,
        "order": 4
      },
      {
        "closure_fraction": 0.04255885458589129,
        "label": "Source-aware hybrid",
        "mean_model_top_te": 0.7967615321390978,
        "oracle_best_in_model_top_k_fraction": 0.7540983606557377,
        "order": 5
      },
      {
        "closure_fraction": 0.21671195185787126,
        "label": "Cascade hard-negative v2",
        "mean_model_top_te": 0.8023785432636124,
        "oracle_best_in_model_top_k_fraction": 0.7377049180327869,
        "order": 6
      }
    ]
  },
  "description": "Offline oracle-gap closure over matched proposal-ranking diagnostics.",
  "encoding": {
    "tooltip": [
      {
        "field": "label",
        "type": "nominal"
      },
      {
        "field": "closure_fraction",
        "type": "quantitative"
      },
      {
        "field": "mean_model_top_te",
        "type": "quantitative"
      },
      {
        "field": "oracle_best_in_model_top_k_fraction",
        "type": "quantitative"
      }
    ],
    "x": {
      "field": "label",
      "sort": {
        "field": "order"
      },
      "type": "nominal"
    },
    "y": {
      "field": "closure_fraction",
      "title": "Oracle-gap closure fraction",
      "type": "quantitative"
    }
  },
  "mark": {
    "point": true,
    "type": "line"
  }
}
```
