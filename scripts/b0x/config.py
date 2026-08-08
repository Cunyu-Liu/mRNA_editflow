"""B0-X configuration: ACTIVE benchmark assets, endpoints, and delta-derivation specs.

Authority: xeditflow_benchmark_registry.yaml (v2.0) under contract
GOAL-XEDITFLOW-MIGRATION-01.  Only ACTIVE effect benchmarks are configured:
  - EditBench-5U-A1-Natural : GSE114002, GSE217518
  - EditBench-3U-A1-Variant : ENCSR854RUF, GSE186455, GSE200304,
                              GSE232571, GSE232572, GSE261709, GSE298114
GSE246381 is SEALED and is NOT configured here.
"""

# Rebuilt D1 canonical staging (authoritative measured data).
STAGING_DIR = "/mnt/cunyuliu/mrna_editflow_v3_1/d1_3u_rebuild_staging/ordinary"
# Read-only main-repo canonical records (sequence text + labels + edit scripts).
CANONICAL_RECORDS = "/home/cunyuliu/mrna_editflow_goal/mrna_editflow/data/d1_canonical_records.jsonl"
# Raw designed library for GSE114002 WT (source) rl anchors.
GSE114002_RAW_LIB = "/home/cunyuliu/mrna_editflow_goal/mrna_editflow/data/raw/sample2019_mpra/GSM3130443_designed_library.csv.gz"
# Raw count file for GSE200304 WT (source) Freq anchors.
GSE200304_COUNT = "/mnt/cunyuliu/mrna_editflow_p0/GSE200304/GSM6030637_log2_cpm_small_seq_on_plasmid.txt.gz"
GSE200304_TWIST = "/mnt/cunyuliu/mrna_editflow_p0/GSE200304/GSM6030637_Twist_Oligo_Order_with_merged_ids.txt.gz"

ACTIVE_5U = ["GSE114002", "GSE217518"]
ACTIVE_3U = [
    "ENCSR854RUF", "GSE186455", "GSE200304", "GSE232571",
    "GSE232572", "GSE261709", "GSE298114",
]
ACTIVE_STUDIES = ACTIVE_5U + ACTIVE_3U

# S4 leave-one-study-out is the primary macro structure; S1 (within-study
# source-disjoint) is secondary.  For effect baselines we use S4.
PRIMARY_SPLIT = "S4"
SECONDARY_SPLIT = "S1"
SEED = 42

# Each study maps to a delta-derivation spec.
#   type=diff_cand_src   : delta = candidate_value - source_value
#                            (MOLE: candidate + source anchors both measured)
#   type=diff_ref_alt    : delta = alt_value - ref_value (both on candidate row)
#   type=log2fc          : delta = the log2FC endpoint value directly
#   type=diff_wt_meta    : delta = candidate_value - wt_anchor (from metadata)
DELTA_SPECS = {
    # ---- 5U-A1 (source-relative effect) ----
    "GSE114002": {
        "type": "diff_cand_src",
        "cand_endpoint": "ep_rl",
        "src_source": "raw_library",   # WT rl from GSE114002 designed library
        "benchmark": "5U-A1",
    },
    "GSE217518": {
        "type": "diff_wt_meta",
        "endpoints": [
            {"cand_endpoint": "ep_stability_hek", "wt_meta": "stability_wt_hek"},
            {"cand_endpoint": "ep_stability_sh", "wt_meta": "stability_wt_sh"},
        ],
        "benchmark": "5U-A1",
    },
    # ---- 3U-A1 (variant transfer) ----
    "ENCSR854RUF": {
        "type": "diff_ref_alt",
        "alt_endpoint": "ep_log2FoldChange_Alt_HEK293FT",
        "ref_endpoint": "ep_log2FoldChange_Ref_HEK293FT",
        "benchmark": "3U-A1",
        "join_policy": {
            "schema_version": "diff_ref_alt_join_policy_v2_explicit_null_semantics",
            "policy_id": (
                "ENCSR854RUF_EXPLICIT_PAIRED_ENDPOINT_SCALE_"
                "V2_EXPLICIT_NULL_SEMANTICS"
            ),
            "key_fields": ["context_id", "assay_context", "replicate"],
            "context_match": "EXACT",
            "context_required_non_null": True,
            "assay_context_match": "EXACT",
            "assay_context_null_policy": (
                "ALLOW_EXACT_NULL_MECHANICAL_NO_ASSAY_CONTEXT_CLAIM"
            ),
            "replicate_match": "EXACT_CANONICAL_TOKEN_INCLUDING_NULL",
            "replicate_field": "replicate",
            "replicate_null_policy": (
                "ALLOW_EXACT_NULL_NO_REPLICATE_CLAIM"
            ),
            "ref_allowed_raw_unit_tokens": [
                "log2FoldChange_Ref_HEK293FT",
            ],
            "alt_allowed_raw_unit_tokens": [
                "log2FoldChange_Alt_HEK293FT",
            ],
            "comparison_scale_id": (
                "ENCSR854RUF_LOG2_FOLD_CHANGE_PAIRED_ASSAY_SCALE_V1"
            ),
            "raw_unit_semantics": (
                "ENDPOINT_SPECIFIC_RAW_LABEL_TOKEN_NOT_PHYSICAL_UNIT"
            ),
        },
    },
    "GSE186455": {
        "type": "diff_ref_alt",
        "alt_endpoint": "ep_n2a_activity_alt_mean",
        "ref_endpoint": "ep_n2a_activity_ref_mean",
        "benchmark": "3U-A1",
        "join_policy": {
            "schema_version": "diff_ref_alt_join_policy_v2_explicit_null_semantics",
            "policy_id": (
                "GSE186455_EXPLICIT_PAIRED_ENDPOINT_SCALE_"
                "V2_EXPLICIT_NULL_SEMANTICS"
            ),
            "key_fields": ["context_id", "assay_context", "replicate"],
            "context_match": "EXACT",
            "context_required_non_null": True,
            "assay_context_match": "EXACT",
            "assay_context_null_policy": (
                "ALLOW_EXACT_NULL_MECHANICAL_NO_ASSAY_CONTEXT_CLAIM"
            ),
            "replicate_match": "EXACT_CANONICAL_TOKEN_INCLUDING_NULL",
            "replicate_field": "replicate",
            "replicate_null_policy": (
                "ALLOW_EXACT_NULL_NO_REPLICATE_CLAIM"
            ),
            "ref_allowed_raw_unit_tokens": [
                "n2a_activity_ref_mean",
            ],
            "alt_allowed_raw_unit_tokens": [
                "n2a_activity_alt_mean",
            ],
            "comparison_scale_id": (
                "GSE186455_N2A_ACTIVITY_PAIRED_ASSAY_SCALE_V1"
            ),
            "raw_unit_semantics": (
                "ENDPOINT_SPECIFIC_RAW_LABEL_TOKEN_NOT_PHYSICAL_UNIT"
            ),
        },
    },
    "GSE200304": {
        "type": "diff_cand_src",
        "cand_endpoint": "ep_Freq",
        "src_source": "raw_count",     # WT Freq from GSE200304 count file
        "benchmark": "3U-A1",
    },
    "GSE232571": {
        "type": "diff_ref_alt",
        "alt_endpoint": "ep_activity_HEK293_alt_mean",
        "ref_endpoint": "ep_activity_HEK293_ref_mean",
        "benchmark": "3U-A1",
        "join_policy": {
            "schema_version": "diff_ref_alt_join_policy_v2_explicit_null_semantics",
            "policy_id": (
                "GSE232571_EXPLICIT_PAIRED_ENDPOINT_SCALE_"
                "V2_EXPLICIT_NULL_SEMANTICS"
            ),
            "key_fields": ["context_id", "assay_context", "replicate"],
            "context_match": "EXACT",
            "context_required_non_null": True,
            "assay_context_match": "EXACT",
            "assay_context_null_policy": (
                "ALLOW_EXACT_NULL_MECHANICAL_NO_ASSAY_CONTEXT_CLAIM"
            ),
            "replicate_match": "EXACT_CANONICAL_TOKEN_INCLUDING_NULL",
            "replicate_field": "replicate",
            "replicate_null_policy": (
                "ALLOW_EXACT_NULL_NO_REPLICATE_CLAIM"
            ),
            "ref_allowed_raw_unit_tokens": [
                "activity_HEK293_ref_mean",
            ],
            "alt_allowed_raw_unit_tokens": [
                "activity_HEK293_alt_mean",
            ],
            "comparison_scale_id": (
                "GSE232571_HEK293_ACTIVITY_PAIRED_ASSAY_SCALE_V1"
            ),
            "raw_unit_semantics": (
                "ENDPOINT_SPECIFIC_RAW_LABEL_TOKEN_NOT_PHYSICAL_UNIT"
            ),
        },
    },
    "GSE232572": {
        "type": "diff_ref_alt",
        "alt_endpoint": "ep_activity_alt_mean",
        "ref_endpoint": "ep_activity_ref_mean",
        "benchmark": "3U-A1",
        "join_policy": {
            "schema_version": "diff_ref_alt_join_policy_v2_explicit_null_semantics",
            "policy_id": (
                "GSE232572_EXPLICIT_PAIRED_ENDPOINT_SCALE_"
                "V2_EXPLICIT_NULL_SEMANTICS"
            ),
            "key_fields": ["context_id", "assay_context", "replicate"],
            "context_match": "EXACT",
            "context_required_non_null": True,
            "assay_context_match": "EXACT",
            "assay_context_null_policy": (
                "ALLOW_EXACT_NULL_MECHANICAL_NO_ASSAY_CONTEXT_CLAIM"
            ),
            "replicate_match": "EXACT_CANONICAL_TOKEN_INCLUDING_NULL",
            "replicate_field": "replicate",
            "replicate_null_policy": (
                "ALLOW_EXACT_NULL_NO_REPLICATE_CLAIM"
            ),
            "ref_allowed_raw_unit_tokens": [
                "activity_ref_mean",
            ],
            "alt_allowed_raw_unit_tokens": [
                "activity_alt_mean",
            ],
            "comparison_scale_id": (
                "GSE232572_ACTIVITY_PAIRED_ASSAY_SCALE_V1"
            ),
            "raw_unit_semantics": (
                "ENDPOINT_SPECIFIC_RAW_LABEL_TOKEN_NOT_PHYSICAL_UNIT"
            ),
        },
    },
    "GSE261709": {
        "type": "log2fc",
        "endpoint": "ep_log2fc_AGS",
        "benchmark": "3U-A1",
    },
    "GSE298114": {
        "type": "log2fc",
        "endpoint": "ep_log2fc",
        "benchmark": "3U-A1",
    },
}

# Max sequence length used for one-hot padding in NN baselines.
MAX_SEQ_LEN = 100
NUC_ORDER = "ACGU"

# Reconstructed pairs (sequence text + labels) for the 3 assets rebuilt in the
# D1-3U-REBUILD (their records are NOT in the pre-rebuild main-repo canonical).
RECONSTRUCTED_PAIRS = {
    "GSE232571": "/home/cunyuliu/mrna_editflow_goal/mrna_editflow/data/p0/GSE232571/reconstructed_pairs.jsonl",
    "GSE261709": "/home/cunyuliu/mrna_editflow_goal/mrna_editflow/data/p0/GSE261709/reconstructed_pairs.jsonl",
    "GSE298114": "/home/cunyuliu/mrna_editflow_goal/mrna_editflow/data/p0/GSE298114/reconstructed_pairs.jsonl",
}
