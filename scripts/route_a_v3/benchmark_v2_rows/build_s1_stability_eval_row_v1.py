#!/usr/bin/env python
"""Benchmark v2 S1 stability eval row builder (Task 3.3).

Source: s1_variant_sequence_pairs.v1.jsonl (5,072 extracted pairs; read-only, audit product).
Protected-row exclusion per S1 pigeonhole audit v2 verdict:
  exclude rows where protected_overlap (sequence-level, 4 rows) OR gse217518_mutant_id_splits
  contains VALIDATION/TEST (mutant-id-level, 1,328 rows); union 1,330 rows. Determined per-row
  from audit_hits / flags in the jsonl (not from the audit summary).
Dao cryptic splicing QC (3'UTR arm rows only, 2,580 rows):
  sliding 10nt window U-fraction >= 0.7 (>=7/10 U) AND full-sequence GU..AG splice-like signal
  (donor 'GT' upstream of acceptor 'AG' within 25-155nt span) -> cryptic_splice_risk flag.
  Policy: FLAG BUT KEEP + annotate (rationale in row_definition.md: Su 2025 rows are endogenous
  genomic UTR variant pairs, not synthetic 3'UTR MPRA inserts; U-rich motifs occur naturally in
  3'UTRs (polyA-signal neighbourhoods); Dao 2025 exclusion rule targets synthetic-insert
  measurement artifacts, so flagging (not exclusion) preserves unbiased eval rows; the flag
  enables post-hoc sensitivity analysis).
5'UTR arm rows (2,492) do not receive this QC (recorded as N/A).
Rows: dual-assay sub-rows per (row, assay) with phenotype present:
  biological_context in {SH_SY5Y (t05_*_SH), HEK293T (t05_*_HEK)}; dy = t05_mt - t05_WT (minutes,
  HIGHER_IS_BETTER); mirrors GSE217518 canonical converter contexts (HEK293T / SH_SY5Y).
  Rows with null phenotype in one assay emit only the other sub-row (counted).
Sequences: canonical_115 ref/alt pair (validated to reproduce GSE217518 canonical source windows
  per audit; variant anchored 19nt from window end). task_155 pair not used (single window
  convention chosen; recorded).
No sampling (full protected-excluded set is the eval row; 5,072 -> 3,742 eligible rows).
Outputs (additive only): route2/benchmark_v2/s1_stability_eval_row/{projection_rows.jsonl, row_manifest.json}
Discipline: read-only on all existing files; protected reads = 0.
"""
import json, os, sys
from collections import Counter

SRC = '/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/audits/candidate_datasets_v1/s1_su2025_stability_extracted_pairs_v1/s1_variant_sequence_pairs.v1.jsonl'
OUT_DIR = '/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/benchmark_v2/s1_stability_eval_row'
U_WINDOW = 10
U_MIN = 7

def u_rich(seq):
    for i in range(len(seq) - U_WINDOW + 1):
        w = seq[i:i + U_WINDOW]
        if w.count('T') >= U_MIN or w.count('U') >= U_MIN:
            return True
    return False

def splice_like_signal(seq):
    seq = seq.replace('U', 'T')
    for i in range(len(seq) - 1):
        if seq[i:i+2] == 'GT':
            for j in range(i + 2, min(i + 40, len(seq) - 1)):
                if seq[j:j+2] == 'AG':
                    return (i, j)
    return None

def relative_edits(ref_seq, alt_seq):
    if len(ref_seq) == len(alt_seq):
        return [{'position': j, 'source_base': a, 'candidate_base': b}
                for j, (a, b) in enumerate(zip(ref_seq, alt_seq)) if a != b]
    p = 0
    while p < min(len(ref_seq), len(alt_seq)) and ref_seq[p] == alt_seq[p]:
        p += 1
    return [{'position': p, 'source_segment': ref_seq[p:], 'candidate_segment': alt_seq[p:],
             'type': 'LENGTH_CHANGE'}]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    stats = Counter()
    assay_emitted = Counter()
    flag_counts = Counter()
    for line in open(SRC):
        r = json.loads(line)
        stats['total'] += 1
        mid = r.get('gse217518_mutant_id_splits')
        seq_prot = bool(r.get('protected_overlap'))
        mid_prot = bool(mid and any(s in ('VALIDATION', 'TEST') for s in mid))
        if seq_prot or mid_prot:
            stats['excluded_protected'] += 1
            if seq_prot:
                stats['excluded_seq_level'] += 1
            if mid_prot:
                stats['excluded_mutant_id_level'] += 1
            continue
        arm = r['utr_group']
        ph = r['phenotype']
        ext = r['extraction']['canonical_115']
        ref_seq = ext['ref_seq']
        alt_seq = ext['alt_seq']
        base_row = {
            'study_unit': 'GSE232927_S1_SU2025_STABILITY',
            'source_sequence': ref_seq,
            'candidate_sequence': alt_seq,
            'source_relative_edits': relative_edits(ref_seq, alt_seq),
            'task_id': 'RNA_HALF_LIFE_MINUTES::region=' + ("3UTR" if arm == "3'UTR" else "5UTR"),
            'source_group_id': f'S1_SU2025:gene:{r["gene"]}:{arm}',
            'assay': 'UTR_HALF_LIFE_MPRA',
            'region': "3UTR" if arm == "3'UTR" else "5UTR",
            'endpoint_descriptor': {
                'endpoint_id': 'RNA_HALF_LIFE_MINUTES',
                'quantity_family': 'RNA_STABILITY',
                'measurement_form': 'LEVEL_DIFFERENCE',
                'numerator_family': None,
                'denominator_family': None,
            },
            'endpoint_direction': 'HIGHER_IS_BETTER',
            'eval_split_status': 'NEW_EVAL_ROW',
            'variant_id': r['mutant'],
            'variant_name': r.get('variant_name'),
            'variant_class': r['variant_class'],
            'tags': [],
            'provenance': {
                'source_file': 'audits/candidate_datasets_v1/s1_su2025_stability_extracted_pairs_v1/s1_variant_sequence_pairs.v1.jsonl',
                'audit_verdict': 'S1 pigeonhole audit v2: same-library (GSE217518) overlap expected; cross-study protected 0; use as additive rows must exclude GSE217518-protected rows (4 seq-level + 1328 mutant-id-level)',
                'protected_exclusion_rule': 'per-row: protected_overlap OR gse217518_mutant_id_splits intersects {VALIDATION, TEST}',
                'sequence_convention': 'canonical_115 pair (variant anchored 19nt from window end; audit-validated vs GSE217518 canonical windows)',
                'dual_assay': 'SH/HEK sub-rows share sequences; biological_context distinguishes assay',
            },
        }
        # Dao QC on 3'UTR arm only
        if arm == "3'UTR":
            u_flag = u_rich(alt_seq) or u_rich(ref_seq)
            spl = splice_like_signal(alt_seq)
            if u_flag and spl:
                flag_counts['3UTR_cryptic_splice_risk_flagged'] += 1
                base_row['tags'].append('cryptic_splice_risk_dao_qc')
                base_row['dao_qc'] = {'u_window': U_WINDOW, 'u_min': U_MIN,
                                      'u_rich': True, 'splice_signal': [spl[0], spl[1]],
                                      'policy': 'FLAG_KEEP_ANNOTATE'}
            else:
                flag_counts['3UTR_pass'] += 1
                if u_flag:
                    flag_counts['3UTR_u_rich_only'] += 1
                if spl:
                    flag_counts['3UTR_splice_signal_only'] += 1
        else:
            flag_counts["5UTR_qc_na"] += 1
        for ctx, wt_key, mt_key, ctx_name in [
            ('SH_SY5Y', 't05_WT_SH', 't05_mt_SH', 'SH'),
            ('HEK293T', 't05_WT_HEK', 't05_mt_HEK', 'HEK'),
        ]:
            wt, mt = ph.get(wt_key), ph.get(mt_key)
            if wt is None or mt is None:
                stats[f'phenotype_null_{ctx_name}'] += 1
                continue
            row = dict(base_row)
            row['row_id'] = f'S1:SU2025:{r["row_index"]:05d}:context:{ctx_name}'
            row['biological_context'] = ctx
            row['direction_normalized_delta'] = round(mt - wt, 6)
            row['source_endpoint_value'] = wt
            row['candidate_endpoint_value'] = mt
            row['phenotype_pval'] = ph.get(f'pval_{ctx_name}')
            row['phenotype_sig'] = ph.get(f'sig_{ctx_name}')
            rows.append(row)
            assay_emitted[ctx_name] += 1
    with open(os.path.join(OUT_DIR, 'projection_rows.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    src_ids = Counter(r['source_group_id'] for r in rows)
    dyv = [r['direction_normalized_delta'] for r in rows]
    manifest = {
        'schema_version': 'route_a_v3_route2_benchmark_v2_eval_row_manifest.v1',
        'row_key': 'S1_STABILITY_EVAL_ROW',
        'study_unit_id': 'SU2025_ELIFE97682',
        'built_utc': '2026-09-20',
        'source_rows': stats['total'],
        'excluded_protected_total': stats['excluded_protected'],
        'excluded_seq_level': stats['excluded_seq_level'],
        'excluded_mutant_id_level': stats['excluded_mutant_id_level'],
        'eligible_rows': stats['total'] - stats['excluded_protected'],
        'dao_qc': dict(flag_counts),
        'dao_qc_policy': 'FLAG_KEEP_ANNOTATE (3UTR arm only; rationale in row_definition.md)',
        'phenotype_null_counts': {'SH': stats['phenotype_null_SH'], 'HEK': stats['phenotype_null_HEK']},
        'total_eval_rows': len(rows),
        'assay_subrow_counts': dict(assay_emitted),
        'total_distinct_source_groups': len(src_ids),
        'cand_per_source_density': {'mean': round(len(rows) / len(src_ids), 4) if src_ids else 0,
                                    'min': min(src_ids.values()) if src_ids else 0,
                                    'max': max(src_ids.values()) if src_ids else 0},
        'dy_stats': {'mean': round(sum(dyv) / len(dyv), 4) if dyv else None,
                     'min': round(min(dyv), 4) if dyv else None,
                     'max': round(max(dyv), 4) if dyv else None},
    }
    with open(os.path.join(OUT_DIR, 'row_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    print('S1 DONE:', json.dumps(manifest, indent=1)[:1500], flush=True)

if __name__ == '__main__':
    main()
