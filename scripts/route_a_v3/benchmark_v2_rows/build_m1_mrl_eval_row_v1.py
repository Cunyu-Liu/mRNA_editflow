#!/usr/bin/env python
"""Benchmark v2 M1 MRL eval row builder (Task 3.1).

Source: GSE232927 (Castillo-Hair 2024) converted MRL files (mrl_converted_20260909, read-only).
Pairing convention mirrors GSE114002 canonical converter (W0 route_a_v3_route2_gse114002_converter_v1.json):
  - mother/variant-family analog: shared 15nt prefix group within each file (defined_end libraries show real prefix families);
  - within group, source = lexicographically-smallest sequence (mother analog), candidates = group members at Hamming distance 1 from source (variant analog, per GSE114002 candidate rule: equal-length Hamming 1-3, here restricted to 1 for eval-row determinism);
  - additionally, for group-size-1 sequences (random libraries), global Hamming-1 neighbor pairs are detected via deletion-neighborhood hashing (real variant siblings / birthday collisions both possible; each sequence used at most once as candidate).
Sampling: frozen VALIDATION-style eval subset, seed 20260920, ~600-900 rows / 200-300 sources per file.
4 TRAIN-overlap sequences (pigeonhole audit v2, tcell_r1 x2 + tcell_r2 x2) are tagged cross_study_duplicate and excluded from the eval subset (they are keep=0 rows anyway; tag is explicit).
Outputs (additive only, new directory route2/benchmark_v2/m1_mrl_eval_row/):
  projection_rows.jsonl, row_definition.md, row_manifest.json
Discipline: read-only on all existing files; protected reads = 0; no writes to existing manifests.
"""
import gzip, json, os, random, sys
from collections import defaultdict, Counter

SRC_DIR = '/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/castillohair2024/mrl_converted_20260909'
OUT_DIR = '/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/benchmark_v2/m1_mrl_eval_row'
SEED = 20260920
FILES = [
    ('GSE232927_processed_defined_end_hepg2_r1_mrl.tsv.gz', 'hepg2_r1'),
    ('GSE232927_processed_defined_end_tcell_r1_mrl.tsv.gz', 'tcell_r1'),
    ('GSE232927_processed_defined_end_tcell_r2_mrl.tsv.gz', 'tcell_r2'),
    ('GSE232927_processed_random_end_hek293t_N25_r1_mrl.tsv.gz', 'N25_r1'),
    ('GSE232927_processed_random_end_hek293t_N25_r2_mrl.tsv.gz', 'N25_r2'),
    ('GSE232927_processed_random_end_hek293t_N50_r1_mrl.tsv.gz', 'N50_r1'),
]
CELL_CTX = {
    'hepg2_r1': 'HEPG2_CELL_LINE',
    'tcell_r1': 'PRIMARY_T_CELL',
    'tcell_r2': 'PRIMARY_T_CELL',
    'N25_r1': 'HEK293T_CELL_LINE',
    'N25_r2': 'HEK293T_CELL_LINE',
    'N50_r1': 'HEK293T_CELL_LINE',
}
REGION = {'hepg2_r1': '5UTR', 'tcell_r1': '5UTR', 'tcell_r2': '5UTR', 'N25_r1': '5UTR', 'N25_r2': '5UTR', 'N50_r1': '5UTR'}
TARGET_ROWS_PER_FILE = 800
TRAIN_OVERLAP_SEQS = {
    'CCCACAGCTTCCCAGGCCGCGGGTGCTGATTGCCCGCCTGCCCGTGGGTC',
    'CTGCTGGCAGAGAAGCTGGAGAACTGTGATTTCAATTAAGGTATTAAGTC',
    'CTTTCAGCAGCTCTCAGGGCCTTGGGCTCATCCCGAGTCCCGGGCTCAGT',
    'CCGGTGAGGCACGGCCCTGCAGATTTTCCAGCGGATCCCCCGGTGGCCTC',
}

def load_file(fn):
    rows = []
    with gzip.open(os.path.join(SRC_DIR, fn), 'rt') as f:
        f.readline()
        for line in f:
            p = line.rstrip('\n').split('\t')
            if p[-1] != '1':
                continue
            rows.append((p[0], float(p[7])))
    return rows

def hamming1_neighbors(seqs):
    buckets = defaultdict(list)
    for s in set(seqs):
        for i in range(len(s)):
            buckets[(i, s[:i] + s[i+1:])].append(s)
    pairs = defaultdict(set)
    for members in buckets.values():
        if len(members) > 1:
            for a in members:
                for b2 in members:
                    if a != b2:
                        pairs[a].add(b2)
    return pairs

def build_pairs(rows):
    seqs = [s for s, _ in rows]
    rl = {s: v for s, v in rows}
    groups = defaultdict(list)
    for s in seqs:
        groups[s[:15]].append(s)
    pair_records = []
    used_as_candidate = set()
    prefix_group_stats = Counter()
    for prefix in sorted(groups):
        members = sorted(set(groups[prefix]))
        prefix_group_stats[len(members)] += 1
        if len(members) < 2:
            continue
        # GSE114002 mother-family analog: mother pairs with ALL Hamming-1 members
        # (candidate rule mirror; each candidate sequence used at most once)
        mother = members[0]
        for m in members[1:]:
            if m in used_as_candidate:
                continue
            if sum(1 for x, y in zip(mother, m) if x != y) == 1:
                pair_records.append((mother, m, 'prefix_group_hamming1', prefix))
                used_as_candidate.add(m)
    leftovers = [s for s in seqs if s not in used_as_candidate]
    global_pairs = 0
    if leftovers:
        neigh = hamming1_neighbors(leftovers)
        used_global = set()
        for s in sorted(neigh):
            if s in used_global or s in used_as_candidate:
                continue
            for c in sorted(p for p in neigh[s] if p not in used_as_candidate and p not in used_global):
                pair_records.append((s, c, 'global_hamming1', None))
                used_global.add(c)
                global_pairs += 1
            used_global.add(s)
    return pair_records, rl, prefix_group_stats, global_pairs

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(SEED)
    all_rows = []
    manifest = {
        'schema_version': 'route_a_v3_route2_benchmark_v2_eval_row_manifest.v1',
        'row_key': 'M1_MRL_EVAL_ROW',
        'study_unit_id': 'GSE232927',
        'built_utc': '2026-09-20',
        'sampling_seed': SEED,
        'files': {},
    }
    total_rows = 0
    excluded_dup = 0
    for fn, short in FILES:
        rows = load_file(fn)
        pairs, rl, pg_stats, global_pairs = build_pairs(rows)
        k = min(TARGET_ROWS_PER_FILE, len(pairs))
        chosen_idx = sorted(rng.sample(range(len(pairs)), k)) if k < len(pairs) else list(range(len(pairs)))
        file_rows = 0
        for idx in chosen_idx:
            src, cand, kind, prefix = pairs[idx]
            if src in TRAIN_OVERLAP_SEQS or cand in TRAIN_OVERLAP_SEQS:
                excluded_dup += 1
                continue
            rl_src = rl[src]
            rl_cand = rl[cand]
            all_rows.append({
                'row_id': f'M1:GSE232927:{short}:pair:{idx:07d}',
                'study_unit': 'GSE232927',
                'source_sequence': src,
                'candidate_sequence': cand,
                'source_relative_edits': [{'position': i, 'source_base': a, 'candidate_base': b}
                                          for i, (a, b) in enumerate(zip(src, cand)) if a != b],
                'task_id': 'MEAN_RIBOSOME_LOAD::region=5UTR',
                'source_group_id': f'GSE232927:{short}:prefixgroup:{prefix if prefix else src[:15]}',
                'assay': 'POLYSOME_FRACTIONATION_MRL',
                'biological_context': CELL_CTX[short],
                'region': REGION[short],
                'endpoint_descriptor': {
                    'endpoint_id': 'MEAN_RIBOSOME_LOAD',
                    'quantity_family': 'RIBOSOME_LOAD',
                    'measurement_form': 'LEVEL_DIFFERENCE',
                    'numerator_family': None,
                    'denominator_family': None,
                },
                'direction_normalized_delta': rl_cand - rl_src,
                'source_endpoint_value': rl_src,
                'candidate_endpoint_value': rl_cand,
                'endpoint_direction': 'HIGHER_IS_BETTER',
                'eval_split_status': 'NEW_EVAL_ROW',
                'pair_kind': kind,
                'tags': [],
                'provenance': {
                    'source_file': f'mrl_converted_20260909/{fn}',
                    'pairing_convention': 'GSE114002_mother_family_analog_prefix15_hamming1_plus_global_hamming1',
                    'mrl_label': 'rl_paper (authoritative per conversion_summary.json)',
                    'sampling_seed': SEED,
                    'train_overlap_policy': 'pigeonhole_audit_v2_TRAIN_4seqs_tagged_excluded',
                },
            })
            file_rows += 1
        total_rows += file_rows
        manifest['files'][short] = {
            'source_file': fn,
            'kept_rows': len(rows),
            'pairs_found': len(pairs),
            'global_hamming1_pairs': global_pairs,
            'prefix_group_size_hist_top': sorted(pg_stats.items())[:8],
            'pairs_sampled': k,
            'eval_rows_emitted': file_rows,
        }
        print(f'{short}: kept={len(rows)} pairs={len(pairs)} sampled={k} emitted={file_rows}', flush=True)
    src_ids = Counter(r['source_group_id'] for r in all_rows)
    total_sources = len(src_ids)
    with open(os.path.join(OUT_DIR, 'projection_rows.jsonl'), 'w') as f:
        for r in all_rows:
            f.write(json.dumps(r) + '\n')
    manifest['total_eval_rows'] = total_rows
    manifest['total_distinct_source_groups'] = total_sources
    manifest['total_train_overlap_tagged_excluded'] = excluded_dup
    manifest['cand_per_source_density'] = {
        'mean': round(total_rows / total_sources, 4),
        'min': min(src_ids.values()) if src_ids else 0,
        'max': max(src_ids.values()) if src_ids else 0,
    }
    manifest['context_distribution'] = dict(Counter(r['biological_context'] for r in all_rows))
    manifest['pair_kind_distribution'] = dict(Counter(r['pair_kind'] for r in all_rows))
    manifest['delta_stats'] = {
        'mean': round(sum(r['direction_normalized_delta'] for r in all_rows) / total_rows, 4) if total_rows else None,
        'min': round(min(r['direction_normalized_delta'] for r in all_rows), 4) if total_rows else None,
        'max': round(max(r['direction_normalized_delta'] for r in all_rows), 4) if total_rows else None,
    }
    with open(os.path.join(OUT_DIR, 'row_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    print(f'M1 DONE: rows={total_rows} source_groups={total_sources} excluded_dup={excluded_dup}')
    print(f'density mean={manifest["cand_per_source_density"]["mean"]} delta_mean={manifest["delta_stats"]["mean"]}')

if __name__ == '__main__':
    main()
