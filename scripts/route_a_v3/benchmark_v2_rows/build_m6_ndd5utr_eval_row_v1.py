#!/usr/bin/env python
"""Benchmark v2 M6 NDD 5'UTR eval row builder (Task 3.2).

Source: GSE246381 (Plassmeyer 2025) HEK combined UMI counts (read-only csv.gz).
Row unit = variant-key REF/ALT allele pairs (1,507 keys; 10 REF + 10 ALT barcode rows each;
Control;Basal rows and Shuf rows are never eval rows).
Delta-y (defined before computation, declared in row_definition.md):
  Polysome load P(x) = sum over 6 HEK Polysome_RNA columns, aggregated over the allele's 10 barcodes
  Monosome load M(x) = sum over 6 HEK 80S_RNA columns (80S_RNA = monosome fraction per GEO titles)
  dy = log2((P_alt+pc)/(M_alt+pc)) - log2((P_ref+pc)/(M_ref+pc)), pseudocount pc = 1.0 (declared)
QC (row inclusion): aggregated DNA_ref+DNA_alt >= 10 AND aggregated (Poly+Mono)_ref + (Poly+Mono)_alt >= 10.
Sequences: hg38 streaming window extraction mirroring S1 audit task_155 rule:
  plus-strand span [pos-1-77, pos-1+78) 0-based half-open, variant at offset 77;
  allele applied as given (SeqID carries no strand field); minus-strand fallback via revcomp(ref)/revcomp(alt);
  rows failing both interpretations are dropped with a counted reason (allele_mismatch).
Column mapping evidence: /home/cunyuliu/mrna_editflow_goal/benchmark_v2_rows/hek_sic_map.json
  (42 HEK GSMs -> SRA original fastq filename SIC#### -> CSV column; 7 fractions x 6 reps).
Overlap re-audit (simplified re-run of pigeonhole v2 + conservative pools):
  locus-level: (chrom,pos) vs GSE200304 canonical ids chr:pos and GSE232572 sequence_id loci;
  sequence-level: exact match of extracted 155nt REF/ALT windows vs conservative frozen corpus
  (GSE114002 + ENCSR854RUF + GSE200304 + GSE217518 + GSE186455 + GSE232572 canonical sequences,
  plus projection VALIDATION sequences with split labels). Any hit -> tagged, excluded from eval subset.
Sampling: frozen eval subset seed 20260920, target ~800 rows (rows are variant pairs).
Outputs (additive only): route2/benchmark_v2/m6_ndd5utr_eval_row/{projection_rows.jsonl, row_manifest.json}
Discipline: read-only on all existing files; protected reads = 0; no writes to existing manifests.
"""
import gzip, json, os, random, sys, math
from collections import defaultdict, Counter

CSV = '/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/external_model_assets/candidate_datasets/plassmeyer2025/GSE246381_hek_combined_umi_counts.csv.gz'
SIC_MAP = '/home/cunyuliu/mrna_editflow_goal/benchmark_v2_rows/hek_sic_map.json'
FA_GZ = '/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/reference/hg38.fa.gz'
OUT_DIR = '/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/benchmark_v2/m6_ndd5utr_eval_row'
PROJ_VAL = '/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/projections/xedit_v3/development_train_validation_v1/validation.jsonl'
CANONICAL = {
    'GSE114002': '/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/canonical/GSE114002/v1/canonical_records.private.jsonl',
    'ENCSR854RUF': '/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/canonical/ENCSR854RUF/v1/canonical_records.private.jsonl',
    'GSE200304': '/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/canonical/GSE200304/v1/canonical_records.jsonl',
    'GSE217518': '/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/canonical/GSE217518/v1/canonical_records.jsonl',
    'GSE186455': '/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/canonical/GSE186455/v1/canonical_records.private.jsonl',
    'GSE232572': '/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/canonical/GSE232572/v1/canonical_records.private.jsonl',
}
SEED = 20260920
TARGET_ROWS = 800
PC = 1.0
DNA_MIN = 10
RNA_MIN = 10
WINDOW_UP = 77
WINDOW_DOWN = 78

def revcomp(s):
    return s.translate(str.maketrans('ACGTNacgtn', 'TGCANtgcan'))[::-1]

def load_sic_map():
    m = json.load(open(SIC_MAP))
    frac_cols = defaultdict(list)
    for g, rec in m.items():
        title = rec['title']
        if ', DNA' in title:
            frac_cols['DNA'].append(rec['column'])
        elif ', TotalRNA' in title:
            frac_cols['TotalRNA'].append(rec['column'])
        elif ', 80S_RNA' in title:
            frac_cols['Monosome'].append(rec['column'])
        elif ', 40S_RNA' in title:
            frac_cols['40S'].append(rec['column'])
        elif ', Polysome_RNA' in title:
            frac_cols['Polysome'].append(rec['column'])
        elif ', TRAP_RNA' in title:
            frac_cols['TRAP'].append(rec['column'])
        elif ', Plasmid_Library' in title:
            frac_cols['Plasmid'].append(rec['column'])
    return frac_cols

def parse_seqid(sid):
    p = sid.split(';')
    if len(p) >= 7 and p[0] == 'Variant' and p[3].startswith('Family=') and p[5] in ('REF', 'ALT'):
        chrom, pos = p[1].split(':')
        ref, alt = p[2].split('|')
        return {'chrom': chrom, 'pos': int(pos), 'ref': ref, 'alt': alt,
                'family': p[3], 'enst': p[4], 'allele': p[5], 'barcode': p[6]}
    return None

def stream_extract_windows(records):
    """One streaming pass over hg38.fa.gz; windows [pos-78, pos+77) 0-based incl variant at 77.
    Robust active-window-set implementation: per chrom, windows sorted by start; a dict of
    accumulating buffers keyed by window index; each line appends the overlap slice and closes
    any window whose end <= line end. hg38 FASTA lines are fixed-width, so windows are closed
    deterministically on the line that contains their end coordinate."""
    by_chrom = defaultdict(list)
    for i, r in enumerate(records):
        start = max(0, r['pos'] - 1 - WINDOW_UP)
        end = r['pos'] - 1 + WINDOW_DOWN
        by_chrom[r['chrom']].append((start, end, i))
    ws_sorted = {c: sorted(v) for c, v in by_chrom.items()}
    ws_by_idx = {c: {w[2]: w for w in ws} for c, ws in ws_sorted.items()}
    ws_pointers = {}
    seqs = {}
    buffers = defaultdict(dict)  # chrom -> i -> list of chunks in order
    cur = None
    pos = 0
    with gzip.open(FA_GZ, 'rt') as f:
        for line in f:
            if line.startswith('>'):
                cur = line[1:].strip().split()[0]
                pos = 0
                continue
            seq = line.strip()
            ln = len(seq)
            if ln == 0 or cur not in ws_sorted:
                pos += ln
                continue
            line_start = pos
            line_end = pos + ln
            ws = ws_sorted[cur]
            sp = ws_pointers.setdefault((cur, 'sp'), 0)
            while sp < len(ws) and ws[sp][0] < line_end:
                buffers[cur].setdefault(ws[sp][2], [])
                sp += 1
            ws_pointers[(cur, 'sp')] = sp
            done = []
            for i, buf in buffers[cur].items():
                w = ws_by_idx[cur][i]
                ov_s = max(w[0], line_start)
                ov_e = min(w[1], line_end)
                if ov_e > ov_s:
                    buf.append(seq[ov_s - line_start: ov_e - line_start].upper())
                if w[1] <= line_end:
                    done.append(i)
            for i in done:
                seqs[cur + ':' + str(i)] = ''.join(buffers[cur].pop(i))
            pos += ln
    # leftovers (should be none if all windows fully covered)
    for cur, bufs in buffers.items():
        for i, buf in bufs.items():
            seqs[cur + ':' + str(i)] = ''.join(buf)
    return seqs

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
    frac_cols = load_sic_map()
    print('fraction columns:', {k: len(v) for k, v in frac_cols.items()}, flush=True)

    variants = defaultdict(lambda: {'REF': defaultdict(int), 'ALT': defaultdict(int)})
    n_scanned = 0
    with gzip.open(CSV, 'rt') as f:
        header = f.readline().rstrip().split(',')
        idx = {c: i for i, c in enumerate(header)}
        frac_idx = {fr: [idx[c] for c in cols] for fr, cols in frac_cols.items()}
        for line in f:
            n_scanned += 1
            v = parse_seqid(line.split(',', 1)[0])
            if v is None:
                continue
            vals = line.rstrip('\n').split(',')
            key = (v['chrom'], v['pos'], v['ref'], v['alt'], v['family'], v['enst'])
            for fr in ('DNA', 'Monosome', 'Polysome'):
                for i in frac_idx[fr]:
                    variants[key][v['allele']][fr] += int(vals[i])
    print('rows scanned:', n_scanned, 'variant keys:', len(variants), flush=True)

    records = []
    qc_fail_dna = qc_fail_rna = 0
    for key in sorted(variants):
        chrom, pos_, ref, alt, family, enst = key
        rec = variants[key]
        dna_r, dna_a = rec['REF']['DNA'], rec['ALT']['DNA']
        rna_r = rec['REF']['Polysome'] + rec['REF']['Monosome']
        rna_a = rec['ALT']['Polysome'] + rec['ALT']['Monosome']
        if dna_r + dna_a < DNA_MIN:
            qc_fail_dna += 1
            continue
        if rna_r + rna_a < RNA_MIN:
            qc_fail_rna += 1
            continue
        dy = (math.log2((rec['ALT']['Polysome'] + PC) / (rec['ALT']['Monosome'] + PC))
              - math.log2((rec['REF']['Polysome'] + PC) / (rec['REF']['Monosome'] + PC)))
        records.append({'chrom': chrom, 'pos': pos_, 'ref': ref, 'alt': alt, 'family': family,
                        'enst': enst, 'dy': dy,
                        'ref_poly': rec['REF']['Polysome'], 'ref_mono': rec['REF']['Monosome'], 'ref_dna': dna_r,
                        'alt_poly': rec['ALT']['Polysome'], 'alt_mono': rec['ALT']['Monosome'], 'alt_dna': dna_a})
    print('QC pass:', len(records), 'fail_dna:', qc_fail_dna, 'fail_rna:', qc_fail_rna, flush=True)

    seqs = stream_extract_windows(records)
    print('windows extracted:', len(seqs), flush=True)

    rows = []
    allele_mismatch = 0
    minus_used = 0
    for i, r in enumerate(records):
        key = r['chrom'] + ':' + str(i)
        window = seqs.get(key)
        if window is None or len(window) != 155:
            allele_mismatch += 1
            continue
        ref_seq = window
        def apply(win, a, b):
            w = list(win)
            if w[77:77 + len(a)] != list(a):
                return None
            w[77:77 + len(a)] = list(b)
            return ''.join(w)
        alt_seq = apply(window, r['ref'], r['alt'])
        strand = 'plus'
        if alt_seq is None:
            alt_seq = apply(window, revcomp(r['ref']), revcomp(r['alt']))
            strand = 'minus_fallback'
            if alt_seq is None:
                allele_mismatch += 1
                continue
            minus_used += 1
        rows.append({
            'row_id': f'M6:GSE246381:variant:{i:05d}',
            'study_unit': 'GSE246381',
            'source_sequence': ref_seq,
            'candidate_sequence': alt_seq,
            'source_relative_edits': relative_edits(ref_seq, alt_seq),
            'task_id': 'POLYSOME_TRANSLATION_EFFICIENCY_DELTA::region=5UTR',
            'source_group_id': f'GSE246381:family:{r["family"].split("=")[1]}',
            'assay': 'POLYSOME_FRACTIONATION_MUTATIONAL_MPRA',
            'biological_context': 'HEK_CELL_LINE',
            'region': '5UTR',
            'endpoint_descriptor': {
                'endpoint_id': 'POLYSOME_MONOSOME_LOAD_LOG2_DELTA',
                'quantity_family': 'TRANSLATION_EFFICIENCY',
                'measurement_form': 'LOG2_FOLD',
                'numerator_family': 'POLYSOME_RNA',
                'denominator_family': 'MONOSOME_RNA',
                'pseudocount': PC,
            },
            'direction_normalized_delta': round(r['dy'], 6),
            'endpoint_direction': 'HIGHER_IS_BETTER',
            'eval_split_status': 'NEW_EVAL_ROW',
            'raw_counts': {'ref': {'poly': r['ref_poly'], 'mono': r['ref_mono'], 'dna': r['ref_dna']},
                           'alt': {'poly': r['alt_poly'], 'mono': r['alt_mono'], 'dna': r['alt_dna']}},
            'variant_locus': {'chrom': r['chrom'], 'pos': r['pos'], 'ref': r['ref'], 'alt': r['alt'], 'enst': r['enst']},
            'allele_orientation': strand,
            'tags': [],
            'provenance': {
                'source_file': 'GSE246381_hek_combined_umi_counts.csv.gz',
                'column_mapping_evidence': 'SRA original fastq filenames vs GEO sample titles (42 HEK GSMs, 7 fractions x 6 reps; hek_sic_map.json)',
                'aggregation': '10 barcodes per allele per variant key, UMI counts summed',
                'dy_formula': 'log2((POLY_alt+1)/(MONO_alt+1)) - log2((POLY_ref+1)/(MONO_ref+1)); Polysome=6 Polysome_RNA cols; Monosome=6 80S_RNA cols',
                'sequence_window': 'hg38 [pos-78,pos+77) 0-based incl, variant offset 77, task_155 mirror; minus-strand fallback revcomp(ref|alt)',
                'qc_thresholds': {'dna_ref_plus_alt_min': DNA_MIN, 'rna_poly_plus_mono_ref_plus_alt_min': RNA_MIN},
                'sampling_seed': SEED,
                'sealed_history': ('GSE246381 historically SEALED_EXCLUDED per W0 route_a_v3_route2_14_study_final_inventory_v1.json '
                                   '(SEALED_EXCLUDED / SEALED_NOT_READ) and utr_editflow_contract_v2.yaml section 2 '
                                   '(gse246381_status: historically_exposed_retrospective_external_stress_test, E4, '
                                   'labels_allowed_for_new_training=false, labels_allowed_for_new_hyperparameter_selection=false, '
                                   'forbidden_wording includes sealed/untouched claims, must_report=historical_exposure_path). '
                                   'Unsealed for eval-row use by M6_PLASSMEYER_2025.manifest.json 2026-09-08 (acquisition, sha256, '
                                   'additive-only split policy) + pigeonhole audit v2 PASS (0 locus overlap). '
                                   'This row uses labels for frozen-delta EVALUATION ONLY; historical exposure path reported here as required.'),
            },
        })
    print('rows with sequences:', len(rows), 'allele_mismatch dropped:', allele_mismatch,
          'minus_fallback used:', minus_used, flush=True)

    # overlap re-audit
    ref_loci = set()
    seq_pool = {}
    for study, path in CANONICAL.items():
        if not os.path.exists(path):
            print('missing canonical:', path, flush=True)
            continue
        with open(path) as f:
            for line in f:
                try:
                    cr = json.loads(line)
                except Exception:
                    continue
                crid = cr.get('canonical_record_id', '')
                if study == 'GSE200304' and crid.startswith('GSE200304:chr'):
                    # GSE200304:chr10:100552081_G-T
                    try:
                        parts = crid.split(':')
                        chrom = parts[1]
                        posra = parts[2]
                        pos = int(posra.split('_')[0])
                        ref_loci.add((chrom, pos))
                    except Exception:
                        pass
                if study == 'GSE232572':
                    sid = (cr.get('candidate_metadata') or {}).get('sequence_id', '')
                    for tok in sid.split('|'):
                        if tok.startswith('chr') and ':' in tok:
                            try:
                                c, p = tok.split(':')
                                ref_loci.add((c, int(p)))
                            except Exception:
                                pass
                for sfield in ('source_sequence', 'candidate_sequence'):
                    s = cr.get(sfield)
                    if s:
                        seq_pool.setdefault(s, set()).add((study, 'canonical'))
    if os.path.exists(PROJ_VAL):
        with open(PROJ_VAL) as f:
            for line in f:
                try:
                    pr = json.loads(line)
                except Exception:
                    continue
                if pr.get('split') == 'VALIDATION':
                    for sfield in ('source_sequence', 'candidate_sequence'):
                        s = pr.get(sfield)
                        if s:
                            seq_pool.setdefault(s, set()).add((pr.get('study_unit_id', '?'), 'VALIDATION'))
    print('re-audit pools: loci=', len(ref_loci), 'seq_keys=', len(seq_pool), flush=True)
    locus_hits = seq_hits = 0
    for r in rows:
        if (r['variant_locus']['chrom'], r['variant_locus']['pos']) in ref_loci:
            locus_hits += 1
            r['tags'].append('locus_overlap_frozen_corpus')
        for s in (r['source_sequence'], r['candidate_sequence']):
            if s in seq_pool:
                r['tags'].append('sequence_overlap_frozen_corpus:' + ','.join(sorted(st + '/' + sp for st, sp in seq_pool[s])))
                seq_hits += 1
                break
    print('re-audit: locus_hits=', locus_hits, 'seq_hits=', seq_hits, flush=True)

    rng = random.Random(SEED)
    eligible = [r for r in rows if not r['tags']]
    k = min(TARGET_ROWS, len(eligible))
    if k < len(eligible):
        chosen = sorted(rng.sample(range(len(eligible)), k))
        eval_rows = [eligible[i] for i in chosen]
    else:
        eval_rows = list(eligible)

    with open(os.path.join(OUT_DIR, 'projection_rows.jsonl'), 'w') as f:
        for r in eval_rows:
            f.write(json.dumps(r) + '\n')
    src_ids = Counter(r['source_group_id'] for r in eval_rows)
    dyv = [r['direction_normalized_delta'] for r in eval_rows]
    manifest = {
        'schema_version': 'route_a_v3_route2_benchmark_v2_eval_row_manifest.v1',
        'row_key': 'M6_NDD5UTR_EVAL_ROW',
        'study_unit_id': 'GSE246381',
        'built_utc': '2026-09-20',
        'sampling_seed': SEED,
        'variant_keys_total': len(variants),
        'rows_scanned_csv': n_scanned,
        'qc': {'pass': len(records), 'fail_dna_lt10': qc_fail_dna, 'fail_rna_lt10': qc_fail_rna},
        'sequence_extraction': {'windows_extracted': len(seqs), 'allele_mismatch_or_len_dropped': allele_mismatch,
                                'minus_fallback_used': minus_used,
                                'window_rule': 'hg38 [pos-78,pos+77) 0-based incl, offset 77, revcomp fallback'},
        'rows_before_sampling': len(rows),
        'overlap_reaudit': {'locus_hits_frozen_corpus': locus_hits,
                            'sequence_hits_frozen_corpus': seq_hits,
                            'tagged_rows_excluded_from_eval': len(rows) - len(eligible)},
        'total_eval_rows': len(eval_rows),
        'total_distinct_source_groups': len(src_ids),
        'cand_per_source_density': {'mean': round(len(eval_rows) / len(src_ids), 4) if src_ids else 0,
                                    'min': min(src_ids.values()) if src_ids else 0,
                                    'max': max(src_ids.values()) if src_ids else 0},
        'dy_stats': {'mean': round(sum(dyv) / len(dyv), 4) if dyv else None,
                     'min': round(min(dyv), 4) if dyv else None,
                     'max': round(max(dyv), 4) if dyv else None},
        'fraction_column_counts': {k: len(v) for k, v in frac_cols.items()},
    }
    with open(os.path.join(OUT_DIR, 'row_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=1)
    print('M6 DONE: eval_rows=', len(eval_rows), 'source_groups=', len(src_ids), 'dy_mean=', manifest['dy_stats']['mean'], flush=True)

if __name__ == '__main__':
    main()
