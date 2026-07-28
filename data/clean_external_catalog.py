"""Per-publication cleaning for the external catalog datasets (2026-07-27).

Implements the ``cleaning_spec`` registered in
:mod:`data.download_external_catalog` for each acquired dataset, turning raw
evidence files (``data/raw/<dataset>/``) into cleaned corpora
(``data/cleaned/<dataset>/``) with a ``cleaning_report.json`` that audits
corpus attrition (same convention as :mod:`data.clean_mrna`).

Design rules:

* All cleaning *predicates* are pure functions, unit-tested without network
  or the raw datasets present.
* Drivers stream gzipped inputs line-by-line so multi-GB RefSeq shards never
  load into memory.
* Every kept record is traceable: outputs are FASTA or JSONL with the source
  accession preserved, and the report records input/kept/dropped counts plus
  the publication protocol applied.

Cleaning protocols per dataset (from the source publications):

* ``refseq_human_mrna_prot`` / ``refseq_mammalian_cds`` / ``sc2_viral_genomes``
  — translation-consistency filter (LucaOne / Life-Code / GenSLM protocol):
  an mRNA<->protein pair is kept only if the protein is an *exact* in-frame
  ORF translation of the mRNA. Pairs are matched by within-shard record
  index (NCBI mRNA_Prot shards are produced as record-aligned pairs); any
  misalignment is caught and dropped by the consistency check itself.
* ``ensembl_human_grch38`` — transcript-ID pairing + exact dedup
  (UTR-LM / mRNA-LM protocol): Ensembl cds/pep FASTA headers share the
  transcript ID, so pairs are matched by ID; identical sequences are
  removed by content hash.
* ``gencode_v50_full`` — chromosome-disjoint split (SpliceAI protocol) and
  canonical-transcript selection via GTF tags (Ensembl_canonical /
  MANE_Select).
* ``optimus_5prime`` — fixed 50-nt 5'UTR window, ACGU alphabet
  (Sample et al. 2019 protocol).
* ``rnacentral_active`` — T->U normalisation + exact dedup (RNA-FM used
  CD-HIT-EST at 100% identity, which is exact dedup); optional length cap
  (ERNIE-RNA removed >1024 nt for their model input).
* ``gtex_v8_expression`` — GCT parsing, IDs harmonised as published (no
  re-normalisation of TPM values).
* ``rfam_seed`` — Stockholm MSA parsing per family (RNA-MSM protocol).
* ``bprna_hf`` — dot-bracket validity and sequence/structure length match
  (UFold protocol).
* ``mrnabert_downstream_zenodo`` / ``utr_lm_repo`` /
  ``e2efold_structure_repo`` — archive integrity verification only; the
  author-shipped labels are not altered.

CLI::

    python3 -m data.clean_external_catalog --list
    python3 -m data.clean_external_catalog --datasets refseq_human_mrna_prot
    python3 -m data.clean_external_catalog --datasets p2
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import logging
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from data.download_external_catalog import EXTERNAL_CATALOG, select_datasets
from mrna_editflow.core.constants import translate

logger = logging.getLogger(__name__)

# Canonical drop-reason keys (stable strings for reports & tests).
REASON_KEPT = "kept"
REASON_ILLEGAL_CHARS = "illegal_chars"
REASON_NO_EXACT_ORF = "no_exact_orf_translation"
REASON_PAIR_MISMATCH = "pair_count_mismatch"
REASON_DUPLICATE = "duplicate_sequence"
REASON_LENGTH = "length_out_of_range"
REASON_WINDOW = "window_not_50nt"
REASON_BAD_STRUCTURE = "invalid_dot_bracket"
REASON_STRUCTURE_LEN = "structure_length_mismatch"
REASON_ARCHIVE_CORRUPT = "archive_corrupt"

DNA_ALPHABET = frozenset("ACGTN")
RNA_ALPHABET = frozenset("ACGUN")
ACGU_STRICT = frozenset("ACGU")
DOT_BRACKET_CHARS = frozenset(".()[]{}<>")
DOT_BRACKET_PAIRS = {"(": ")", "[": "]", "{": "}", "<": ">"}

# SpliceAI (Jaganathan et al. 2019, Cell) chromosome-disjoint split.
SPLICEAI_TEST_CHROMS = frozenset({"chr1", "chr3", "chr5", "chr7", "chr9"})


# ---------------------------------------------------------------------------
# Generic helpers (pure)
# ---------------------------------------------------------------------------
def normalise_dna(seq: str) -> str:
    """Uppercase + strip whitespace (no alphabet assertion)."""
    return seq.strip().upper()


def t_to_u(seq: str) -> str:
    """Uppercase, strip whitespace, map DNA ``T`` -> RNA ``U``."""
    return normalise_dna(seq).replace("T", "U")


def content_hash(seq: str) -> str:
    """SHA-256 of the sequence body (dedup key, RNA-FM 100%-identity proxy)."""
    return hashlib.sha256(seq.encode("ascii")).hexdigest()


def parse_fasta(lines: Iterable[str]) -> Iterator[Tuple[str, str]]:
    """Yield ``(header_without_>, sequence)`` from a FASTA line stream.

    Complexity: O(total bytes). Multi-line sequences are concatenated.
    """
    header: Optional[str] = None
    chunks: List[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(chunks)
            header = line[1:]
            chunks = []
        else:
            if header is None:
                raise ValueError("FASTA content before first header")
            chunks.append(line)
    if header is not None:
        yield header, "".join(chunks)


def accession_of(header: str) -> str:
    """First whitespace-delimited token of a FASTA header (the accession)."""
    return header.split()[0] if header.split else header


def ensembl_transcript_id(header: str) -> str:
    """Transcript ID from an Ensembl cds/pep FASTA header (first token)."""
    return accession_of(header)


# ---------------------------------------------------------------------------
# Translation consistency (LucaOne / Life-Code / GenSLM protocol)
# ---------------------------------------------------------------------------
def find_exact_orf(mrna: str, protein: str) -> Optional[str]:
    """Return the CDS substring of ``mrna`` whose translation == ``protein``.

    ``mrna`` is a sense-strand transcript (DNA or RNA alphabet). The three
    forward frames are translated; each maximal stop-delimited ORF that
    starts with ``M`` is compared to ``protein`` (trailing ``*`` stripped on
    both sides). Exact match required — no mismatches tolerated
    (translation-consistency filter of LucaOne / Life-Code).

    Complexity: O(len(mrna)).
    """
    prot = protein.strip().rstrip("*").upper()
    if not prot:
        return None
    seq = t_to_u(mrna)
    for frame in range(3):
        aa = translate(seq[frame:])
        # A CDS runs from a start codon (M) to a stop. Stop-delimited segments
        # may carry 5'UTR translation prefix, so try every M-start candidate
        # that extends exactly to the segment end (the stop).
        pos_aa = 0
        for seg in aa.split("*"):
            for i, ch in enumerate(seg):
                if ch == "M" and seg[i:] == prot:
                    start = frame + (pos_aa + i) * 3
                    return seq[start:start + 3 * (len(prot) + 1)]
            pos_aa += len(seg) + 1  # +1 accounts for the stop character
    return None


def translation_consistent(mrna: str, protein: str) -> bool:
    """True iff ``protein`` is an exact in-frame ORF translation of ``mrna``."""
    return find_exact_orf(mrna, protein) is not None


# ---------------------------------------------------------------------------
# GENCODE GTF helpers (SpliceAI / 3UTRBERT protocol)
# ---------------------------------------------------------------------------
def parse_gtf_attributes(attr_field: str) -> Dict[str, str]:
    """Parse the 9th GTF column into a dict (``key "value";`` pairs)."""
    attrs: Dict[str, str] = {}
    for item in attr_field.strip().rstrip(";").split(";"):
        item = item.strip()
        if not item:
            continue
        key, _, rest = item.partition(" ")
        attrs[key] = rest.strip().strip('"')
    return attrs


def gtf_tags(attrs: Dict[str, str]) -> frozenset:
    """Normalise the GTF ``tag`` attribute (single or repeated) to a set."""
    raw = attrs.get("tag", "")
    if not raw:
        return frozenset()
    return frozenset(t for t in raw.replace(",", " ").split() if t)


def is_canonical_transcript(attrs: Dict[str, str]) -> bool:
    """GENCODE canonical: tagged Ensembl_canonical or MANE_Select."""
    tags = gtf_tags(attrs)
    return "Ensembl_canonical" in tags or "MANE_Select" in tags


def spliceai_chrom_split(chrom: str) -> str:
    """Chromosome-disjoint split label per the SpliceAI protocol."""
    return "test" if chrom in SPLICEAI_TEST_CHROMS else "train"


# ---------------------------------------------------------------------------
# Optimus 5-Prime (Sample et al. 2019) 50-nt window filter
# ---------------------------------------------------------------------------
def optimus_window_ok(seq: str, window: int = 50) -> bool:
    """Fixed 50-nt 5'UTR window + strict ACGU alphabet (Sample et al. 2019)."""
    s = t_to_u(seq)
    return len(s) == window and set(s) <= ACGU_STRICT


# ---------------------------------------------------------------------------
# GTEx GCT helper (TPM kept exactly as published)
# ---------------------------------------------------------------------------
def parse_gct(lines: Iterable[str]) -> Tuple[List[str], Iterator[List[str]]]:
    """Parse a GCT v1.2 stream: returns (column_names, row iterator).

    Layout: line 1 = ``#1.2``, line 2 = dims, line 3 = header
    (``Name``, ``Description``, then sample IDs). Rows are yielded as
    string lists with TPM values unparsed (left exactly as published).
    """
    it = iter(lines)
    version = next(it).strip()
    if version != "#1.2":
        raise ValueError(f"not a GCT v1.2 stream: {version!r}")
    next(it)  # dims line (n_rows, n_cols) — stream so we don't need it
    header = next(it).rstrip("\n").split("\t")
    return header, (line.rstrip("\n").split("\t") for line in it if line.strip())


# ---------------------------------------------------------------------------
# Rfam Stockholm parser (RNA-MSM protocol)
# ---------------------------------------------------------------------------
def parse_stockholm(lines: Iterable[str]) -> Iterator[Dict[str, Any]]:
    """Yield one dict per Stockholm record: ``{id, sequences}``.

    A record runs from ``# STOCKHOLM`` to a bare ``//`` line. Sequence rows
    are ``<seqname> <seq>``; markup lines (``#=``) are ignored.
    """
    seqs: Dict[str, List[str]] = {}
    in_record = False
    for raw in lines:
        line = raw.strip()
        if line.startswith("# STOCKHOLM"):
            in_record, seqs = True, {}
        elif line == "//" and in_record:
            yield {"sequences": {k: "".join(v) for k, v in seqs.items()}}
            in_record, seqs = False, {}
        elif in_record and line and not line.startswith("#"):
            name, _, s = line.partition(" ")
            seqs.setdefault(name, []).append(s.strip())


# ---------------------------------------------------------------------------
# bpRNA dot-bracket validity (UFold protocol)
# ---------------------------------------------------------------------------
def dot_bracket_valid(structure: str) -> bool:
    """Balanced dot-bracket over ``.()[]{}<>`` (UFold validity check)."""
    stack: List[str] = []
    for ch in structure:
        if ch == ".":
            continue
        if ch in DOT_BRACKET_PAIRS:
            stack.append(ch)
        elif ch in DOT_BRACKET_PAIRS.values():
            if not stack or DOT_BRACKET_PAIRS[stack.pop()] != ch:
                return False
        else:
            return False
    return not stack


def bprna_record_ok(sequence: str, structure: str) -> bool:
    """Sequence/structure length match + valid dot-bracket + strict ACGU."""
    seq = t_to_u(sequence)
    return (
        len(seq) == len(structure)
        and set(seq) <= ACGU_STRICT
        and dot_bracket_valid(structure)
    )


# ---------------------------------------------------------------------------
# Archive integrity (mRNABERT Zenodo / UTR-LM / E2Efold)
# ---------------------------------------------------------------------------
def zip_integrity_ok(path: Path) -> Tuple[bool, int]:
    """(all_members_ok, n_members) via ``ZipFile.testzip`` (no extraction)."""
    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        return bad is None, len(zf.namelist())


# ---------------------------------------------------------------------------
# Streaming drivers (never exercised in unit tests)
# ---------------------------------------------------------------------------
def _gz_lines(path: Path) -> Iterator[str]:
    with gzip.open(path, "rt", encoding="ascii", errors="replace") as fh:
        yield from fh


def _write_clean_fasta(out_path: Path, records: Iterator[Tuple[str, str]]) -> int:
    n = 0
    with gzip.open(out_path, "wt", encoding="ascii") as fh:
        for header, seq in records:
            fh.write(f">{header}\n{seq}\n")
            n += 1
    return n


def clean_refseq_pair_shards(rna_gz: Path, prot_gz: Path,
                             out_rna_gz: Path, out_prot_gz: Path) -> Dict[str, int]:
    """Translation-consistency filter over one record-aligned shard pair."""
    stats = {REASON_KEPT: 0, REASON_NO_EXACT_ORF: 0, REASON_ILLEGAL_CHARS: 0,
             REASON_PAIR_MISMATCH: 0}
    rna_iter = parse_fasta(_gz_lines(rna_gz))
    prot_iter = parse_fasta(_gz_lines(prot_gz))
    kept: List[Tuple[Tuple[str, str], Tuple[str, str]]] = []
    n_rna = n_prot = 0
    while True:
        rna_rec = next(rna_iter, None)
        prot_rec = next(prot_iter, None)
        if rna_rec is None or prot_rec is None:
            if rna_rec is not None or prot_rec is not None:
                stats[REASON_PAIR_MISMATCH] += 1
            break
        n_rna += 1
        n_prot += 1
        rna_head, rna_seq = rna_rec
        prot_head, prot_seq = prot_rec
        if not set(normalise_dna(rna_seq)) <= DNA_ALPHABET:
            stats[REASON_ILLEGAL_CHARS] += 1
            continue
        if find_exact_orf(rna_seq, prot_seq) is None:
            stats[REASON_NO_EXACT_ORF] += 1
            continue
        stats[REASON_KEPT] += 1
        kept.append((rna_rec, prot_rec))
    _write_clean_fasta(out_rna_gz, ((h, s) for (h, s), _ in kept))
    _write_clean_fasta(out_prot_gz, ((h, s) for _, (h, s) in kept))
    stats["input_pairs"] = min(n_rna, n_prot)
    return stats


def clean_ensembl_cds_pep(cds_gz: Path, pep_gz: Path,
                          out_cds_gz: Path, out_pep_gz: Path) -> Dict[str, int]:
    """Transcript-ID pairing + exact dedup (UTR-LM / mRNA-LM protocol)."""
    pep_by_tx = {ensembl_transcript_id(h): (h, s) for h, s in parse_fasta(_gz_lines(pep_gz))}
    stats = {REASON_KEPT: 0, REASON_NO_EXACT_ORF: 0, REASON_DUPLICATE: 0,
             REASON_ILLEGAL_CHARS: 0}
    seen: set = set()
    kept: List[Tuple[Tuple[str, str], Tuple[str, str]]] = []
    n_in = 0
    for cds_head, cds_seq in parse_fasta(_gz_lines(cds_gz)):
        n_in += 1
        tx = ensembl_transcript_id(cds_head)
        pep = pep_by_tx.get(tx)
        if pep is None:
            stats[REASON_NO_EXACT_ORF] += 1
            continue
        if not set(normalise_dna(cds_seq)) <= DNA_ALPHABET:
            stats[REASON_ILLEGAL_CHARS] += 1
            continue
        h = content_hash(t_to_u(cds_seq))
        if h in seen:
            stats[REASON_DUPLICATE] += 1
            continue
        # Ensembl CDS FASTA is the coding sequence: translate and compare.
        if translate(t_to_u(cds_seq)).rstrip("*") != pep[1].rstrip("*"):
            stats[REASON_NO_EXACT_ORF] += 1
            continue
        seen.add(h)
        stats[REASON_KEPT] += 1
        kept.append(((cds_head, cds_seq), pep))
    _write_clean_fasta(out_cds_gz, ((h, s) for (h, s), _ in kept))
    _write_clean_fasta(out_pep_gz, ((h, s) for _, (h, s) in kept))
    stats["input_cds"] = n_in
    return stats


def clean_rnacentral(in_gz: Path, out_gz: Path,
                     max_len: Optional[int] = None) -> Dict[str, int]:
    """T->U normalisation + exact dedup (RNA-FM 100%-identity protocol)."""
    stats = {REASON_KEPT: 0, REASON_DUPLICATE: 0, REASON_ILLEGAL_CHARS: 0,
             REASON_LENGTH: 0}
    seen: set = set()

    def _records() -> Iterator[Tuple[str, str]]:
        for header, seq in parse_fasta(_gz_lines(in_gz)):
            s = t_to_u(seq)
            if not set(s) <= RNA_ALPHABET:
                stats[REASON_ILLEGAL_CHARS] += 1
                continue
            if max_len is not None and len(s) > max_len:
                stats[REASON_LENGTH] += 1
                continue
            h = content_hash(s)
            if h in seen:
                stats[REASON_DUPLICATE] += 1
                continue
            seen.add(h)
            stats[REASON_KEPT] += 1
            yield header, s

    stats["output"] = _write_clean_fasta(out_gz, _records())
    return stats


def clean_optimus_csv(in_csv: Path, out_csv: Path,
                      seq_column: str = "utr") -> Dict[str, int]:
    """50-nt window + ACGU filter over the author SNV phenotype table."""
    import csv

    stats = {REASON_KEPT: 0, REASON_WINDOW: 0}
    with open(in_csv, newline="") as fin, open(out_csv, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if optimus_window_ok(row.get(seq_column, "")):
                writer.writerow(row)
                stats[REASON_KEPT] += 1
            else:
                stats[REASON_WINDOW] += 1
    return stats


# ---------------------------------------------------------------------------
# Per-dataset drivers (streamed; never exercised in unit tests)
# ---------------------------------------------------------------------------
def _merge_stats(total: Dict[str, int], part: Dict[str, int]) -> Dict[str, int]:
    for key, value in part.items():
        total[key] = total.get(key, 0) + value
    return total


def drive_refseq_pairs(raw_dir: Path, out_dir: Path, prefix: str,
                       indices: Iterable[int]) -> Dict[str, int]:
    """Run the translation-consistency filter over record-aligned shard pairs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total: Dict[str, int] = {}
    for i in indices:
        rna_gz = raw_dir / f"{prefix}.{i}.rna.fna.gz"
        prot_gz = raw_dir / f"{prefix}.{i}.protein.faa.gz"
        if not (rna_gz.exists() and prot_gz.exists()):
            logger.warning("SKIP missing shard pair %s.%d", prefix, i)
            _merge_stats(total, {"shards_missing": 1})
            continue
        stats = clean_refseq_pair_shards(
            rna_gz, prot_gz,
            out_dir / f"{prefix}.{i}.rna.clean.fna.gz",
            out_dir / f"{prefix}.{i}.protein.clean.faa.gz",
        )
        _merge_stats(total, stats)
        _merge_stats(total, {"shards_done": 1})
        logger.info("%s.%d: %s", prefix, i, stats)
    return total


def drive_ensembl(raw_dir: Path, out_dir: Path) -> Dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return clean_ensembl_cds_pep(
        raw_dir / "Homo_sapiens.GRCh38.cds.all.fa.gz",
        raw_dir / "Homo_sapiens.GRCh38.pep.all.fa.gz",
        out_dir / "ensembl_cds.clean.fa.gz",
        out_dir / "ensembl_pep.clean.fa.gz",
    )


def drive_gencode_v50(raw_dir: Path, out_dir: Path) -> Dict[str, int]:
    """Canonical-transcript selection + SpliceAI chromosome-disjoint split."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tx_meta: Dict[str, Tuple[str, bool]] = {}  # tx_id -> (chrom, canonical)
    with gzip.open(raw_dir / "gencode.v50.annotation.gtf.gz", "rt",
                   encoding="ascii", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9 or cols[2] != "transcript":
                continue
            attrs = parse_gtf_attributes(cols[8])
            tx = attrs.get("transcript_id")
            if tx:
                tx_meta[tx] = (cols[0], is_canonical_transcript(attrs))
    stats = {"input_transcripts_fasta": 0, "kept_train": 0, "kept_test": 0,
             "dropped_not_canonical": 0, "dropped_no_gtf_transcript": 0}

    def _split_records(split: str) -> Iterator[Tuple[str, str]]:
        for header, seq in parse_fasta(_gz_lines(raw_dir / "gencode.v50.pc_transcripts.fa.gz")):
            stats["input_transcripts_fasta"] += 1
            tx = header.split("|")[0].split()[0]
            meta = tx_meta.get(tx)
            if meta is None:
                stats["dropped_no_gtf_transcript"] += 1
                continue
            chrom, canonical = meta
            if not canonical:
                stats["dropped_not_canonical"] += 1
                continue
            if spliceai_chrom_split(chrom) != split:
                continue
            stats[f"kept_{split}"] += 1
            yield header, t_to_u(seq)

    n_train = _write_clean_fasta(out_dir / "gencode_v50_canonical_train.fa.gz",
                                 _split_records("train"))
    n_test = _write_clean_fasta(out_dir / "gencode_v50_canonical_test.fa.gz",
                                _split_records("test"))
    stats["gtf_transcripts"] = len(tx_meta)
    stats["output_train"] = n_train
    stats["output_test"] = n_test
    return stats


def drive_optimus(raw_dir: Path, out_dir: Path) -> Dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = clean_optimus_csv(
        raw_dir / "snv_phenotype_log_diff.csv",
        out_dir / "snv_phenotype_log_diff.clean.csv",
    )
    ok, n_members = zip_integrity_ok(raw_dir / "human_5utr_modeling_master.zip")
    stats["repo_zip_ok"] = int(ok)
    stats["repo_zip_members"] = n_members
    return stats


def drive_rnacentral(raw_dir: Path, out_dir: Path) -> Dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return clean_rnacentral(
        raw_dir / "rnacentral_active.fasta.gz",
        out_dir / "rnacentral_active.clean.fasta.gz",
    )


def drive_gtex(raw_dir: Path, out_dir: Path) -> Dict[str, Any]:
    """Verify GCT streams and re-emit with version-stripped IDs (TPM untouched)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, Any] = {}
    for name in ("GTEx_v8_gene_median_tpm", "GTEx_v8_gene_tpm",
                 "GTEx_v8_transcript_tpm"):
        src = raw_dir / f"{name}.gct.gz"
        if not src.exists():
            stats[name] = {"skipped": "not downloaded"}
            continue
        dst = out_dir / f"{name}.harmonised.tsv.gz"
        n_rows = 0
        with gzip.open(src, "rt", encoding="ascii", errors="replace") as fin:
            header, rows = parse_gct(fin)
            with gzip.open(dst, "wt", encoding="ascii") as fout:
                fout.write("\t".join(header) + "\n")
                for fields in rows:
                    fields[0] = fields[0].split(".")[0]  # strip version suffix
                    fout.write("\t".join(fields) + "\n")
                    n_rows += 1
        stats[name] = {"rows": n_rows, "cols": len(header)}
        logger.info("gtex %s: %d rows x %d cols", name, n_rows, len(header))
    attrs = raw_dir / "GTEx_v8_sample_attributes.txt"
    if attrs.exists():
        n = sum(1 for _ in _gz_lines(attrs)) if attrs.suffix == ".gz" else \
            sum(1 for _ in open(attrs, encoding="ascii", errors="replace"))
        stats["GTEx_v8_sample_attributes"] = {"lines": n}
    return stats


def drive_rfam(raw_dir: Path, out_dir: Path) -> Dict[str, int]:
    """Parse Rfam.seed Stockholm MSAs; per-family summary (RNA-MSM protocol)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"families": 0, "sequences": 0}
    with gzip.open(out_dir / "rfam_seed_family_summary.jsonl.gz", "wt",
                   encoding="ascii") as fout:
        for idx, record in enumerate(parse_stockholm(_gz_lines(raw_dir / "Rfam.seed.gz"))):
            seqs = record["sequences"]
            stats["families"] += 1
            stats["sequences"] += len(seqs)
            fout.write(json.dumps({
                "record_index": idx,
                "n_sequences": len(seqs),
                "mean_length": (sum(len(s) for s in seqs.values()) / max(1, len(seqs))),
            }) + "\n")
    return stats


def drive_bprna(raw_dir: Path, out_dir: Path) -> Dict[str, int]:
    """Dot-bracket validity + seq/struct length match + exact dedup (UFold).

    Note: CD-HIT-EST 80% redundancy reduction requires the external CD-HIT
    binary; here we apply the validity filters and exact (100%) dedup, and
    record that 80% reduction is deferred to a CD-HIT install.
    """
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(raw_dir / "bprna_1m_data.parquet")
    seq_col = next((c for c in df.columns if c.lower() in
                    ("sequence", "seq", "rna")), df.columns[0])
    struct_col = next((c for c in df.columns if c.lower() in
                       ("structure", "dot_bracket", "dotbracket", "dbn")),
                      df.columns[1])
    stats = {"input": int(len(df)), REASON_KEPT: 0, REASON_BAD_STRUCTURE: 0,
             REASON_DUPLICATE: 0}
    seen: set = set()
    keep_idx: List[int] = []
    for idx, (seq, struct) in enumerate(zip(df[seq_col], df[struct_col])):
        if not bprna_record_ok(str(seq), str(struct)):
            stats[REASON_BAD_STRUCTURE] += 1
            continue
        h = content_hash(t_to_u(str(seq)))
        if h in seen:
            stats[REASON_DUPLICATE] += 1
            continue
        seen.add(h)
        stats[REASON_KEPT] += 1
        keep_idx.append(idx)
    df.iloc[keep_idx].to_parquet(out_dir / "bprna_1m.clean.parquet", index=False)
    stats["sequence_column"] = seq_col  # type: ignore[assignment]
    stats["structure_column"] = struct_col  # type: ignore[assignment]
    stats["cdhit_est_80_reduction"] = "deferred: external CD-HIT binary not installed"  # type: ignore[assignment]
    return stats


def drive_zip_archive(names: Sequence[str]):
    """Archive-integrity driver factory (mRNABERT / UTR-LM / E2Efold)."""

    def _drive(raw_dir: Path, out_dir: Path) -> Dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        stats: Dict[str, Any] = {}
        for fn in names:
            ok, n_members = zip_integrity_ok(raw_dir / fn)
            stats[fn] = {"ok": ok, "members": n_members}
            if not ok:
                stats.setdefault(REASON_ARCHIVE_CORRUPT, []).append(fn)
        return stats

    return _drive


def drive_sc2_viral(raw_dir: Path, out_dir: Path) -> Dict[str, Any]:
    """FASTA parse/count only; per-gene CDS extraction needs GBFF coordinates
    (the viral genomic/protein FASTA pair is not record-aligned), so the
    GenSLM translation verification is deferred until GBFF is acquired."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stats: Dict[str, Any] = {}
    for fn, key in (("viral.1.1.genomic.fna.gz", "genomic_records"),
                    ("viral.1.protein.faa.gz", "protein_records")):
        src = raw_dir / fn
        if not src.exists():
            stats[key] = "not downloaded"
            continue
        stats[key] = sum(1 for _ in parse_fasta(_gz_lines(src)))
    stats["translation_verification"] = (
        "deferred: genomic/protein FASTA not record-aligned; requires viral GBFF"
    )
    return stats


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def write_cleaning_report(out_dir: Path, dataset: str, protocol: str,
                          stats: Dict[str, Any]) -> Path:
    """Write ``cleaning_report.json`` (same audit role as clean_mrna stats)."""
    report = {
        "dataset_name": dataset,
        "protocol": protocol,
        "cleaning_spec": EXTERNAL_CATALOG[dataset]["cleaning_spec"],
        "citation": EXTERNAL_CATALOG[dataset]["citation"],
        "stats": stats,
    }
    path = out_dir / "cleaning_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# dataset -> (driver(raw_dir, out_dir) -> stats, protocol label)
DRIVERS: Dict[str, Tuple[Any, str]] = {
    "refseq_human_mrna_prot": (
        lambda r, o: drive_refseq_pairs(r, o, "human", range(1, 16)),
        "translation-consistency filter (LucaOne / Life-Code)",
    ),
    "refseq_mammalian_cds": (
        lambda r, o: drive_refseq_pairs(r, o, "vertebrate_mammalian", range(1, 140)),
        "translation-consistency filter (CaLM); CD-HIT 40% deferred",
    ),
    "sc2_viral_genomes": (drive_sc2_viral, "GenSLM (FASTA audit; GBFF deferred)"),
    "gencode_v50_full": (
        drive_gencode_v50,
        "canonical transcript + SpliceAI chromosome-disjoint split",
    ),
    "ensembl_human_grch38": (
        drive_ensembl,
        "transcript-ID pairing + exact dedup (UTR-LM / mRNA-LM)",
    ),
    "mrnabert_downstream_zenodo": (
        drive_zip_archive([
            "5UTR.zip", "3UTR.zip", "CDS.zip", "full_length.zip",
            "protein.zip", "Spliceator.zip", "te_ultra_full_length.zip",
        ]),
        "archive integrity only (author-cleaned labels untouched)",
    ),
    "optimus_5prime": (drive_optimus, "50-nt window + ACGU (Sample et al. 2019)"),
    "utr_lm_repo": (
        drive_zip_archive(["UTR-LM_main.zip"]),
        "archive integrity only; corpus derivation deferred",
    ),
    "gtex_v8_expression": (drive_gtex, "GCT parse + ID harmonise (TPM untouched)"),
    "rnacentral_active": (drive_rnacentral, "T->U + exact dedup (RNA-FM 100%)"),
    "rfam_seed": (drive_rfam, "Stockholm MSA parse (RNA-MSM)"),
    "bprna_hf": (drive_bprna, "dot-bracket validity + exact dedup (UFold)"),
    "e2efold_structure_repo": (
        drive_zip_archive(["e2efold_master.zip"]),
        "archive integrity only (author splits untouched)",
    ),
}


def run_cleaning(name: str, raw_root: Path, clean_root: Path) -> Dict[str, Any]:
    """Run one dataset's driver and write its ``cleaning_report.json``."""
    driver, protocol = DRIVERS[name]
    out_dir = clean_root / name
    stats = driver(raw_root / name, out_dir)
    report = write_cleaning_report(out_dir, name, protocol, stats)
    logger.info("report: %s", report)
    return stats


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m data.clean_external_catalog",
        description="Clean external catalog datasets per the source publications.",
    )
    parser.add_argument("--datasets", nargs="+", metavar="SEL",
                        help="Dataset names, priority tiers (p0/p1/p2), or 'all'.")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--clean-root", default="data/cleaned")
    parser.add_argument("--list", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    if args.list:
        for name, entry in EXTERNAL_CATALOG.items():
            print(f"{entry['priority']}  {name}: {entry['cleaning_spec'][:72]}...")
        return 0
    if not args.datasets:
        _build_parser().error("--datasets is required unless --list is given")
    names = select_datasets(args.datasets)
    raw_root = Path(args.raw_root)
    clean_root = Path(args.clean_root)
    failures: List[str] = []
    for name in names:
        logger.info("=== %s ===", name)
        if not (raw_root / name).is_dir():
            logger.error("SKIP %s: raw dir %s missing (download first)", name, raw_root / name)
            failures.append(name)
            continue
        try:
            run_cleaning(name, raw_root, clean_root)
        except Exception as exc:  # keep cleaning the remaining datasets
            logger.exception("FAILED %s: %s", name, exc)
            failures.append(name)
    if failures:
        logger.error("cleaning failures: %s", failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
