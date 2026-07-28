"""External catalog acquisition for mRNA-EditFlow (2026-07-27 data scale-up).

Downloads the P0/P1/P2 dataset catalog ranked in
``codonflow_integrated_dataset_catalog_ranked.xlsx`` (mRNA-related training
corpora from published work) into ``data/raw/<dataset_name>/`` with per-file
SHA-256 verification and a ``manifest.json`` per dataset (same convention as
``data/raw/sample2019_mpra/manifest.json``).

Design rules (project data governance):

* URLs live in the static :data:`EXTERNAL_CATALOG` registry only; nothing is
  fetched at import time or during unit tests.
* Raw files are written read-only evidence: no cleaning happens here. Cleaning
  per the source publication lives in ``data/clean_external_catalog.py``.
* Reuses the streaming downloader + SHA256 + checksum cache from
  :mod:`data.download_mrna` so integrity semantics are identical to the
  existing corpora.

CLI::

    python3 -m data.download_external_catalog --list
    python3 -m data.download_external_catalog --datasets p0
    python3 -m data.download_external_catalog --datasets refseq_human_mrna_prot
    python3 -m data.download_external_catalog --datasets all --force
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from data.download_mrna import (
    _CHECKSUM_CACHE_FILENAME,
    _compute_sha256,
    _download_file,
    load_cached_checksums,
)

logger = logging.getLogger(__name__)

PRIORITIES = ("P0", "P1", "P2")

_NCBI = "https://ftp.ncbi.nlm.nih.gov"
_EBI = "https://ftp.ebi.ac.uk/pub/databases"
_ENSEMBL = "https://ftp.ensembl.org/pub"
_GCP_GTEX = "https://storage.googleapis.com/adult-gtex"


def _refseq_human_files() -> List[Dict[str, str]]:
    """RefSeq H. sapiens mRNA_Prot shards 1..15 (rna.fna + protein.faa)."""
    files: List[Dict[str, str]] = []
    base = f"{_NCBI}/refseq/H_sapiens/mRNA_Prot"
    for i in range(1, 16):
        for kind in ("rna.fna", "protein.faa"):
            fn = f"human.{i}.{kind}.gz"
            files.append({"url": f"{base}/{fn}", "filename": fn})
    return files


def _refseq_mammalian_files() -> List[Dict[str, str]]:
    """RefSeq vertebrate_mammalian release shards 1..139 (rna.fna + protein.faa).

    Shard counts verified against the FTP listing on 2026-07-27 (139 rna.fna
    shards, 139 protein.faa shards).
    """
    files: List[Dict[str, str]] = []
    base = f"{_NCBI}/refseq/release/vertebrate_mammalian"
    for i in range(1, 140):
        for kind in ("rna.fna", "protein.faa"):
            fn = f"vertebrate_mammalian.{i}.{kind}.gz"
            files.append({"url": f"{base}/{fn}", "filename": fn})
    return files


# ---------------------------------------------------------------------------
# Static external catalog registry (URLs only; never fetched in tests)
# ---------------------------------------------------------------------------
EXTERNAL_CATALOG: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------ P0
    "refseq_human_mrna_prot": {
        "priority": "P0",
        "description": (
            "RefSeq Homo sapiens mRNA + protein paired shards (human.1..15). "
            "Provides CDS<->protein pairs for translation-consistency checks "
            "and back-translation supervision (LucaOne / Life-Code protocol)."
        ),
        "citation": (
            "O'Leary NA et al. Nucleic Acids Res. 2016;44(D1):D733-D745. "
            "DOI:10.1093/nar/gkv1189. RefSeq release (downloaded 2026-07)."
        ),
        "license": "US Government public domain (NCBI RefSeq)",
        "cleaning_spec": (
            "Translate CDS with the standard codon table and require exact "
            "match to the paired protein record; drop inconsistent pairs "
            "(LucaOne / Life-Code protocol)."
        ),
        "files": _refseq_human_files(),
    },
    "gencode_v50_full": {
        "priority": "P0",
        "description": (
            "GENCODE v50 human full annotation: GTF, protein-coding "
            "transcripts, all transcripts, polyA annotation. Fixed release "
            "for reproducible splice/UTR ground truth."
        ),
        "citation": (
            "Frankish A et al. Nucleic Acids Res. 2023;51(D1):D933-D941. "
            "DOI:10.1093/nar/gkac1057. GENCODE v50."
        ),
        "license": "EMBL-EBI terms of use (open)",
        "cleaning_spec": (
            "Split train/test by chromosome; keep canonical transcripts; "
            "rebuild exon/intron/UTR/CDS masks from GTF (SpliceAI / 3UTRBERT "
            "protocol)."
        ),
        "files": [
            {"url": f"{_EBI}/gencode/Gencode_human/release_50/gencode.v50.annotation.gtf.gz",
             "filename": "gencode.v50.annotation.gtf.gz"},
            {"url": f"{_EBI}/gencode/Gencode_human/release_50/gencode.v50.pc_transcripts.fa.gz",
             "filename": "gencode.v50.pc_transcripts.fa.gz"},
            {"url": f"{_EBI}/gencode/Gencode_human/release_50/gencode.v50.transcripts.fa.gz",
             "filename": "gencode.v50.transcripts.fa.gz"},
            {"url": f"{_EBI}/gencode/Gencode_human/release_50/gencode.v50.polyAs.gff3.gz",
             "filename": "gencode.v50.polyAs.gff3.gz"},
        ],
    },
    "ensembl_human_grch38": {
        "priority": "P0",
        "description": (
            "Ensembl GRCh38 human cDNA / CDS / protein FASTA + GFF3 (release "
            "116 filenames verified 2026-07-27). Complements GENCODE with "
            "Ensembl-coordinate transcripts."
        ),
        "citation": (
            "Cunningham F et al. Nucleic Acids Res. 2022;50(D1):D988-D995. "
            "DOI:10.1093/nar/gkab1049. Ensembl release 116."
        ),
        "license": "Ensembl terms of use (open)",
        "cleaning_spec": (
            "Unify coordinates, deduplicate, verify CDS translation "
            "(UTR-LM / mRNA-LM protocol)."
        ),
        "files": [
            {"url": f"{_ENSEMBL}/current_fasta/homo_sapiens/cdna/Homo_sapiens.GRCh38.cdna.all.fa.gz",
             "filename": "Homo_sapiens.GRCh38.cdna.all.fa.gz"},
            {"url": f"{_ENSEMBL}/current_fasta/homo_sapiens/cds/Homo_sapiens.GRCh38.cds.all.fa.gz",
             "filename": "Homo_sapiens.GRCh38.cds.all.fa.gz"},
            {"url": f"{_ENSEMBL}/current_fasta/homo_sapiens/pep/Homo_sapiens.GRCh38.pep.all.fa.gz",
             "filename": "Homo_sapiens.GRCh38.pep.all.fa.gz"},
            {"url": f"{_ENSEMBL}/current_gff3/homo_sapiens/Homo_sapiens.GRCh38.116.chr.gff3.gz",
             "filename": "Homo_sapiens.GRCh38.116.chr.gff3.gz"},
        ],
    },
    "refseq_mammalian_cds": {
        "priority": "P0",
        "description": (
            "RefSeq vertebrate_mammalian release: 139 rna.fna + 139 "
            "protein.faa shards (multi-species mRNA/CDS corpus, "
            "CodonBERT-scale). Enables cross-species codon-usage priors."
        ),
        "citation": (
            "O'Leary NA et al. Nucleic Acids Res. 2016;44(D1):D733-D745. "
            "DOI:10.1093/nar/gkv1189. RefSeq release (downloaded 2026-07)."
        ),
        "license": "US Government public domain (NCBI RefSeq)",
        "cleaning_spec": (
            "Translation-consistency filter; per-species codon-usage "
            "statistics; redundancy reduction at 40% identity "
            "(CaLM protocol, CD-HIT-EST)."
        ),
        "files": _refseq_mammalian_files(),
    },
    # ------------------------------------------------------------------ P1
    "mrnabert_downstream_zenodo": {
        "priority": "P1",
        "description": (
            "mRNABERT downstream task datasets (Zenodo record 17786045): "
            "5UTR / 3UTR / CDS / full_length / protein / Spliceator / "
            "te_ultra_full_length. Closest public resource to therapeutic "
            "mRNA design evaluation."
        ),
        "citation": (
            "mRNABERT authors. Zenodo record 17786045. "
            "https://zenodo.org/records/17786045 (CC-BY-4.0)."
        ),
        "license": "CC-BY-4.0",
        "cleaning_spec": (
            "Author-cleaned downstream splits; verify archive integrity and "
            "convert to region-labelled JSONL without altering labels."
        ),
        "files": [
            {"url": f"https://zenodo.org/api/records/17786045/files/{name}.zip/content",
             "filename": f"{name}.zip"}
            for name in (
                "5UTR", "3UTR", "CDS", "full_length", "protein",
                "Spliceator", "te_ultra_full_length",
            )
        ],
    },
    "optimus_5prime": {
        "priority": "P1",
        "description": (
            "Optimus 5-Prime (Sample et al. 2019) author repo archive + "
            "processed SNV phenotype table (~280k synthetic 5'UTR designs "
            "with polysome readout). Complements the GEO raw MPRA already "
            "held under sample2019_mpra."
        ),
        "citation": (
            "Sample PJ et al. Nat Biotechnol. 2019;37(7):803-809. "
            "DOI:10.1038/s41587-019-0164-5."
        ),
        "license": "GPL-3.0 (repo); data per GEO terms",
        "cleaning_spec": (
            "Use author-processed polysome readout at fixed 50-nt 5'UTR "
            "window; tag source to keep distinct from raw GEO MPRA cohort."
        ),
        "files": [
            {"url": "https://github.com/pjsample/human_5utr_modeling/archive/refs/heads/master.zip",
             "filename": "human_5utr_modeling_master.zip"},
            {"url": "https://raw.githubusercontent.com/pjsample/human_5utr_modeling/master/human_5utrs/data/snv_phenotype_log_diff.csv",
             "filename": "snv_phenotype_log_diff.csv"},
        ],
    },
    "utr_lm_repo": {
        "priority": "P1",
        "description": (
            "UTR-LM author repo (code + downstream assets). The 214,349 "
            "endogenous 5'UTR corpus is derived from Ensembl multi-species "
            "per paper Methods; derivation happens in the cleaning stage "
            "from the Ensembl/GENCODE corpora above."
        ),
        "citation": (
            "Chu Y, Yu D, Li Y, et al. Nat Mach Intell. 2024;6:449-460. "
            "DOI:10.1038/s42256-024-00823-9."
        ),
        "license": "GPL-3.0",
        "cleaning_spec": (
            "Derive 5'UTR sequences from Ensembl multi-species annotations "
            "with the paper's length filters; record species list from paper "
            "Methods (no guessing)."
        ),
        "files": [
            {"url": "https://github.com/a96123155/UTR-LM/archive/refs/heads/main.zip",
             "filename": "UTR-LM_main.zip"},
        ],
    },
    "gtex_v8_expression": {
        "priority": "P1",
        "description": (
            "GTEx V8 bulk RNA-seq: gene median TPM, gene TPM matrix, "
            "transcript TPM matrix, sample attributes. Tissue-expression "
            "conditioning and evaluation resource (BigRNA / Enformer / "
            "Borzoi protocol)."
        ),
        "citation": (
            "GTEx Consortium. Science. 2020;369(6509):1318-1330. "
            "DOI:10.1126/science.aaz1776. GTEx V8."
        ),
        "license": "GTEx open-access data (dbGaP not required for these files)",
        "cleaning_spec": (
            "Harmonise gene/transcript IDs, keep TPM normalisation as "
            "published, align transcript IDs to GENCODE/Ensembl."
        ),
        "files": [
            {"url": f"{_GCP_GTEX}/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz",
             "filename": "GTEx_v8_gene_median_tpm.gct.gz"},
            {"url": f"{_GCP_GTEX}/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz",
             "filename": "GTEx_v8_gene_tpm.gct.gz"},
            {"url": f"{_GCP_GTEX}/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RSEMv1.3.0_transcript_tpm.gct.gz",
             "filename": "GTEx_v8_transcript_tpm.gct.gz"},
            {"url": f"{_GCP_GTEX}/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt",
             "filename": "GTEx_v8_sample_attributes.txt"},
        ],
    },
    "sc2_viral_genomes": {
        "priority": "P1",
        "description": (
            "NCBI RefSeq viral release: complete viral genomic FASTA + viral "
            "protein FASTA (includes SARS-CoV-2 reference set). Viral RNA / "
            "codon sub-task and robustness evaluation (GenSLM protocol)."
        ),
        "citation": (
            "O'Leary NA et al. Nucleic Acids Res. 2016;44(D1):D733-D745. "
            "DOI:10.1093/nar/gkv1189. RefSeq viral release (2026-07)."
        ),
        "license": "US Government public domain (NCBI RefSeq)",
        "cleaning_spec": (
            "Extract per-gene CDS with codon grouping; translation "
            "verification against viral protein FASTA (GenSLM protocol)."
        ),
        "files": [
            {"url": f"{_NCBI}/refseq/release/viral/viral.1.1.genomic.fna.gz",
             "filename": "viral.1.1.genomic.fna.gz"},
            {"url": f"{_NCBI}/refseq/release/viral/viral.1.protein.faa.gz",
             "filename": "viral.1.protein.faa.gz"},
        ],
    },
    # ------------------------------------------------------------------ P2
    "rnacentral_active": {
        "priority": "P2",
        "description": (
            "RNAcentral active sequences FASTA (all ncRNA, current release). "
            "RNA-prior / representation pretraining corpus (RNA-FM / "
            "RiNALMo / AIDO.RNA protocol)."
        ),
        "citation": (
            "RNAcentral Consortium. Nucleic Acids Res. 2023;51(D1):D272-D278. "
            "DOI:10.1093/nar/gkac961."
        ),
        "license": "RNAcentral terms (open, CC0-style aggregation)",
        "cleaning_spec": (
            "T->U normalisation, exact dedup (RNA-FM used CD-HIT-EST 100% "
            "identity), length filters per downstream task; strictly keep "
            "ncRNA separate from CDS corpora."
        ),
        "files": [
            {"url": f"{_EBI}/RNAcentral/current_release/sequences/rnacentral_active.fasta.gz",
             "filename": "rnacentral_active.fasta.gz"},
        ],
    },
    "rfam_seed": {
        "priority": "P2",
        "description": (
            "Rfam CURRENT seed alignments (Stockholm), full-region table and "
            "clan info. RNA family / homology prior and MSA-based evaluation "
            "(RNA-MSM protocol)."
        ),
        "citation": (
            "Kalvari I et al. Nucleic Acids Res. 2021;49(D1):D192-D200. "
            "DOI:10.1093/nar/gkaa1047. Rfam CURRENT (2026-07)."
        ),
        "license": "Rfam terms (open, CC0)",
        "cleaning_spec": (
            "Parse Stockholm MSAs per family; control homology leakage in "
            "splits (RNA-MSM protocol)."
        ),
        "files": [
            {"url": f"{_EBI}/Rfam/CURRENT/Rfam.seed.gz",
             "filename": "Rfam.seed.gz"},
            {"url": f"{_EBI}/Rfam/CURRENT/Rfam.full_region.gz",
             "filename": "Rfam.full_region.gz"},
            {"url": f"{_EBI}/Rfam/CURRENT/Rfam.clanin",
             "filename": "Rfam.clanin"},
        ],
    },
    "bprna_hf": {
        "priority": "P2",
        "description": (
            "bpRNA-1m secondary-structure dataset (multimolecule parquet "
            "mirror, 102,318 sequences). Structure-rationality evaluation "
            "for UTR/mRNA designs (UFold protocol)."
        ),
        "citation": (
            "Danaee P et al. Genome Biol. 2018;19:46. "
            "DOI:10.1186/s13059-018-1422-4. Mirror: HF multimolecule/bprna."
        ),
        "license": "bpRNA terms (academic); mirror CC-BY-4.0",
        "cleaning_spec": (
            "CD-HIT-EST 80% redundancy reduction; dot-bracket validity "
            "checks (UFold protocol)."
        ),
        "files": [
            {"url": "https://hf-mirror.com/datasets/multimolecule/bprna/resolve/main/data.parquet",
             "filename": "bprna_1m_data.parquet"},
        ],
    },
    "e2efold_structure_repo": {
        "priority": "P2",
        "description": (
            "E2Efold author repo archive with preprocessing scripts for "
            "RNAStralign (~30,451) + ArchiveII (3,975) secondary-structure "
            "benchmarks."
        ),
        "citation": (
            "Chen X, Li Y, Umarov R, Gao X, Song L. ICLR 2020. "
            "arXiv:2002.05810."
        ),
        "license": "MIT",
        "cleaning_spec": (
            "Standard RNAStralign/ArchiveII benchmark splits as shipped by "
            "the authors."
        ),
        "files": [
            {"url": "https://github.com/ml4bio/e2efold/archive/refs/heads/master.zip",
             "filename": "e2efold_master.zip"},
        ],
    },
}


# ---------------------------------------------------------------------------
# Selection / download / manifest
# ---------------------------------------------------------------------------
def select_datasets(selector: Sequence[str]) -> List[str]:
    """Resolve a CLI selector list into dataset names.

    Accepts explicit dataset names, case-insensitive priority tiers
    (``p0``/``p1``/``p2``) or ``all``. Duplicates are removed preserving
    registry order.
    """
    if not selector:
        raise ValueError("Empty selector; pass dataset names, a priority tier, or 'all'.")
    lowered = {s.lower() for s in selector}
    # normalise: explicit names are matched case-insensitively
    name_by_lower = {n.lower(): n for n in EXTERNAL_CATALOG}
    unknown = {s for s in lowered if s not in name_by_lower and s not in {p.lower() for p in PRIORITIES} and s != "all"}
    if unknown:
        raise ValueError(f"Unknown dataset selector(s): {sorted(unknown)}")
    if "all" in lowered:
        return list(EXTERNAL_CATALOG)
    chosen: List[str] = []
    for name, entry in EXTERNAL_CATALOG.items():
        if name.lower() in lowered or entry["priority"].lower() in lowered:
            chosen.append(name)
    return chosen


def write_manifest(dataset_dir: Path, name: str, entry: Dict[str, Any],
                   file_records: List[Dict[str, Any]]) -> Path:
    """Write ``manifest.json`` for a downloaded dataset. Returns its path."""
    manifest = {
        "dataset_name": name,
        "priority": entry["priority"],
        "description": entry["description"],
        "citation": entry["citation"],
        "license": entry["license"],
        "cleaning_spec": entry["cleaning_spec"],
        "acquisition_date_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "n_files": len(file_records),
        "total_bytes": sum(r["byte_size"] for r in file_records),
        "files": file_records,
    }
    path = dataset_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def download_catalog_dataset(name: str, target_root: str = "data/raw",
                             force: bool = False) -> Dict[str, Any]:
    """Download one registered dataset; never called in unit tests."""
    if name not in EXTERNAL_CATALOG:
        raise ValueError(f"Unknown dataset {name!r}. Available: {list(EXTERNAL_CATALOG)}")
    entry = EXTERNAL_CATALOG[name]
    dataset_dir = Path(target_root) / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    cache_path = dataset_dir / _CHECKSUM_CACHE_FILENAME
    cached = load_cached_checksums(str(dataset_dir))
    file_records: List[Dict[str, Any]] = []
    for f in entry["files"]:
        dest = dataset_dir / f["filename"]
        _download_file(f["url"], dest, force=force,
                       expected_sha256=cached.get(f["filename"]),
                       checksum_cache_path=cache_path)
        file_records.append({
            "filename": f["filename"],
            "url": f["url"],
            "sha256": _compute_sha256(dest),
            "byte_size": dest.stat().st_size,
        })
    manifest_path = write_manifest(dataset_dir, name, entry, file_records)
    logger.info("Wrote manifest %s (%d files)", manifest_path, len(file_records))
    return json.loads(manifest_path.read_text())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m data.download_external_catalog",
        description="Download the P0/P1/P2 external mRNA dataset catalog.",
    )
    parser.add_argument("--datasets", nargs="+", metavar="SEL",
                        help="Dataset names, priority tiers (p0/p1/p2), or 'all'.")
    parser.add_argument("--target-root", default="data/raw",
                        help="Root directory for raw datasets (default: data/raw).")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even when files are present.")
    parser.add_argument("--list", action="store_true",
                        help="List registered datasets and exit.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    if args.list:
        for name, entry in EXTERNAL_CATALOG.items():
            print(f"{entry['priority']}  {name}  ({len(entry['files'])} files)")
        return 0
    if not args.datasets:
        _build_parser().error("--datasets is required unless --list is given")
    names = select_datasets(args.datasets)
    failures: Dict[str, str] = {}
    for name in names:
        logger.info("=== %s (%s) ===", name, EXTERNAL_CATALOG[name]["priority"])
        try:
            download_catalog_dataset(name, target_root=args.target_root, force=args.force)
        except Exception as exc:  # keep going; report at end
            logger.error("FAILED %s: %s", name, exc)
            failures[name] = str(exc)
    if failures:
        for name, err in failures.items():
            print(f"FAILED\t{name}\t{err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
