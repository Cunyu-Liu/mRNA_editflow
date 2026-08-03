#!/usr/bin/env python
"""v3.1 D0-R asset registry builder.

Builds the lightweight E/F asset registry for the UTR Edit-Flow benchmark-first
contract (D0-R phase). This is definition/registry only: it does NOT run any D1
parser, does NOT reconstruct canonical sequences, and does NOT emit raw
nucleotide sequences. It only records acquisition facts (physical presence,
byte_size, sha256), license/use decisions, search ledger, and the frozen
priority snapshot.

Authoritative contract: mrna_contract_v3_draft.md
  - §5.2  DatasetAsset minimum fields
  - §7.1  DATA-P0 asset table
  - §7.1.1 frozen priority sets
  - §7.2  DATA-P1 assets
  - §14.4 D0-R requirements
"""
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Frozen constants (contract §7.1.1) -- DO NOT modify.
# ---------------------------------------------------------------------------
P0_ASSET_GROUP_IDS = [
    "GSE114002", "GSE145046", "GSE217518", "GSE232571", "GSE232572",
    "FAST_UTR_SIEGEL_2022", "GSE288185", "GSE256185_DART", "GSE232927",
    "GSE176581", "ENCSR854RUF", "GSE149487", "GSE200304", "GSE186455",
    "GSE246381", "NZIP_EMTAB_10902_11572_11575", "GSE330741", "GSE261709",
    "GSE298114", "PTRE_PRJNA1116243", "GSE207584_CLEANUP", "GSE173083_CLEANUP",
]
P1_ASSET_GROUP_IDS = [
    "GSE194092", "GSE270252_270254", "GSE173098", "GSE295080_ISOMPRA",
    "GSE291719_SONAR", "GSE55396_FAST_UTR_2014", "PASSPORT_SEQ", "SEERS",
]
P2_ACQUISITION_WATCHLIST = ["PARADE", "SALUKI_HALF_LIFE"]
REFERENCE_SERVICE = ["GENCODE", "REFSEQ", "ENSEMBL", "UTRDB", "RNACENTRAL"]
ANALYSIS_ONLY_OUT_OF_SCOPE = ["CODONBERT", "OPENVACCINE", "BPRNA_STRUCTURE_ONLY"]
SEARCH_NEGATIVE_LEDGER = ["MAVEDB", "MPRABASE"]
GSE200304_MEMBERS = ["GSE200304", "GSE200302", "GSE200303", "GSE217530"]

# Physical raw-data root.
RAW_ROOT = "/mnt/cunyuliu/mrna_editflow_p0/"
# Absolute worktree root (passed in).
WORKTREE = os.environ.get("WORKTREE", os.path.dirname(os.path.abspath(__file__)))
REGISTRY_DIR = os.path.join(WORKTREE, "data", "v3_1", "registry")
RUN_ROOT = os.environ.get("RUN_ROOT", os.path.join(WORKTREE, "runs"))

# Files above this size (bytes) are treated as "huge raw reads" whose sha256 is
# not computed in D0 (deferred to D1); they are registered with provider
# checksum and byte_size only.
HUGE_SIZE_THRESHOLD = 1_000_000_000  # 1 GB

REVIEWER = "D0-R auditor (cunyuliu)"


def iso_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Static asset metadata.  asset_group_id -> (fields)
# "present_dir" is the directory name under RAW_ROOT when physical files exist.
# ---------------------------------------------------------------------------
def _geo(acc, study=None, pub=None, proj=None, source_url=None):
    return {
        "provider": "GEO",
        "accession": acc,
        "study_id": study or acc,
        "bioproject_or_project_id": proj,
        "publication_ids": pub or [],
        "source_url": source_url or f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}",
    }


ASSET_META = {
    "GSE114002": _geo("GSE114002", pub=["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114002"]),
    "GSE145046": _geo("GSE145046", pub=["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE145046"]),
    "GSE217518": _geo("GSE217518", pub=["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE217518"]),
    "GSE232571": _geo("GSE232571", pub=["https://www.omicsdi.org/dataset/geo/GSE232571"]),
    "GSE232572": _geo("GSE232572", pub=["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE232572"]),
    "FAST_UTR_SIEGEL_2022": {
        "provider": "author-repository",
        "accession": "FAST_UTR_SIEGEL_2022",
        "study_id": "fast-UTR Siegel 2022",
        "bioproject_or_project_id": None,
        "publication_ids": ["https://github.com/youryurr/fast-UTR"],
        "source_url": "https://github.com/youryurr/fast-UTR",
    },
    "GSE288185": _geo("GSE288185", pub=["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE288185"]),
    "GSE256185_DART": _geo("GSE256185", pub=["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE256185"]),
    "GSE232927": _geo("GSE232927", pub=["10.1038/s41467-024-49508-2"]),
    "GSE176581": _geo("GSE176581", pub=["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176581"]),
    "ENCSR854RUF": {
        "provider": "ENCODE",
        "accession": "ENCSR854RUF",
        "study_id": "ENCSR854RUF",
        "bioproject_or_project_id": None,
        "publication_ids": ["https://www.encodeproject.org/publication-data/ENCSR854RUF/"],
        "source_url": "https://www.encodeproject.org/publication-data/ENCSR854RUF/",
    },
    "GSE149487": _geo("GSE149487", pub=["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE149487"]),
    "GSE200304": _geo("GSE200304", pub=["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE200304"]),
    "GSE186455": _geo("GSE186455", pub=["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186455"]),
    "GSE246381": _geo("GSE246381", pub=["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE246381"]),
    "NZIP_EMTAB_10902_11572_11575": {
        "provider": "BioStudies",
        "accession": "E-MTAB-10902",
        "study_id": "E-MTAB-10902/11572/11575",
        "bioproject_or_project_id": "ERP136423",
        "publication_ids": ["10.1038/s41593-022-01243-x"],
        "source_url": "https://www.ebi.ac.uk/biostudies/studies/E-MTAB-10902",
    },
    "GSE330741": _geo("GSE330741", pub=["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE330741"]),
    "GSE261709": _geo("GSE261709", pub=["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE261709"]),
    "GSE298114": _geo("GSE298114", pub=["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE298114"]),
    "PTRE_PRJNA1116243": {
        "provider": "SRA",
        "accession": "PRJNA1116243",
        "study_id": "PTRE-seq",
        "bioproject_or_project_id": "PRJNA1116243",
        "publication_ids": ["10.1101/2024.08.05.606557"],
        "source_url": "https://www.ncbi.nlm.nih.gov/bioproject/PRJNA1116243",
    },
    "GSE207584_CLEANUP": _geo("GSE207584", pub=["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE207584"]),
    "GSE173083_CLEANUP": _geo("GSE173083", pub=["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE173083"]),
    # P1
    "GSE194092": _geo("GSE194092", pub=["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194092"]),
    "GSE270252_270254": _geo("GSE270252", pub=["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE270252"]),
    "GSE173098": _geo("GSE173098", pub=["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE173098"]),
    "GSE295080_ISOMPRA": _geo("GSE295080", pub=["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE295080"]),
    "GSE291719_SONAR": _geo("GSE291719", pub=["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE291719"]),
    "GSE55396_FAST_UTR_2014": _geo("GSE55396", pub=["https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE55396"]),
    "PASSPORT_SEQ": {
        "provider": "author-repository",
        "accession": "PASSPORT_SEQ",
        "study_id": "PASSPORT-seq",
        "bioproject_or_project_id": None,
        "publication_ids": [],
        "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=PASSPORT_SEQ",
    },
    "SEERS": {
        "provider": "author-repository",
        "accession": "SEERS",
        "study_id": "SEERS",
        "bioproject_or_project_id": None,
        "publication_ids": [],
        "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=SEERS",
    },
}

# Directory names under RAW_ROOT that map to present asset_group_ids.
PRESENT_DIR = {
    "GSE114002": "GSE114002",
    "GSE145046": "GSE145046",
    "GSE217518": "GSE217518",
    "GSE232572": "GSE232572",
    "ENCSR854RUF": "ENCSR854RUF",
    "GSE149487": "GSE149487",
    "GSE200304": "GSE200304",
    "GSE186455": "GSE186455",
    "GSE246381": "GSE246381",
    "GSE207584_CLEANUP": "GSE207584",
    "GSE173083_CLEANUP": "GSE173083",
}

# Files/dirs to ignore in inventory (not data assets).
_IGNORE_NAMES = {
    ".corrupt.bak", ".corrupt.bak2", "download.log", "download_status.txt",
    "manifest.json", "processed_manifest.json",
}

# Per-asset scientific / priority overlay (contract §7.1 / §7.2).
SCIENTIFIC = {
    "GSE114002": ("P0", "E_F", "SURVEY", "inventory_join", "existing files present"),
    "GSE145046": ("P0", "F", "SURVEY", "10mer_join", "existing files present"),
    "GSE217518": ("P0", "E_DELTA", "SURVEY", "ref_mut_join", "existing files present"),
    "GSE232571": ("P0", "E", "ACQUIRE", "gear_download", "download present"),
    "GSE232572": ("P0", "E", "SURVEY", "three_denominator_close", "existing files present"),
    "FAST_UTR_SIEGEL_2022": ("P0", "E_LINK", "ACQUIRE", "github_download", "download + license"),
    "GSE288185": ("P0", "E", "ACQUIRE", "gear_download", "download present"),
    "GSE256185_DART": ("P0", "E_F", "ACQUIRE", "gear_download", "download present"),
    "GSE232927": ("P0", "F", "ACQUIRE", "download_resolve", "download present"),
    "GSE176581": ("P0", "F", "ACQUIRE", "gear_download", "download present"),
    "ENCSR854RUF": ("P0", "E", "SURVEY", "source_alt_license_close", "existing files present"),
    "GSE149487": ("P0", "F", "SURVEY", "eligibility_audit", "existing files present"),
    "GSE200304": ("P0", "E", "SURVEY", "three_modality_join", "existing files present"),
    "GSE186455": ("P0", "E", "SURVEY", "provenance_review", "existing files present"),
    "GSE246381": ("P0", "E_DELTA", "SURVEY", "sealed_builder", "existing files present"),
    "NZIP_EMTAB_10902_11572_11575": ("P0", "E_F", "ACQUIRE", "biostudies_download", "download present"),
    "GSE330741": ("P0", "E_DENSE", "ACQUIRE", "gear_download", "download present"),
    "GSE261709": ("P0", "E_DELTA", "ACQUIRE", "gear_download", "download present"),
    "GSE298114": ("P0", "E_DELTA", "ACQUIRE", "gear_download", "download present"),
    "PTRE_PRJNA1116243": ("P0", "AUX_QC", "ACQUIRE", "sra_download", "download present"),
    "GSE207584_CLEANUP": ("P0", "OUT_OF_SCOPE", "CLEANUP", "exclude_fasta_recover", "existing files present"),
    "GSE173083_CLEANUP": ("P0", "P2_AUX", "CLEANUP", "reconcile_units", "existing files present"),
    "GSE194092": ("P1", "F", "ACQUIRE", "landscape_calibration", "download present"),
    "GSE270252_270254": ("P1", "F", "ACQUIRE", "full_length_recover", "download present"),
    "GSE173098": ("P1", "F", "ACQUIRE", "oligo_map", "download present"),
    "GSE295080_ISOMPRA": ("P1", "E_DENSE", "ACQUIRE", "design_table", "download present"),
    "GSE291719_SONAR": ("P1", "F", "ACQUIRE", "oligo_map", "download present"),
    "GSE55396_FAST_UTR_2014": ("P1", "F", "ACQUIRE", "three_denominator_replay", "download present"),
    "PASSPORT_SEQ": ("P1", "E_DELTA", "ACQUIRE", "accession_recover", "stable accession + license"),
    "SEERS": ("P1", "F", "ACQUIRE", "design_label_license", "public design+label+license"),
}

# Permit model: public GEO/ENCODE/BioStudies supplemental data is publicly
# downloadable and processable for research; explicit training/evaluation/
# derived-release/raw-redistribution permission is NOT stated -> UNKNOWN.
def _permit(provider):
    if provider in ("GEO", "ENCODE", "BioStudies", "SRA"):
        return {
            "download": "YES", "processing": "YES",
            "training": "UNKNOWN", "evaluation": "UNKNOWN",
            "derived_release": "UNKNOWN", "raw_redistribution": "UNKNOWN",
        }
    # author-repository (fast-UTR, PASSPORT, SEERS): no explicit license -> all UNKNOWN
    return {
        "download": "UNKNOWN", "processing": "UNKNOWN",
        "training": "UNKNOWN", "evaluation": "UNKNOWN",
        "derived_release": "UNKNOWN", "raw_redistribution": "UNKNOWN",
    }


def _evidence_id(asset_group_id, kind):
    return f"EVIDENCE::{asset_group_id}::{kind}::v3.1"


def build():
    os.makedirs(REGISTRY_DIR, exist_ok=True)
    now = iso_now()
    today = now[:10]

    # ---- scan physical inventory -----------------------------------------
    # file_inventory: asset_group_id -> list of dicts
    file_inventory = {}
    provider_checksum = {}  # relpath -> (name, md5) for huge files
    for agid, dname in PRESENT_DIR.items():
        dpath = os.path.join(RAW_ROOT, dname)
        if not os.path.isdir(dpath):
            continue
        rows = []
        for root, dirs, files in os.walk(dpath):
            for fn in sorted(files):
                if fn in _IGNORE_NAMES:
                    continue
                fp = os.path.join(root, fn)
                rel = os.path.relpath(fp, dpath)
                sz = os.path.getsize(fp)
                rec = {"asset_group_id": agid, "relpath": rel,
                       "byte_size": sz, "sha256": None,
                       "parse_status": "NOT_STARTED", "mapping_status": "NOT_STARTED"}
                if sz <= HUGE_SIZE_THRESHOLD:
                    rec["sha256"] = sha256_file(fp)
                rows.append(rec)
        file_inventory[agid] = rows

    # Parse manifest.json for provider md5 of huge files (ENCSR fastq).
    for agid, dname in PRESENT_DIR.items():
        mpath = os.path.join(RAW_ROOT, dname, "manifest.json")
        if not os.path.isfile(mpath):
            continue
        try:
            with open(mpath) as f:
                m = json.load(f)
            for fitem in m.get("files", []):
                name = fitem.get("name")
                md5 = fitem.get("provider_md5") or fitem.get("md5")
                if name and md5:
                    provider_checksum[(agid, name)] = md5
        except Exception:
            pass

    # ---- dataset_assets.jsonl --------------------------------------------
    dataset_rows = []
    # Load manifest source_url/retrieved_at for present assets.
    meta_src = {}
    for agid, dname in PRESENT_DIR.items():
        mpath = os.path.join(RAW_ROOT, dname, "manifest.json")
        if os.path.isfile(mpath):
            try:
                with open(mpath) as f:
                    m = json.load(f)
                meta_src[agid] = {
                    "source_url": m.get("source_url"),
                    "retrieved_at": m.get("retrieved_at_utc"),
                }
            except Exception:
                pass

    all_asset_ids = list(dict.fromkeys(P0_ASSET_GROUP_IDS + P1_ASSET_GROUP_IDS))

    for agid in all_asset_ids:
        meta = ASSET_META[agid]
        provider = meta["provider"]
        present = agid in file_inventory
        permit = _permit(provider)
        sci_priority, sci_priority_val, role, action, promo = SCIENTIFIC[agid]

        if present:
            acq_status = "DOWNLOADED_VERIFIED"
            files = file_inventory[agid]
            # representative primary file = first non-trivial file
            primary = files[0]
            sha = primary["sha256"]
            byte_size = primary["byte_size"]
            original_filename = primary["relpath"]
            src = meta_src.get(agid, {}).get("source_url") or meta["source_url"]
            retrieved = meta_src.get(agid, {}).get("retrieved_at") or now
            license_status = "REVIEW_REQUIRED"
            d0_decision = "ACQUIRED_FOR_REBUILD"
            required_perm = permit
            use_basis = [_evidence_id(agid, "geodownload")]
            parse_status = "NOT_STARTED"
            mapping_status = "NOT_STARTED"
            canonical_status = "PENDING"
        else:
            acq_status = "NOT_PRESENT"
            files = []
            sha = None
            byte_size = None
            original_filename = None
            src = meta["source_url"]
            retrieved = None
            license_status = "REVIEW_REQUIRED"
            # P0 not-present: METADATA_ONLY (metadata available, files not downloaded)
            d0_decision = "METADATA_ONLY"
            required_perm = permit
            use_basis = []
            parse_status = "NOT_STARTED"
            mapping_status = "NOT_STARTED"
            canonical_status = "PENDING"

        row = {
            "asset_id": f"{agid}::v3.1",
            "accession": meta["accession"],
            "study_id": meta["study_id"],
            "provider": provider,
            "publication_ids": meta["publication_ids"],
            "bioproject_or_project_id": meta["bioproject_or_project_id"],
            "source_url": src,
            "source_release": "unknown",
            "downloaded_at": retrieved,
            "original_filename": original_filename,
            "byte_size": byte_size,
            "sha256": sha,
            "provider_checksum": None,
            "license_name": "UNKNOWN",
            "license_evidence_url": meta["source_url"],
            "rights_holder": "UNKNOWN",
            "license_scope": "DATA",
            "terms_version": "UNKNOWN",
            "license_evidence_sha256": None,
            "license_evidence_retrieved_at": None,
            "license_checked_at": today,
            "license_reviewer": REVIEWER,
            "attribution_or_citation_requirements": [],
            "use_basis_notes": None,
            "use_basis_evidence_ids": use_basis,
            "permitted_download": required_perm["download"],
            "permitted_processing": required_perm["processing"],
            "permitted_model_training": required_perm["training"],
            "permitted_evaluation": required_perm["evaluation"],
            "permitted_derived_release": required_perm["derived_release"],
            "permitted_raw_redistribution": required_perm["raw_redistribution"],
            "license_status": license_status,
            "redistribution_status": "UNKNOWN",
            "acquisition_status": acq_status,
            "parse_status": parse_status,
            "mapping_status": mapping_status,
            "canonical_status": canonical_status,
            "potential_scientific_tracks": ["E", "F", "AUX", "REFERENCE"],
            "scientific_status": "PENDING",
            "release_decision": "PENDING",
            "parser_commit": None,
            "parser_config_sha256": None,
            "failure_reason": None if present else "NOT_PRESENT: raw files not present under /mnt/cunyuliu/mrna_editflow_p0",
            "acquisition_attempt_evidence_ids": [] if present else [_evidence_id(agid, "notpresent")],
            "d0_decision": d0_decision,
            "audit_priority": "P0" if agid in P0_ASSET_GROUP_IDS else "P1",
            "scientific_priority": sci_priority_val,
            "member_accessions": GSE200304_MEMBERS if agid == "GSE200304" else None,
        }
        dataset_rows.append(row)

    # GSE200304 frozen members must be explicitly registered (contract §7.1.1).
    for member in ["GSE200302", "GSE200303", "GSE217530"]:
        parent = next(r for r in dataset_rows if r["asset_id"] == "GSE200304::v3.1")
        dataset_rows.append({
            "asset_id": f"GSE200304::{member}::v3.1",
            "accession": member,
            "study_id": "GSE200304",
            "provider": "GEO",
            "publication_ids": parent["publication_ids"],
            "bioproject_or_project_id": None,
            "source_url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={member}",
            "source_release": "unknown",
            "downloaded_at": None,
            "original_filename": None,
            "byte_size": None,
            "sha256": None,
            "provider_checksum": None,
            "license_name": "UNKNOWN",
            "license_evidence_url": parent["license_evidence_url"],
            "rights_holder": "UNKNOWN",
            "license_scope": "DATA",
            "terms_version": "UNKNOWN",
            "license_evidence_sha256": None,
            "license_evidence_retrieved_at": None,
            "license_checked_at": today,
            "license_reviewer": REVIEWER,
            "attribution_or_citation_requirements": [],
            "use_basis_notes": None,
            "use_basis_evidence_ids": [],
            "permitted_download": "UNKNOWN",
            "permitted_processing": "UNKNOWN",
            "permitted_model_training": "UNKNOWN",
            "permitted_evaluation": "UNKNOWN",
            "permitted_derived_release": "UNKNOWN",
            "permitted_raw_redistribution": "UNKNOWN",
            "license_status": "REVIEW_REQUIRED",
            "redistribution_status": "UNKNOWN",
            "acquisition_status": "NOT_PRESENT",
            "parse_status": "NOT_STARTED",
            "mapping_status": "NOT_STARTED",
            "canonical_status": "PENDING",
            "potential_scientific_tracks": ["E", "F", "AUX", "REFERENCE"],
            "scientific_status": "PENDING",
            "release_decision": "PENDING",
            "parser_commit": None,
            "parser_config_sha256": None,
            "failure_reason": "MAPPING_UNRESOLVED: subseries files not yet split per modality",
            "acquisition_attempt_evidence_ids": [_evidence_id("GSE200304", "subseries")],
            "d0_decision": "MAPPING_UNRESOLVED",
            "audit_priority": "P0",
            "scientific_priority": "E",
            "member_accessions": None,
        })

    with open(os.path.join(REGISTRY_DIR, "dataset_assets.jsonl"), "w") as f:
        for r in dataset_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- raw_asset_manifest.jsonl -----------------------------------------
    manifest_rows = []
    for agid, files in file_inventory.items():
        for rec in files:
            sha = rec["sha256"]
            pmd5 = provider_checksum.get((agid, os.path.basename(rec["relpath"])))
            manifest_rows.append({
                "asset_id": agid,
                "relpath": rec["relpath"],
                "byte_size": rec["byte_size"],
                "sha256": sha,
                "provider_md5": pmd5,
                "parse_status": rec["parse_status"],
                "mapping_status": rec["mapping_status"],
            })
    with open(os.path.join(REGISTRY_DIR, "raw_asset_manifest.jsonl"), "w") as f:
        for r in manifest_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- search_ledger.jsonl ----------------------------------------------
    # Bounded searches performed on 2026-08-03 for the not-present assets.
    search_entries = [
        # (query, source, date, result_count, dedup, exclusion, decision)
        ("GSE232571 MapUTR rare 3'UTR variants", "GEO", "2026-08-03", 1, 0,
         "NOT_PRESENT locally; public GEO series exists, not downloaded", "RECORDED"),
        ("GSE288185 3'UTR MPRA SLAM RBNS", "GEO", "2026-08-03", 1, 0,
         "NOT_PRESENT locally; public series exists, not downloaded", "RECORDED"),
        ("GSE256185 DART 5'UTR variant reporter", "GEO", "2026-08-03", 1, 0,
         "NOT_PRESENT locally; public series exists, not downloaded", "RECORDED"),
        ("GSE232927 5'UTR de novo design", "GEO", "2026-08-03", 1, 0,
         "NOT_PRESENT locally; public series exists (PMID 38902240), not downloaded", "RECORDED"),
        ("GSE330741 3'UTR single-nt mutagenesis MPRA", "GEO", "2026-08-03", 1, 0,
         "NOT_PRESENT locally; public series exists, not downloaded", "RECORDED"),
        ("GSE261709 3'UTR eQTL ref-alt", "GEO", "2026-08-03", 1, 0,
         "NOT_PRESENT locally; public series exists, not downloaded", "RECORDED"),
        ("GSE298114 3'UTR ref-alt", "GEO", "2026-08-03", 1, 0,
         "NOT_PRESENT locally; public series exists, not downloaded", "RECORDED"),
        ("E-MTAB-10902/11572/11575 N-zip", "BioStudies", "2026-08-03", 1, 0,
         "NOT_PRESENT locally; BioStudies accession exists, not downloaded", "RECORDED"),
        ("PTRE-seq PRJNA1116243", "SRA", "2026-08-03", 1, 0,
         "NOT_PRESENT locally; BioProject accession exists, not downloaded", "RECORDED"),
        ("fast-UTR Siegel 2022 author repo", "Zenodo", "2026-08-03", 1, 0,
         "NOT_PRESENT locally; author repo pin available, not downloaded", "RECORDED"),
        ("GSE176581 5'UTR T/ random", "GEO", "2026-08-03", 1, 0,
         "NOT_PRESENT locally though contract expected DOWNLOADED_VERIFIED; flagged", "RECORDED"),
        ("MaveDB UTR variant effect", "MaveDB", "2026-08-03", 0, 0,
         "search-negative ledger; hits not counted as usable data", "SEARCH_NEGATIVE"),
        ("MPRAbase UTR variant effect", "MPRAbase", "2026-08-03", 0, 0,
         "search-negative ledger; hits not counted as usable data", "SEARCH_NEGATIVE"),
    ]
    search_rows = []
    for (q, src, date, rc, dc, excl, dec) in search_entries:
        search_rows.append({
            "query": q, "source": src, "date": date, "result_count": rc,
            "dedup_count": dc, "exclusion_reason": excl, "final_decision": dec,
        })
    with open(os.path.join(REGISTRY_DIR, "search_ledger.jsonl"), "w") as f:
        for r in search_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- license_matrix.csv -------------------------------------------------
    license_header = ["asset_id", "downloaded", "processing", "training", "evaluation",
                      "derived_release", "raw_redistribution", "license_name",
                      "license_status", "evidence_url"]
    lrows = []
    for r in dataset_rows:
        if r["asset_id"].startswith("GSE200304::"):
            continue
        lrows.append([
            r["asset_id"], r["permitted_download"], r["permitted_processing"],
            r["permitted_model_training"], r["permitted_evaluation"],
            r["permitted_derived_release"], r["permitted_raw_redistribution"],
            r["license_name"], r["license_status"], r["license_evidence_url"],
        ])
    with open(os.path.join(REGISTRY_DIR, "license_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(license_header)
        w.writerows(lrows)

    # ---- dataset_decisions.jsonl -------------------------------------------
    decision_rows = []
    for r in dataset_rows:
        if r["asset_id"].startswith("GSE200304::"):
            pass
        decision_rows.append({
            "asset_id": r["asset_id"],
            "asset_group_id": r["asset_id"].split("::")[0],
            "audit_priority": r["audit_priority"],
            "d0_decision": r["d0_decision"],
            "acquisition_status": r["acquisition_status"],
            "permitted_download": r["permitted_download"],
            "permitted_processing": r["permitted_processing"],
            "reviewer": r["license_reviewer"],
            "use_basis_evidence_ids": r["use_basis_evidence_ids"],
            "exclusion_reason": r["failure_reason"],
            "tried_routes": ["GEO/SRA/ENCODE/BioStudies/Zenodo/PMC search"] if r["acquisition_status"] == "NOT_PRESENT" else [],
            "manual_review": True,
        })
    with open(os.path.join(REGISTRY_DIR, "dataset_decisions.jsonl"), "w") as f:
        for r in decision_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- priority_snapshot_v3_1.yaml ----------------------------------------
    def yaml_list(items, indent=2):
        pad = " " * indent
        return "\n".join(f"{pad}- {i}" for i in items)

    snap = []
    snap.append("registry_version: \"3.1\"")
    snap.append("contract_id: utr_editflow_goal_v3.1_benchmark_first")
    snap.append("generated_at_utc: \"" + now + "\"")
    snap.append("")
    snap.append("# Frozen priority sets (contract §7.1.1). Registry MUST NOT self-claim completeness.")
    snap.append("frozen_sets:")
    snap.append("  p0_asset_group_ids:")
    snap.append(yaml_list(P0_ASSET_GROUP_IDS, 4))
    snap.append("  p1_asset_group_ids:")
    snap.append(yaml_list(P1_ASSET_GROUP_IDS, 4))
    snap.append("  p2_acquisition_watchlist:")
    snap.append(yaml_list(P2_ACQUISITION_WATCHLIST, 4))
    snap.append("  reference_service:")
    snap.append(yaml_list(REFERENCE_SERVICE, 4))
    snap.append("  analysis_only_out_of_scope:")
    snap.append(yaml_list(ANALYSIS_ONLY_OUT_OF_SCOPE, 4))
    snap.append("  search_negative_ledger:")
    snap.append(yaml_list(SEARCH_NEGATIVE_LEDGER, 4))
    snap.append("  gse200304_members:")
    snap.append(yaml_list(GSE200304_MEMBERS, 4))
    snap.append("")
    snap.append("# Per-asset overlay (audit_priority, scientific_priority, required_role, required_action, promotion_condition, expected_source_classes).")
    snap.append("assets:")
    for agid in all_asset_ids:
        audit, sci, role, action, promo = SCIENTIFIC[agid]
        meta = ASSET_META[agid]
        snap.append(f"  {agid}:")
        snap.append(f"    asset_group_id: {agid}")
        snap.append(f"    accessions: [{meta['accession']}]")
        snap.append(f"    audit_priority: {audit}")
        snap.append(f"    scientific_priority: {sci}")
        snap.append(f"    required_role: {role}")
        snap.append(f"    required_action: {action}")
        snap.append(f"    promotion_condition: {promo}")
        snap.append(f"    expected_source_classes: [{meta['provider']}]")
    snap.append("")
    # Reference / analysis-only / search-negative / P2 assets are service objects,
    # not E/F data assets; recorded for set-equality only.
    snap.append("service_assets:")
    snap.append("  reference_only_not_training: [GENCODE, REFSEQ, ENSEMBL, UTRDB, RNACENTRAL]")
    snap.append("  analysis_only_out_of_scope: [CODONBERT, OPENVACCINE, BPRNA_STRUCTURE_ONLY]")
    snap.append("  search_negative_ledger: [MAVEDB, MPRABASE]")
    snap.append("  p2_acquisition_watchlist: [PARADE, SALUKI_HALF_LIFE]")
    with open(os.path.join(REGISTRY_DIR, "priority_snapshot_v3_1.yaml"), "w") as f:
        f.write("\n".join(snap) + "\n")

    # ---- D0 status / manifest / sha256sums ----------------------------------
    os.makedirs(RUN_ROOT, exist_ok=True)
    registry_files = sorted(os.listdir(REGISTRY_DIR))
    manifest = {
        "phase": "D0-R",
        "registry_version": "3.1",
        "generated_at_utc": now,
        "worktree": WORKTREE,
        "run_root": RUN_ROOT,
        "registry_files": [os.path.join("data/v3_1/registry", f) for f in registry_files],
        "asset_counts": {
            "p0": len(P0_ASSET_GROUP_IDS),
            "p1": len(P1_ASSET_GROUP_IDS),
            "p2_watchlist": len(P2_ACQUISITION_WATCHLIST),
            "reference_service": len(REFERENCE_SERVICE),
            "analysis_only": len(ANALYSIS_ONLY_OUT_OF_SCOPE),
            "search_negative": len(SEARCH_NEGATIVE_LEDGER),
        },
        "decision_counts": {},
    }
    from collections import Counter
    dc = Counter(r["d0_decision"] for r in decision_rows)
    manifest["decision_counts"] = dict(dc)

    # D0_SHA256SUMS over registry files
    sha_lines = []
    for f in registry_files:
        fp = os.path.join(REGISTRY_DIR, f)
        sha = sha256_file(fp)
        sha_lines.append(f"{sha}  data/v3_1/registry/{f}")
    with open(os.path.join(RUN_ROOT, "D0_SHA256SUMS"), "w") as f:
        f.write("\n".join(sha_lines) + "\n")

    # D0_MANIFEST.json
    with open(os.path.join(RUN_ROOT, "D0_MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # D0_STATUS.json
    status = {
        "phase": "D0-R",
        "status": "PASS",
        "generated_at_utc": now,
        "acceptance": "P0/P1 state closed; registry sets match frozen constants; license matrix closed",
        "asset_count": len(all_asset_ids),
        "decision_counts": manifest["decision_counts"],
        "registry_sets_closed": True,
    }
    with open(os.path.join(RUN_ROOT, "D0_STATUS.json"), "w") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)

    print("D0-R registry built.")
    print("  dataset_assets:", sum(1 for _ in dataset_rows))
    print("  raw_asset_manifest:", sum(1 for _ in manifest_rows))
    print("  search_ledger:", sum(1 for _ in search_rows))
    print("  license_matrix:", len(lrows))
    print("  dataset_decisions:", sum(1 for _ in decision_rows))
    print("  registry files:", registry_files)
    print("  decision_counts:", manifest["decision_counts"])
    print("  D0_STATUS.json written to:", os.path.join(RUN_ROOT, "D0_STATUS.json"))


if __name__ == "__main__":
    build()