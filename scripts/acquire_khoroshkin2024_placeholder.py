#!/usr/bin/env python3
"""
Generate manifest for Khoroshkin 2024 bioRxiv PARADE dataset.

Status: Dataset NOT yet publicly deposited.
- Preprint: bioRxiv 2024.12.31.630783 (Dec 31, 2024)
- PMID: 39803435, PMCID: PMC11722239
- 60,000 5' and 3' UTRs × 6 cell types (Jurkat, Nalm-6, SW-480, PA-1, MDA-MB-231, HepG2)
- 15,800 de novo-designed validation sequences
- No public GEO/Zenodo/GitHub accession found as of 2026-07-19

This manifest documents the dataset metadata and the blocker (data not yet public).
Authors should be contacted for data access. Once data is deposited, this manifest
should be updated with source_url, sha256, and per-file records.
"""

import json
import os
from pathlib import Path

_REPO_ROOT = os.environ.get("PYTHONPATH", "").split(":")[0] or "/home/cunyuliu/mrna_editflow_goal"
DATA_ROOT = Path(_REPO_ROOT) / "mrna_editflow" / "data" / "raw" / "khoroshkin2024_parade"
DATA_ROOT.mkdir(parents=True, exist_ok=True)

CITATION = (
    "Khoroshkin M, Zinkevich A, Aristova E, Yousefi H, Lee SB, Mittmann T, Manegold K, "
    "Penzar D, Raleigh DR, Kulakovskiy IV, Goodarzi H. A generative framework for "
    "enhanced cell-type specificity in rationally designed mRNAs. bioRxiv [Preprint]. "
    "2024 Dec 31:2024.12.31.630783. PMID:39803435. DOI:10.1101/2024.12.31.630783."
)

LICENSE_TEXT = (
    "Preprint: CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/). "
    "Dataset NOT yet publicly deposited as of 2026-07-19. "
    "Contact authors: hani.goodarzi@arcinstitute.org, ivan.kulakovskiy@gmail.com, khorms21@gmail.com"
)

manifest = {
    "dataset_name": "khoroshkin2024_parade",
    "description": (
        "Khoroshkin et al. 2024 bioRxiv PARADE dataset. 60,000 5' and 3' UTRs screened "
        "across 6 cell types (Jurkat, Nalm-6, SW-480, PA-1, MDA-MB-231, HepG2) using MPRA. "
        "15,800 de novo-designed validation sequences. Used to train PARADE Predictor and "
        "PARADE Generator for cell type-specific mRNA UTR design."
    ),
    "citation": CITATION,
    "license": LICENSE_TEXT,
    "paper": {
        "title": "A generative framework for enhanced cell-type specificity in rationally designed mRNAs",
        "authors": (
            "Khoroshkin M, Zinkevich A, Aristova E, Yousefi H, Lee SB, Mittmann T, "
            "Manegold K, Penzar D, Raleigh DR, Kulakovskiy IV, Goodarzi H"
        ),
        "journal": "bioRxiv (Preprint)",
        "year": 2024,
        "pmid": "39803435",
        "pmcid": "PMC11722239",
        "doi": "10.1101/2024.12.31.630783",
    },
    "status": "BLOCKED - dataset not yet publicly deposited",
    "block_reason": (
        "As of 2026-07-19, no public GEO/Zenodo/GitHub accession found for the 60k UTR MPRA "
        "dataset described in the Khoroshkin 2024 bioRxiv preprint. Preprint is on bioRxiv "
        "but supplementary data not exposed via standard WebFetch. Authors must be contacted "
        "directly for data access."
    ),
    "expected_data": [
        {
            "name": "natural_set_library_5utr",
            "description": "30,000 50-nt 5'UTR fragments from 2068 transcripts with cell-type variable TE",
            "expected_count": 30000,
            "cell_types": ["Jurkat", "Nalm-6", "SW-480", "PA-1", "MDA-MB-231", "HepG2"],
        },
        {
            "name": "natural_set_library_3utr",
            "description": "30,000 3'UTR fragments from 2068 transcripts with cell-type variable TE",
            "expected_count": 30000,
            "cell_types": ["Jurkat", "Nalm-6", "SW-480", "PA-1", "MDA-MB-231", "HepG2"],
        },
        {
            "name": "designed_validation_set",
            "description": "15,800 de novo-designed UTR sequences tested for validation",
            "expected_count": 15800,
            "cell_types": ["Jurkat", "Nalm-6", "SW-480", "PA-1", "MDA-MB-231", "HepG2"],
        },
        {
            "name": "stability_mpra",
            "description": "3'UTR MPRA for mRNA stability (reporter RNA-to-DNA ratio)",
            "expected_count": None,
            "cell_types": ["Jurkat", "Nalm-6", "SW-480", "PA-1", "MDA-MB-231", "HepG2"],
        },
    ],
    "cell_type_details": {
        "Jurkat": "T cells (CD8+)",
        "Nalm-6": "B cells",
        "SW-480": "Colon adenocarcinoma",
        "PA-1": "Ovarian teratocarcinoma",
        "MDA-MB-231": "Breast adenocarcinoma (triple-negative)",
        "HepG2": "Hepatocellular carcinoma",
    },
    "source_repositories": {
        "biorxiv_preprint": "https://doi.org/10.1101/2024.12.31.630783",
        "pmc_article": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11722239/",
        "github_code": None,
        "zenodo_data": None,
        "geo_raw_data": None,
    },
    "next_steps_to_unblock": [
        "1. Email corresponding authors (hani.goodarzi@arcinstitute.org, ivan.kulakovskiy@gmail.com) requesting data access",
        "2. Check bioRxiv for updated preprint versions with data accession",
        "3. Check if a peer-reviewed publication has been issued with deposited data",
        "4. Once data is obtained, re-run acquire script with source_url, sha256, and per-file records",
    ],
    "files": [],
    "deferred": True,
    "deferred_reason": "Data not yet publicly available. Documented here as a known dataset for future acquisition.",
}

manifest_path = DATA_ROOT / "manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(f"[manifest] {manifest_path}")
print(f"  status: {manifest['status']}")
print(f"  files: 0 (deferred)")
