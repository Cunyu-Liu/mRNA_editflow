#!/usr/bin/env python3
"""Build the provenance-first NMI Benchmark v2 registry.

The canonical store contains two explicitly separated task families:

* ``local_delta`` records from the source-matched intervention tier (Layer C),
  which are the only records eligible as biological local-delta ground truth;
* ``absolute_property_*`` records from public absolute libraries (Layer B),
  which make context/assay axes observable without being relabelled as local
  interventions.

The builder never aliases one record into multiple final roles.  Final labels
remain physically present in the canonical store for reproducibility, but the
loader is fail-closed and refuses final manifests unless explicitly opened.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


ROLES = ["train", "val", "test_id", "test_family", "test_context", "test_assay", "test_ood"]
FINAL_ROLES = set(ROLES[2:])
REQUIRED_SOURCE_MATCHED_FIELDS = [
    "source_id", "candidate_id", "source_sequence", "candidate_sequence",
    "edit_list", "edit_count", "measured_source", "measured_candidate",
    "measured_delta", "cargo", "cell_context", "assay", "batch", "replicate",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_fraction(value: str) -> float:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:12], 16) / float(16**12)


def assign_local_role(old_role: str, source_id: str) -> str:
    """Re-use the frozen P3 source split while making all local axes distinct.

    Historical P3 ``test``/``ood`` and validation labels were exposed during
    prior development and are therefore never promoted to v2 final roles.
    They remain development/quarantine inputs only; new final roles come from
    the raw designed library below.
    """
    if old_role == "train":
        return "train"
    if old_role == "val":
        return "val"
    raise ValueError(f"unsupported P3 split role {old_role!r}")


def parse_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_sequence(value: object) -> str:
    return str(value or "").strip().upper().replace("T", "U")


def normalize_p3_record(rec: Mapping[str, object], role: str) -> Dict[str, object]:
    confidence = str(rec.get("confidence", "unknown"))
    measured = confidence == "measured"
    source_id = str(rec.get("source_id", ""))
    record_id = str(rec.get("record_id", ""))
    source_sequence = normalize_sequence(rec.get("source_sequence"))
    candidate_sequence = normalize_sequence(rec.get("candidate_sequence"))
    internal = rec.get("internal_features") if isinstance(rec.get("internal_features"), dict) else {}
    measured_source = parse_float(rec.get("measured_or_proxy_source_value")) if measured else None
    measured_candidate = parse_float(rec.get("measured_or_proxy_candidate_value")) if measured else None
    measured_delta = parse_float(rec.get("delta")) if measured else None
    edit_list = rec.get("edit_list") if isinstance(rec.get("edit_list"), list) else []
    result = dict(rec)
    result.update({
        "benchmark_version": "nmi_benchmark_v2",
        "v2_source_role": role,
        "data_layer": "C_source_matched_intervention" if measured else "B_absolute_design_library",
        "task_kind": "local_delta" if measured else "derived_candidate_asset",
        "local_delta_eligible": measured and measured_delta is not None,
        "candidate_id": f"{record_id}:candidate",
        "mother_id": rec.get("mother_id") or source_id,
        "source_sequence": source_sequence,
        "candidate_sequence": candidate_sequence,
        "edit_list": edit_list,
        "edit_count": int(rec.get("edit_count") or len(edit_list)),
        "measured_source": measured_source,
        "measured_candidate": measured_candidate,
        "measured_delta": measured_delta,
        "cargo": rec.get("cargo_id"),
        "cell_context": rec.get("cell_context"),
        "assay": rec.get("assay_type"),
        "batch": rec.get("data_source"),
        "replicate": internal.get("replicate", internal.get("n_wt_replicates")),
        "source_sequence_sha256": sha256_text(source_sequence),
        "candidate_sequence_sha256": sha256_text(candidate_sequence),
        "label_visibility": "hidden_before_freeze" if role in FINAL_ROLES else "development_allowed",
        "label_semantics": "wet_lab_source_matched_delta" if measured else "not_ground_truth",
    })
    return result


def iter_fasta(path: Path) -> Iterator[Tuple[str, str]]:
    header: Optional[str] = None
    chunks: List[str] = []
    with path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, normalize_sequence("".join(chunks))
                header = line[1:]
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        yield header, normalize_sequence("".join(chunks))


def header_field(header: str, key: str) -> Optional[str]:
    prefix = key + "="
    for part in header.split("|"):
        if part.startswith(prefix):
            return part[len(prefix):]
    return None


def make_absolute_record(
    *, record_id: str, source_id: str, sequence: str, value: float,
    cargo: str, context: str, assay: str, batch: str, replicate: Optional[object],
    family: str, role: str, task_kind: str, data_source: str,
) -> Dict[str, object]:
    return {
        "benchmark_version": "nmi_benchmark_v2",
        "record_id": record_id,
        "source_id": source_id,
        "mother_id": source_id,
        "candidate_id": f"{record_id}:candidate",
        "source_sequence": sequence,
        "candidate_sequence": sequence,
        "source_sequence_sha256": sha256_text(sequence),
        "candidate_sequence_sha256": sha256_text(sequence),
        "edit_list": [],
        "edit_count": 0,
        "measured_source": value,
        "measured_candidate": value,
        "measured_delta": None,
        "delta": None,
        "measured_or_proxy_source_value": value,
        "measured_or_proxy_candidate_value": value,
        "cargo": cargo,
        "cargo_id": cargo,
        "cell_context": context,
        "assay": assay,
        "assay_type": assay,
        "batch": batch,
        "replicate": replicate,
        "family_cluster_id": family,
        "confidence": "measured",
        "data_source": data_source,
        "data_layer": "B_absolute_design_library",
        "task_kind": task_kind,
        "local_delta_eligible": False,
        "label_visibility": "hidden_before_freeze" if role in FINAL_ROLES else "development_allowed",
        "label_semantics": "absolute_property_only_not_local_delta_ground_truth",
        "split_role": role,
        "v2_source_role": role,
        "task_eligibility": "absolute_only",
        "value_qualifier": "wet-lab measured absolute property; not a source-matched intervention",
    }


def make_source_matched_axis_record(
    *, record_id: str, source_id: str, source_sequence: str,
    candidate_sequence: str, measured_source: float, measured_candidate: float,
    cargo: str, context: str, assay: str, batch: str,
    family: str, role: str, task_kind: str, data_source: str,
) -> Dict[str, object]:
    """Represent an assay/context intervention without calling it a sequence edit."""
    return {
        "benchmark_version": "nmi_benchmark_v2",
        "record_id": record_id,
        "source_id": source_id,
        "mother_id": source_id,
        "candidate_id": f"{record_id}:candidate",
        "source_sequence": source_sequence,
        "candidate_sequence": candidate_sequence,
        "source_sequence_sha256": sha256_text(source_sequence),
        "candidate_sequence_sha256": sha256_text(candidate_sequence),
        "edit_list": [],
        "edit_count": 0,
        "measured_source": measured_source,
        "measured_candidate": measured_candidate,
        "measured_delta": measured_candidate - measured_source,
        "delta": measured_candidate - measured_source,
        "measured_or_proxy_source_value": measured_source,
        "measured_or_proxy_candidate_value": measured_candidate,
        "cargo": cargo,
        "cargo_id": cargo,
        "cell_context": context,
        "assay": assay,
        "assay_type": assay,
        "batch": batch,
        "replicate": None,
        "family_cluster_id": family,
        "confidence": "measured",
        "data_source": data_source,
        "data_layer": "C_source_matched_intervention",
        "task_kind": task_kind,
        "local_delta_eligible": False,
        "label_visibility": "hidden_before_freeze" if role in FINAL_ROLES else "development_allowed",
        "label_semantics": "source_matched_assay_or_context_delta; not a nucleotide edit ground truth",
        "split_role": role,
        "v2_source_role": role,
        "task_eligibility": "axis_shift_only",
        "value_qualifier": "wet-lab measured source-matched assay/context shift",
    }


def derive_single_nucleotide_edits(source: str, candidate: str) -> Optional[List[Dict[str, object]]]:
    if len(source) != len(candidate):
        return None
    changes = []
    for pos, (ref, alt) in enumerate(zip(source, candidate)):
        if ref != alt:
            changes.append({"pos": pos, "ref": ref, "alt": alt, "region": "five_utr"})
    return changes


def raw_source_role(source_id: str, family_id: str, source: str) -> Optional[str]:
    """Assign only new raw-library records to untouched final roles."""
    gc = (source.count("G") + source.count("C")) / max(1, len(source))
    if gc <= 0.20 or gc >= 0.80:
        return "test_ood"
    family_fraction = stable_fraction("family:" + family_id)
    if family_fraction < 0.20:
        return "test_family"
    source_fraction = stable_fraction("source:" + source_id)
    if 0.20 <= source_fraction < 0.40:
        return "test_id"
    return None


def iter_raw_untouched_records(
    path: Path, *, excluded_source_hashes: set[str], excluded_candidate_hashes: set[str],
) -> Iterator[Tuple[str, Dict[str, object]]]:
    """Read source-matched rows not present in the historical P3 measured tier."""
    anchors: Dict[str, float] = {}
    rows: List[Tuple[int, Dict[str, str]]] = []
    with gzip.open(path, "rt", newline="") as fh:
        reader = csv.DictReader(fh)
        for row_i, row in enumerate(reader):
            source = normalize_sequence(row.get("mother"))
            candidate = normalize_sequence(row.get("utr"))
            value = parse_float(row.get("rl"))
            if not source or not candidate or value is None:
                continue
            rows.append((row_i, row))
            if source == candidate:
                anchors.setdefault(source, value)
    seen_pairs = set()
    source_family_by_hash: Dict[str, str] = {}
    for row_i, row in rows:
        source = normalize_sequence(row.get("mother"))
        candidate = normalize_sequence(row.get("utr"))
        if source == candidate or source not in anchors:
            continue
        source_hash = sha256_text(source)
        candidate_hash = sha256_text(candidate)
        if source_hash in excluded_source_hashes or candidate_hash in excluded_candidate_hashes:
            continue
        edit_list = derive_single_nucleotide_edits(source, candidate)
        if not edit_list or len(edit_list) > 3:
            continue
        pair_key = (source, candidate)
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        family_raw = str(row.get("info1") or row.get("info2") or "unknown")
        family_raw = source_family_by_hash.setdefault(source_hash, family_raw)
        family_id = f"raw_library_family:{family_raw}"
        source_id = f"raw_sample2019:{source_hash[:20]}"
        role = raw_source_role(source_id, family_id, source)
        if role is None:
            continue
        candidate_value = parse_float(row.get("rl"))
        if candidate_value is None:
            continue
        record_id = f"v2untouched:sample2019:{row_i:07d}:{candidate_hash[:12]}"
        record = {
            "benchmark_version": "nmi_benchmark_v2",
            "record_id": record_id,
            "source_id": source_id,
            "mother_id": source_id,
            "candidate_id": f"{record_id}:candidate",
            "source_sequence": source,
            "candidate_sequence": candidate,
            "source_sequence_sha256": source_hash,
            "candidate_sequence_sha256": candidate_hash,
            "edit_list": edit_list,
            "edit_count": len(edit_list),
            "measured_source": anchors[source],
            "measured_candidate": candidate_value,
            "measured_delta": candidate_value - anchors[source],
            "delta": candidate_value - anchors[source],
            "measured_or_proxy_source_value": anchors[source],
            "measured_or_proxy_candidate_value": candidate_value,
            "cargo": "eGFP",
            "cargo_id": "eGFP",
            "cell_context": "HEK293T",
            "assay": "MPRA_2019_designed_library_untouched",
            "assay_type": "MPRA_2019_designed_library_untouched",
            "batch": path.name,
            "replicate": 1,
            "family_cluster_id": family_id,
            "confidence": "measured",
            "data_source": f"sample2019_mpra:{path.name}:raw_row_{row_i}",
            "data_layer": "C_source_matched_intervention",
            "task_kind": "local_delta",
            "local_delta_eligible": True,
            "label_semantics": "raw_source_matched_single_edit; excluded from historical P3 membership",
            "task_eligibility": "task_a_active",
            "value_qualifier": "wet-lab measured MPRA raw row not in historical P3 measured membership",
            "split_role": role,
            "v2_source_role": role,
        }
        yield role, record


def iter_mpra_records(
    path: Path, *, role: str, max_records: int, task_kind: str, assay_label: str,
    cargo: str, context: str, batch: str, replicate: Optional[int],
) -> Iterator[Dict[str, object]]:
    """Read a bounded public MPRA absolute library without loading it in RAM."""
    with gzip.open(path, "rt", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return
        try:
            seq_i = header.index("utr")
        except ValueError:
            seq_i = 1
        try:
            value_i = header.index("rl")
        except ValueError:
            value_i = len(header) - 1
        emitted = 0
        for row_i, row in enumerate(reader):
            if emitted >= max_records:
                break
            if len(row) <= max(seq_i, value_i):
                continue
            sequence = normalize_sequence(row[seq_i])
            value = parse_float(row[value_i])
            if not sequence or value is None:
                continue
            rid = f"v2abs:mpra:{path.stem}:{row_i:07d}"
            yield make_absolute_record(
                record_id=rid, source_id=f"{rid}:source", sequence=sequence,
                value=value, cargo=cargo, context=context, assay=assay_label,
                batch=batch, replicate=replicate, family=f"mpra:{cargo}", role=role,
                task_kind=task_kind, data_source=f"sample2019_mpra:{path.name}",
            )
            emitted += 1


def iter_mpra_assay_pairs(
    source_path: Path, candidate_path: Path, *, role: str, max_pairs: int,
) -> Iterator[Dict[str, object]]:
    """Pair identical 50mers across chemistry conditions for assay-shift testing."""
    def values(path: Path) -> Dict[str, float]:
        result: Dict[str, float] = {}
        with gzip.open(path, "rt", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            seq_i, value_i = header.index("utr"), header.index("rl")
            for row in reader:
                if len(row) <= max(seq_i, value_i):
                    continue
                sequence, value = normalize_sequence(row[seq_i]), parse_float(row[value_i])
                if sequence and value is not None:
                    result.setdefault(sequence, value)
        return result

    candidate_values = values(candidate_path)
    emitted = 0
    with gzip.open(source_path, "rt", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        seq_i, value_i = header.index("utr"), header.index("rl")
        for row_i, row in enumerate(reader):
            if emitted >= max_pairs:
                break
            if len(row) <= max(seq_i, value_i):
                continue
            sequence, source_value = normalize_sequence(row[seq_i]), parse_float(row[value_i])
            candidate_value = candidate_values.get(sequence)
            if not sequence or source_value is None or candidate_value is None:
                continue
            rid = f"v2axis:assay:{source_path.stem}:{candidate_path.stem}:{row_i:07d}"
            yield make_source_matched_axis_record(
                record_id=rid, source_id=f"{rid}:source", source_sequence=sequence,
                candidate_sequence=sequence, measured_source=source_value,
                measured_candidate=candidate_value, cargo="eGFP", context="HEK293T",
                assay="MPRA_1methylpseudouridine_vs_unmodified",
                batch=f"{source_path.name}|{candidate_path.name}", family="mpra:eGFP",
                role=role, task_kind="assay_delta",
                data_source=f"sample2019_mpra:{source_path.name}+{candidate_path.name}",
            )
            emitted += 1


def iter_cao_records(path: Path, *, role: str, max_records: int) -> Iterator[Dict[str, object]]:
    emitted = 0
    context = "HEK293T" if path.name.startswith("hek_") else ("PC3" if path.name.startswith("pc3_") else "Muscle")
    task_kind = "absolute_property_context_shift" if context != "HEK293T" else "absolute_property_regression"
    for row_i, (header, sequence) in enumerate(iter_fasta(path)):
        if emitted >= max_records:
            break
        value = parse_float(header_field(header, "te"))
        if not sequence or value is None:
            continue
        accession = header.split("|", 1)[0]
        rid = f"v2abs:cao:{path.stem}:{row_i:06d}"
        yield make_absolute_record(
            record_id=rid, source_id=f"{rid}:source", sequence=sequence,
            value=value, cargo="endogenous_transcript", context=context,
            assay="Cao2021_TE", batch=path.name, replicate=None,
            family=f"cao:{accession}", role=role, task_kind=task_kind,
            data_source=f"cao2021_5utr:{path.name}",
        )
        emitted += 1


def iter_cao_context_pairs(
    hek_paths: Sequence[Path], context_path: Path, *, role: str, max_pairs: int,
) -> Iterator[Dict[str, object]]:
    """Pair matching transcript accessions across HEK293T and another context."""
    hek: Dict[str, Tuple[str, float, str]] = {}
    for path in hek_paths:
        for header, sequence in iter_fasta(path):
            value = parse_float(header_field(header, "te"))
            accession = header.split("|", 1)[0]
            if accession and sequence and value is not None:
                hek.setdefault(accession, (sequence, value, path.name))
    context = "PC3" if context_path.name.startswith("pc3_") else "Muscle"
    emitted = 0
    for row_i, (header, sequence) in enumerate(iter_fasta(context_path)):
        if emitted >= max_pairs:
            break
        value = parse_float(header_field(header, "te"))
        accession = header.split("|", 1)[0]
        source = hek.get(accession)
        if source is None or not sequence or value is None:
            continue
        source_sequence, source_value, source_batch = source
        rid = f"v2axis:context:{context_path.stem}:{row_i:06d}"
        yield make_source_matched_axis_record(
            record_id=rid, source_id=f"{rid}:source", source_sequence=source_sequence,
            candidate_sequence=sequence, measured_source=source_value,
            measured_candidate=value, cargo="endogenous_transcript", context=context,
            assay="Cao2021_TE_context_shift", batch=f"{source_batch}|{context_path.name}",
            family=f"cao:{accession}", role=role, task_kind="context_delta",
            data_source=f"cao2021_5utr:{source_batch}+{context_path.name}",
        )
        emitted += 1


def asset_entry(data_root: Path, relative: str, *, level: str, name: str,
                label_semantics: str, provenance: str, status: str = "available") -> Dict[str, object]:
    path = data_root / relative
    obj = {
        "name": name,
        "level": level,
        "relative_path": relative,
        "label_semantics": label_semantics,
        "provenance": provenance,
        "status": status if path.exists() else "not_present",
    }
    if path.exists() and path.is_file():
        obj["byte_size"] = path.stat().st_size
        obj["sha256"] = sha256_file(path)
    return obj


def build_asset_registry(data_root: Path) -> Dict[str, List[Dict[str, object]]]:
    """Register only existing assets, while making missing structure explicit."""
    assets = {
        "A_observational_pretraining": [
            asset_entry(data_root, "data/raw/gencode.v45.pc_transcripts.fa.gz", level="A", name="GENCODE v45 protein-coding transcripts", label_semantics="representation/observational_only", provenance="GENCODE v45 source asset"),
            asset_entry(data_root, "data/reconstructed/p0_data_reconstruction_v1/sources/gencode_v45/canonical.records.jsonl", level="A", name="GENCODE canonical transcript records", label_semantics="representation/observational_only", provenance="P0 canonical reconstruction"),
            asset_entry(data_root, "data/raw/cao2021_5utr/final_endogenous_5utr.txt", level="A", name="endogenous 5UTR abundance/TE source table", label_semantics="representation/observational_only", provenance="Cao et al. 2021 source asset"),
            asset_entry(data_root, "data/raw/saluki_halflife/rna_hl_human.npz", level="A", name="human RNA half-life arrays", label_semantics="representation/auxiliary_only", provenance="Saluki RNA half-life asset"),
            asset_entry(data_root, "data/raw/saluki_halflife/rna_hl_mouse.npz", level="A", name="mouse RNA half-life arrays", label_semantics="representation/auxiliary_only", provenance="Saluki RNA half-life asset"),
            asset_entry(data_root, "data/raw/rna_structure/gencode_v45_5utr_structure.jsonl", level="A", name="GENCODE v45 5UTR RNA structure features", label_semantics="representation/auxiliary_only", provenance="ViennaRNA-derived observational feature asset from GENCODE canonical records"),
        ],
        "B_absolute_design_libraries": [
            asset_entry(data_root, "data/raw/sample2019_mpra/GSM3130435_egfp_unmod_1.csv.gz", level="B", name="Sample 2019 random 50mer absolute MPRA", label_semantics="absolute_property_only", provenance="Sample et al. 2019, GSE114002"),
            asset_entry(data_root, "data/raw/sample2019_mpra/GSM3130439_egfp_m1pseudo_1.csv.gz", level="B", name="Sample 2019 modified-RNA absolute MPRA", label_semantics="absolute_property_only", provenance="Sample et al. 2019, GSE114002"),
            asset_entry(data_root, "data/raw/cao2021_5utr/hek_top1000_high_TE.fasta", level="B", name="Cao HEK293T high-TE library", label_semantics="absolute_property_only", provenance="Cao et al. 2021 source asset"),
            asset_entry(data_root, "data/raw/cao2021_5utr/hek_bottom500_low_TE.fasta", level="B", name="Cao HEK293T low-TE library", label_semantics="absolute_property_only", provenance="Cao et al. 2021 source asset"),
            asset_entry(data_root, "data/raw/codonbert_stability/mRNA_Stability.csv", level="B", name="CodonBERT mRNA stability library", label_semantics="absolute_property_only", provenance="CodonBERT stability asset"),
            asset_entry(data_root, "data/raw/codonbert_stability/CoV_Vaccine_Degradation.csv", level="B", name="full-length/CDS-containing vaccine mRNA stability library", label_semantics="absolute_property_only", provenance="CodonBERT stability asset"),
        ],
        "C_source_matched_intervention": [
            {"name": "P3 measured source-matched intervention tier", "level": "C", "relative_path": "data/p3/benchmark/measured_tier.jsonl", "label_semantics": "wet_lab_local_delta_ground_truth", "provenance": "P3 source-matched measured tier", "status": "available"},
            {"name": "Sample 2019 chemistry-matched assay pairs", "level": "C", "relative_path": "data/nmi_benchmark_v2/records.jsonl", "label_semantics": "source_matched_assay_delta_not_sequence_edit", "provenance": "paired unmodified and 1-methylpseudouridine MPRA records", "status": "generated_by_builder"},
            {"name": "Cao context-matched pairs", "level": "C", "relative_path": "data/nmi_benchmark_v2/records.jsonl", "label_semantics": "source_matched_context_delta_not_sequence_edit", "provenance": "paired HEK293T and PC3/Muscle TE records by transcript accession", "status": "generated_by_builder"},
        ],
        "D_prospective_data": [
            {"name": "prospective post-freeze intake", "level": "D", "relative_path": "data/nmi_benchmark_v2/manifests/prospective.json", "label_semantics": "future_only_not_available_during_model_development", "provenance": "P1-01 freeze gate", "status": "empty_until_freeze", "frozen": False},
        ],
    }
    return assets


def build(input_paths: Iterable[Path], out_dir: Path, *, data_root: Path,
          mpra_train_limit: int = 1000, mpra_assay_limit: int = 1000,
          cao_train_limit: int = 1000, cao_context_limit: int = 1000) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifests").mkdir(exist_ok=True)
    (out_dir / "indices").mkdir(exist_ok=True)
    records_path = out_dir / "records.jsonl"
    index_fhs = {role: (out_dir / "indices" / f"{role}.txt").open("w") for role in ROLES}
    counts = Counter()
    confidence = Counter()
    task_kinds = Counter()
    source_role: Dict[str, str] = {}
    family_by_role: Dict[str, set] = defaultdict(set)
    sources_by_role: Dict[str, set] = defaultdict(set)
    contexts_by_role: Dict[str, set] = defaultdict(set)
    assays_by_role: Dict[str, set] = defaultdict(set)
    legacy_excluded = Counter()
    historical_measured_source_hashes: set[str] = set()
    historical_measured_candidate_hashes: set[str] = set()

    def write_record(out, rec: Dict[str, object], role: str) -> None:
        rid = str(rec.get("record_id", ""))
        if not rid:
            raise ValueError("record missing record_id")
        if role not in ROLES:
            raise ValueError(f"invalid role {role!r}")
        rec["split_role"] = role
        rec["v2_source_role"] = role
        rec["label_visibility"] = "hidden_before_freeze" if role in FINAL_ROLES else "development_allowed"
        for field in REQUIRED_SOURCE_MATCHED_FIELDS:
            if field not in rec:
                raise ValueError(f"record {rid} missing required field {field}")
        out.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
        index_fhs[role].write(rid + "\n")
        counts[role] += 1
        confidence[(role, str(rec.get("confidence")))] += 1
        task_kinds[(role, str(rec.get("task_kind")))] += 1
        sid = str(rec.get("source_id"))
        previous = source_role.get(sid)
        if previous is not None and previous != role:
            raise ValueError(f"source {sid} crosses roles: {previous} vs {role}")
        source_role[sid] = role
        sources_by_role[role].add(sid)
        family_by_role[role].add(str(rec.get("family_cluster_id")))
        contexts_by_role[role].add(str(rec.get("cell_context")))
        assays_by_role[role].add(str(rec.get("assay")))

    try:
        with records_path.open("w") as out:
            for input_path in input_paths:
                with input_path.open() as fh:
                    for line_no, line in enumerate(fh, 1):
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        old_role = str(rec.get("split_role", ""))
                        sid = str(rec.get("source_id", ""))
                        if old_role in {"test", "ood"}:
                            legacy_excluded[old_role] += 1
                            continue
                        role = assign_local_role(old_role, sid)
                        normalized = normalize_p3_record(rec, role)
                        write_record(out, normalized, role)
                        if str(rec.get("confidence")) == "measured":
                            historical_measured_source_hashes.add(str(normalized["source_sequence_sha256"]))
                            historical_measured_candidate_hashes.add(str(normalized["candidate_sequence_sha256"]))

            raw_library = data_root / "data/raw/sample2019_mpra/GSM3130443_designed_library.csv.gz"
            if raw_library.exists():
                for role, rec in iter_raw_untouched_records(
                    raw_library,
                    excluded_source_hashes=historical_measured_source_hashes,
                    excluded_candidate_hashes=historical_measured_candidate_hashes,
                ):
                    write_record(out, rec, role)

            # Absolute MPRA: unmodified HEK is development-only; m1pseudo is a
            # held-out assay/chemistry condition. These records never enter the
            # local-delta metric stream.
            mpra_dir = data_root / "data/raw/sample2019_mpra"
            unmodified = mpra_dir / "GSM3130435_egfp_unmod_1.csv.gz"
            if unmodified.exists():
                for i, rec in enumerate(iter_mpra_records(
                    unmodified, role="train", max_records=mpra_train_limit,
                    task_kind="absolute_property_regression", assay_label="MPRA_unmodified",
                    cargo="eGFP", context="HEK293T", batch=unmodified.name,
                    replicate=1,
                )):
                    write_record(out, rec, "train" if i % 5 else "val")
            assay_file = mpra_dir / "GSM3130439_egfp_m1pseudo_1.csv.gz"
            if assay_file.exists():
                for rec in iter_mpra_records(
                    assay_file, role="test_assay", max_records=mpra_assay_limit,
                    task_kind="absolute_property_assay_shift", assay_label="MPRA_1methylpseudouridine",
                    cargo="eGFP", context="HEK293T", batch=assay_file.name,
                    replicate=1,
                ):
                    write_record(out, rec, "test_assay")
            if unmodified.exists() and assay_file.exists():
                for rec in iter_mpra_assay_pairs(
                    mpra_dir / "GSM3130436_egfp_unmod_2.csv.gz", assay_file,
                    role="test_assay", max_pairs=1000,
                ):
                    write_record(out, rec, "test_assay")

            cao_dir = data_root / "data/raw/cao2021_5utr"
            for name in ("hek_top1000_high_TE.fasta", "hek_bottom500_low_TE.fasta"):
                path = cao_dir / name
                if not path.exists():
                    continue
                for i, rec in enumerate(iter_cao_records(path, role="train", max_records=cao_train_limit)):
                    write_record(out, rec, "train" if i % 5 else "val")
            for name in ("pc3_top1000_high_TE.fasta", "pc3_bottom500_low_TE.fasta", "muscle_all_5utr.fasta"):
                path = cao_dir / name
                if not path.exists():
                    continue
                for rec in iter_cao_records(path, role="test_context", max_records=cao_context_limit):
                    write_record(out, rec, "test_context")
                for rec in iter_cao_context_pairs(
                    (cao_dir / "hek_top1000_high_TE.fasta", cao_dir / "hek_bottom500_low_TE.fasta"),
                    path, role="test_context", max_pairs=cao_context_limit,
                ):
                    write_record(out, rec, "test_context")
    finally:
        for fh in index_fhs.values():
            fh.close()

    manifest_paths = {}
    for role in ROLES:
        idx = out_dir / "indices" / f"{role}.txt"
        task_role_counts = {k[1]: n for k, n in task_kinds.items() if k[0] == role}
        manifest = {
            "schema_version": "nmi_benchmark_v2",
            "role": role,
            "final_test": role in FINAL_ROLES,
            "records_path": "records.jsonl",
            "index_path": f"indices/{role}.txt",
            "record_count": int(counts[role]),
            "index_sha256": sha256_file(idx),
            "source_count": len(sources_by_role.get(role, set())),
            "task_kind_counts": dict(sorted(task_role_counts.items())),
            "label_policy": "hidden_by_default" if role in FINAL_ROLES else "development_allowed",
            "local_delta_ground_truth": role in {"train", "val", "test_id", "test_family", "test_ood"},
            "absolute_property_records_are_not_local_delta_ground_truth": True,
            "role_scope": {
                "test_context": "source_matched context_delta plus absolute_property_context_shift; no nucleotide edit context holdout",
                "test_assay": "source_matched assay_delta plus absolute_property_assay_shift; no nucleotide edit assay holdout",
                "test_family": "raw-library family_cluster holdout; cargo/protein-family labels are not available in this source",
            }.get(role, "source_matched_or_distribution_split"),
        }
        p = out_dir / "manifests" / f"{role}.json"
        p.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifest_paths[role] = str(p.relative_to(out_dir))

    prospective = {
        "schema_version": "nmi_benchmark_v2_prospective_v1",
        "frozen": False,
        "label_access": "unavailable_until_post_development_freeze",
        "record_count": 0,
        "required_source_matched_fields": REQUIRED_SOURCE_MATCHED_FIELDS,
    }
    (out_dir / "manifests" / "prospective.json").write_text(json.dumps(prospective, indent=2, sort_keys=True) + "\n")

    registry = {
        "schema_version": "nmi_benchmark_v2",
        "source_inputs": [str(p) for p in input_paths],
        "additional_raw_inputs": ["data/raw/sample2019_mpra/GSM3130443_designed_library.csv.gz"],
        "records_path": str(records_path.relative_to(out_dir)),
        "records_sha256": sha256_file(records_path),
        "total_records": sum(counts.values()),
        "counts_by_role": dict(sorted(counts.items())),
        "confidence_by_role": {f"{r}:{c}": n for (r, c), n in sorted(confidence.items())},
        "task_kinds_by_role": {r: dict(sorted({k[1]: n for k, n in task_kinds.items() if k[0] == r}.items())) for r in ROLES},
        "source_counts": {r: len(v) for r, v in sorted(sources_by_role.items())},
        "family_counts": {r: len(v) for r, v in sorted(family_by_role.items())},
        "contexts_by_role": {r: sorted(v) for r, v in sorted(contexts_by_role.items())},
        "assays_by_role": {r: sorted(v) for r, v in sorted(assays_by_role.items())},
        "required_source_matched_fields": REQUIRED_SOURCE_MATCHED_FIELDS,
        "manifests": manifest_paths,
        "four_layers": build_asset_registry(data_root),
        "prospective_manifest": "manifests/prospective.json",
        "final_test_roles": sorted(FINAL_ROLES),
        "final_test_policy": "loader refuses final roles without explicit allow_final_labels",
        "proxy_is_biological_ground_truth": False,
        "legacy_excluded_split_counts": dict(sorted(legacy_excluded.items())),
        "untouched_test_policy": "historical P3 test/ood/val labels are never promoted to v2 final; final local-edit roles come from raw designed-library rows excluded by historical measured source/candidate membership",
        "family_axis_policy": "test_family uses the raw designed-library family_cluster_id; it is not a cargo/protein-family claim",
        "claim_policy": "absolute-property context/assay records do not authorize local-delta or SOTA claims",
    }
    (out_dir / "registry.json").write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    return registry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True, help="source P3 JSONL; repeatable")
    ap.add_argument("--out-dir", default="data/nmi_benchmark_v2")
    ap.add_argument("--data-root", default=".", help="repository root containing data/raw")
    ap.add_argument("--mpra-train-limit", type=int, default=1000)
    ap.add_argument("--mpra-assay-limit", type=int, default=1000)
    ap.add_argument("--cao-train-limit", type=int, default=1000)
    ap.add_argument("--cao-context-limit", type=int, default=1000)
    args = ap.parse_args()
    registry = build(
        [Path(p) for p in args.input], Path(args.out_dir), data_root=Path(args.data_root),
        mpra_train_limit=args.mpra_train_limit, mpra_assay_limit=args.mpra_assay_limit,
        cao_train_limit=args.cao_train_limit, cao_context_limit=args.cao_context_limit,
    )
    print(json.dumps(registry, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
