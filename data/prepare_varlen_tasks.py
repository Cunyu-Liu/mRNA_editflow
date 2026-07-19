"""Variable-length downstream task construction for mRNA-EditFlow.

The tasks are intentionally edit-centric:

* T5: minimal-edit synonymous CDS recoding with protein identity constraint.
* T6: UTR length-target editing with an explicit target length.
* T7: functional-element insertion and removal.

Each sample stores ``source``, ``target`` when available, and a constraints
dictionary with a recomputable minimal edit budget.
"""
from __future__ import annotations

import json
from typing import Iterable, List, Mapping, Optional, Sequence

from mrna_editflow.core.constants import translate
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.augment import synonymously_perturb_cds
from mrna_editflow.data.download_mrna import synthesize_corpus
from mrna_editflow.data.element_library import find_motifs, insert_element

TASK_T5 = "T5_MIN_EDIT_CDS_RECODING"
TASK_T6 = "T6_LENGTH_TARGET_UTR"
TASK_T7 = "T7_ELEMENT_INSERT_REMOVE"


def levenshtein_distance(a: str, b: str) -> int:
    """Unit-cost edit distance using O(min(len(a), len(b))) memory."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                )
            )
        prev = cur
    return prev[-1]


def _sample(
    task_id: str,
    record: MRNARecord,
    source: str,
    target: Optional[str],
    constraints: Mapping[str, object],
) -> dict:
    item = {
        "task_id": task_id,
        "task_group": task_id.split("_", 1)[0],
        "record_id": record.transcript_id,
        "source": source,
        "target": target,
        "constraints": dict(constraints),
    }
    if target is not None:
        budget = levenshtein_distance(source, target)
        item["constraints"].setdefault("minimal_edit_budget", budget)
        item["constraints"].setdefault("max_edit_budget", budget)
    return item


def make_t5_min_edit_sample(record: MRNARecord, seed: int = 0) -> dict:
    """T5: synonymous CDS recoding under exact protein identity."""
    target_cds = synonymously_perturb_cds(record.cds, edit_fraction=0.12, seed=seed)
    source = record.seq
    target = record.five_utr + target_cds + record.three_utr
    source_protein = translate(record.cds)
    target_protein = translate(target_cds)
    assert source_protein == target_protein, "T5 target changed encoded protein"
    return _sample(
        TASK_T5,
        record,
        source,
        target,
        {
            "task": "T5",
            "edit_objective": "minimal_synonymous_cds_recoding",
            "region": "CDS",
            "cds_frame_locked": True,
            "protein_identity_required": True,
            "protein": source_protein[:-1],
        },
    )


def make_t6_length_sample(record: MRNARecord, delta: int = 6) -> dict:
    """T6: resize the 5'UTR to a target total length."""
    if delta == 0:
        delta = 3
    source = record.seq
    five = record.five_utr
    if delta > 0:
        insert = ("GCUAUA" * ((delta + 5) // 6))[:delta]
        pos = len(five) // 2
        new_five = five[:pos] + insert + five[pos:]
    else:
        remove = min(len(five), abs(delta))
        new_five = five[remove:]
    target = new_five + record.cds + record.three_utr
    return _sample(
        TASK_T6,
        record,
        source,
        target,
        {
            "task": "T6",
            "edit_objective": "match_length_target",
            "region": "5UTR",
            "length_target": len(target),
            "length_delta": len(target) - len(source),
            "cds_unchanged": True,
        },
    )


def make_t7_element_samples(record: MRNARecord, family: str = "polyA") -> List[dict]:
    """T7: paired insertion and removal tasks for a functional element."""
    source = record.seq
    insert_pos = len(record.three_utr) // 2
    new_three, info = insert_element(record.three_utr, family, position=insert_pos)
    inserted = record.five_utr + record.cds + new_three
    hits_after_insert = find_motifs(new_three)
    if not any(h["family"] == info["family"] and h["start"] == info["start"] for h in hits_after_insert):
        raise AssertionError("inserted functional element was not detectable")

    insert_task = _sample(
        TASK_T7,
        record,
        source,
        inserted,
        {
            "task": "T7",
            "edit_objective": "insert_functional_element",
            "action": "insert",
            "region": info["region"],
            "element_family": info["family"],
            "element_name": info["name"],
            "element_sequence": info["sequence"],
            "motif_detection_required": True,
        },
    )
    remove_task = _sample(
        TASK_T7,
        record,
        inserted,
        source,
        {
            "task": "T7",
            "edit_objective": "remove_functional_element",
            "action": "remove",
            "region": info["region"],
            "element_family": info["family"],
            "element_name": info["name"],
            "element_sequence": info["sequence"],
            "motif_detection_required": True,
        },
    )
    return [insert_task, remove_task]


def prepare_varlen_task_pairs(
    records: Optional[Iterable[MRNARecord]] = None,
    n_synthetic: int = 6,
    seed: int = 0,
    output_jsonl: Optional[str] = None,
) -> List[dict]:
    """Construct T5/T6/T7 task samples from records or synthetic fallback."""
    if records is None:
        records = synthesize_corpus(n_synthetic, seed=seed)
    recs = list(records)
    tasks: List[dict] = []
    for i, record in enumerate(recs):
        tasks.append(make_t5_min_edit_sample(record, seed=seed + i))
        delta = 6 if i % 2 == 0 else -min(6, len(record.five_utr))
        tasks.append(make_t6_length_sample(record, delta=delta))
        tasks.extend(make_t7_element_samples(record, family="polyA"))
    if output_jsonl is not None:
        write_jsonl(tasks, output_jsonl)
    return tasks


def write_jsonl(samples: Sequence[Mapping[str, object]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(dict(sample), sort_keys=True) + "\n")


__all__ = [
    "TASK_T5",
    "TASK_T6",
    "TASK_T7",
    "levenshtein_distance",
    "make_t5_min_edit_sample",
    "make_t6_length_sample",
    "make_t7_element_samples",
    "prepare_varlen_task_pairs",
    "write_jsonl",
]
