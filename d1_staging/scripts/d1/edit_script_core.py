"""Core edit_script functions for D1-01 canonical records.

Edit operations: INS, DEL, SUB, STOP
- INS(pos, token): insert token before position pos in current state
- DEL(pos): delete token at position pos in current state
- SUB(pos, token): substitute token at position pos with token
- STOP: end of edit trajectory (not used in canonical records, reserved for generation)

Edit script = ordered list of EditOp applied sequentially.
apply(edit_script, source) applies operations in order to produce candidate.

Contract: utr_editflow_contract_v2 (FROZEN)
Task: D1-01
"""

from dataclasses import dataclass, asdict
from typing import List, Literal, Optional
import json


@dataclass(frozen=True)
class EditOp:
    """A single edit operation in an edit script.

    Attributes:
        op: operation type ("INS", "DEL", "SUB", "STOP")
        pos: 0-indexed position in the current state at time of application
        token: nucleotide for INS/SUB; empty string for DEL/STOP
    """
    op: Literal["INS", "DEL", "SUB", "STOP"]
    pos: int
    token: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EditOp":
        return cls(op=d["op"], pos=d["pos"], token=d.get("token", ""))


def compute_edit_script(source: str, candidate: str) -> List[EditOp]:
    """Compute a minimal-length edit script from source to candidate.

    Uses standard Levenshtein edit distance DP with traceback.
    Traceback preference order: MATCH > DEL > INS > SUB (deterministic).

    Two-pass position computation: the traceback first records the alignment
    in source/candidate coordinates, then a forward pass converts positions
    to "current state at time of application" coordinates so that
    ``apply_edit_script(source, ops) == candidate`` holds 100%.

    Args:
        source: source sequence (nucleotides)
        candidate: candidate sequence (nucleotides)

    Returns:
        List of EditOp (excluding STOP) that transforms source into candidate.
        Empty list if source == candidate.
    """
    n, m = len(source), len(candidate)

    # dp[i][j] = min edits to transform source[:i] into candidate[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i  # i DELs
    for j in range(1, m + 1):
        dp[0][j] = j  # j INSs

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if source[i - 1] == candidate[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # MATCH
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # DEL source[i-1]
                    dp[i][j - 1],      # INS candidate[j-1]
                    dp[i - 1][j - 1],  # SUB source[i-1] -> candidate[j-1]
                )

    # Pass 1: traceback to build alignment in reverse, then reverse to
    # get forward order. Preference: MATCH > DEL > INS > SUB (deterministic).
    alignment: List[str] = []  # forward-order op-type strings
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and source[i - 1] == candidate[j - 1]:
            alignment.append("MATCH")
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append("DEL")
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            alignment.append("INS")
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            alignment.append("SUB")
            i -= 1
            j -= 1
        else:
            raise RuntimeError(
                f"Traceback dead end at ({i},{j}): dp={dp[i][j]}, "
                f"source={source}, candidate={candidate}"
            )

    alignment.reverse()  # now in forward (application) order

    # Pass 2: walk the forward alignment to compute positions in
    # current-state coordinates. ``cur`` tracks the write cursor in the
    # evolving state:
    #   MATCH / SUB / INS -> advance cur by 1 (char consumed/added)
    #   DEL -> cur stays (char removed, next char slides into cur)
    ops: List[EditOp] = []
    si = 0  # source index
    ci = 0  # candidate index
    cur = 0  # position in current state
    for atype in alignment:
        if atype == "MATCH":
            si += 1
            ci += 1
            cur += 1
        elif atype == "DEL":
            ops.append(EditOp("DEL", cur, ""))
            si += 1
            # cur stays: the deleted char is gone, next source char now at cur
        elif atype == "INS":
            ops.append(EditOp("INS", cur, candidate[ci]))
            ci += 1
            cur += 1
        elif atype == "SUB":
            ops.append(EditOp("SUB", cur, candidate[ci]))
            si += 1
            ci += 1
            cur += 1

    return ops


def apply_edit_script(source: str, ops: List[EditOp]) -> str:
    """Apply edit script to source to produce candidate.

    Operations are applied sequentially. Each position is relative to the
    current state at the time of application.

    Args:
        source: source sequence
        ops: list of EditOp to apply

    Returns:
        The resulting candidate sequence after applying all operations.
    """
    result = list(source)
    for op in ops:
        if op.op == "INS":
            result.insert(op.pos, op.token)
        elif op.op == "DEL":
            if op.pos < len(result):
                result.pop(op.pos)
            else:
                raise IndexError(
                    f"DEL at pos {op.pos} out of range (len={len(result)})"
                )
        elif op.op == "SUB":
            if op.pos < len(result):
                result[op.pos] = op.token
            else:
                raise IndexError(
                    f"SUB at pos {op.pos} out of range (len={len(result)})"
                )
        elif op.op == "STOP":
            break
        else:
            raise ValueError(f"Unknown operation type: {op.op}")
    return "".join(result)


def count_optimal_alignments(source: str, candidate: str) -> int:
    """Count the number of distinct minimal-length edit scripts.

    This quantifies path ambiguity: how many different minimal edit scripts
    can transform source into candidate.

    Uses DP counting the number of optimal paths through the Levenshtein matrix.

    Args:
        source: source sequence
        candidate: candidate sequence

    Returns:
        Number of distinct minimal-length edit scripts (>= 1).
    """
    n, m = len(source), len(candidate)

    # dp[i][j] = min edit distance
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    # cnt[i][j] = number of optimal paths to reach (i, j)
    cnt = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        cnt[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = j
        cnt[0][j] = 1
    cnt[0][0] = 1

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if source[i - 1] == candidate[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                # Match is always optimal, but DEL/INS may also be optimal
                # (e.g. "AAA"->"AAAA": the extra A can be an INS at any gap).
                c = cnt[i - 1][j - 1]  # MATCH path
                if dp[i][j] == dp[i - 1][j] + 1:  # DEL also optimal
                    c += cnt[i - 1][j]
                if dp[i][j] == dp[i][j - 1] + 1:  # INS also optimal
                    c += cnt[i][j - 1]
                # SUB (dp[i-1][j-1]+1) is never optimal when match exists
                cnt[i][j] = c
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1],
                )
                c = 0
                if dp[i][j] == dp[i - 1][j] + 1:  # DEL
                    c += cnt[i - 1][j]
                if dp[i][j] == dp[i][j - 1] + 1:  # INS
                    c += cnt[i][j - 1]
                if dp[i][j] == dp[i - 1][j - 1] + 1:  # SUB
                    c += cnt[i - 1][j - 1]
                cnt[i][j] = c

    return cnt[n][m]


def edit_distance(source: str, candidate: str) -> int:
    """Compute Levenshtein edit distance between source and candidate."""
    n, m = len(source), len(candidate)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            if source[i - 1] == candidate[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr
    return prev[m]


def summarize_edit_script(ops: List[EditOp]) -> dict:
    """Summarize edit script statistics.

    Returns:
        dict with counts: n_ins, n_del, n_sub, n_total, ops_summary
    """
    n_ins = sum(1 for op in ops if op.op == "INS")
    n_del = sum(1 for op in ops if op.op == "DEL")
    n_sub = sum(1 for op in ops if op.op == "SUB")
    return {
        "n_ins": n_ins,
        "n_del": n_del,
        "n_sub": n_sub,
        "n_total": len(ops),
        "ops_summary": [op.to_dict() for op in ops],
    }


def canonical_record(
    record_id: str,
    dataset: str,
    accession: str,
    region: str,
    source: str,
    candidate: str,
    labels: dict,
    metadata: Optional[dict] = None,
) -> dict:
    """Build a canonical record with verified edit script and path ambiguity.

    Args:
        record_id: unique record identifier
        dataset: dataset name (e.g., "sample2019")
        accession: GEO/ENCODE accession
        region: "5'UTR" or "3'UTR"
        source: source sequence
        candidate: candidate sequence
        labels: dict of endpoint -> value
        metadata: optional extra metadata

    Returns:
        dict with all canonical record fields including verified edit_script
        and path_ambiguity.
    """
    ops = compute_edit_script(source, candidate)
    verified = apply_edit_script(source, ops) == candidate
    ambiguity = count_optimal_alignments(source, candidate)
    stats = summarize_edit_script(ops)

    record = {
        "record_id": record_id,
        "dataset": dataset,
        "accession": accession,
        "region": region,
        "source_sequence": source,
        "candidate_sequence": candidate,
        "edit_script": [op.to_dict() for op in ops],
        "edit_script_verified": verified,
        "edit_distance": stats["n_total"],
        "n_ins": stats["n_ins"],
        "n_del": stats["n_del"],
        "n_sub": stats["n_sub"],
        "path_ambiguity": ambiguity,
        "labels": labels,
        "metadata": metadata or {},
    }
    return record


def canonical_record_no_edit(
    record_id: str,
    dataset: str,
    accession: str,
    region: str,
    sequence: str,
    labels: dict,
    metadata: Optional[dict] = None,
) -> dict:
    """Build a canonical record for observational data (no source-candidate pair).

    Used for D_A datasets (unlabeled/observational) where no edit script exists.
    """
    return {
        "record_id": record_id,
        "dataset": dataset,
        "accession": accession,
        "region": region,
        "source_sequence": None,
        "candidate_sequence": sequence,
        "edit_script": [],
        "edit_script_verified": True,
        "edit_distance": 0,
        "n_ins": 0,
        "n_del": 0,
        "n_sub": 0,
        "path_ambiguity": 1,
        "labels": labels,
        "metadata": {**(metadata or {}), "record_type": "observational"},
    }
