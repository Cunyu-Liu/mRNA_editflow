"""P3-00A contract tests: frozen publication contract alignment.

Guards the amended P3-00 deliverables (p3_task_v2 / p3_contract_v2) against
silent edits:

  - configs/p3_primary_task.yaml — method/task hierarchy: paper primary
    method covers the full mRNA grammar flow; active Task A carries ONLY
    five_utr_substitution; Task B is frozen fallback with ONLY synonymous
    CDS substitution; Task C is locked behind H1/H2/H3 in BOTH regions;
    Task D rejected; motif policy is versioned (motif_policy_v1).
  - configs/p3_frozen_research_contract.yaml — master contract; Generic
    FlexFlow-mRNA registered as closest baseline; RL is not a paper blocker.
  - All gate files carry phase_status=PASS together with
    scientific_validation_status=PENDING (P3-00 PASS never implies H1-H6).
  - docs/p3_00_scientific_landscape.tsv — 12 columns, >= 13 studies
    including every spec-required study.
  - docs/p3_00_go_no_go_matrix.json — verdict PASS, corrected totals
    (A=41, B=29, C=29, D=21), P3-00A acceptance criteria PASS, and embedded
    SHA-256 hashes match the files on disk. Hash verification canonicalizes
    newlines (CRLF -> LF) and resolves paths relative to the repo root, so
    neither line-ending style nor caller cwd affects verification.

Zero-dependency: the two config files are parsed with a restricted
indent-based YAML-subset reader (PyYAML is not a repo dependency). Anything
outside the subset raises ValueError so unexpected structure fails closed
rather than being silently misread.

Run:
    python -m pytest tests/test_p3_00_task_contract.py -v
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TASK_CONFIG = _REPO_ROOT / "configs" / "p3_primary_task.yaml"
_CONTRACT = _REPO_ROOT / "configs" / "p3_frozen_research_contract.yaml"
_TSV = _REPO_ROOT / "docs" / "p3_00_scientific_landscape.tsv"
_MATRIX = _REPO_ROOT / "docs" / "p3_00_go_no_go_matrix.json"
_GATE_DOCS = (
    "docs/p3_00_frozen_scientific_question.md",
    "docs/p3_00_claim_ladder.md",
    "docs/p3_00_change_governance.md",
    "docs/p3_00_scientific_problem_lock.md",
    "docs/p3_00_hypothesis_preregistration.md",
)

_EXPECTED_METHOD_SCOPE = {
    "full_length_mrna_modeling",
    "protein_conditioned_cds_generation",
    "region_infilling",
    "source_conditioned_editing",
}
_TASK_C_UNLOCK = {
    "H1_pass_in_five_utr",
    "H2_pass_in_five_utr",
    "H3_pass_in_five_utr",
    "H1_pass_in_cds",
    "H2_pass_in_cds",
    "H3_pass_in_cds",
}


def _read_yaml_subset(path: Path) -> dict:
    """Parse the restricted YAML subset used by the P3 frozen configs.

    Supports 2-space-indented nested mappings (``key: value`` /
    ``key:`` + deeper block) and ``- item`` scalar lists. Anything else
    raises ValueError (fail closed).
    """
    entries: list[tuple[int, str, int]] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        entries.append((indent, line.strip(), lineno))
    if not entries:
        raise ValueError(f"{path}: empty document")

    pos = 0

    def parse_block(indent: int):
        nonlocal pos
        if entries[pos][1].startswith("- "):
            items: list = []
            while (
                pos < len(entries)
                and entries[pos][0] == indent
                and entries[pos][1].startswith("- ")
            ):
                item = entries[pos][1][2:].strip()
                items.append(int(item) if re.fullmatch(r"-?\d+", item) else item)
                pos += 1
            return items
        mapping: dict[str, object] = {}
        while (
            pos < len(entries)
            and entries[pos][0] == indent
            and not entries[pos][1].startswith("- ")
        ):
            text, lineno = entries[pos][1], entries[pos][2]
            m = re.fullmatch(r"([A-Za-z0-9_]+):\s*(.*)", text)
            if not m:
                raise ValueError(f"{path} line {lineno}: unparseable: {text!r}")
            key, value = m.group(1), m.group(2).strip()
            pos += 1
            if value:
                mapping[key] = value
            elif pos < len(entries) and entries[pos][0] > indent:
                mapping[key] = parse_block(entries[pos][0])
            else:
                mapping[key] = None
        return mapping

    doc = parse_block(entries[0][0])
    if pos != len(entries):
        raise ValueError(f"{path} line {entries[pos][2]}: trailing unparsed content")
    return doc


def _sha256_canonical(path: Path) -> str:
    """SHA-256 over newline-canonicalized bytes (CRLF -> LF)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


class TestPrimaryTaskHierarchy(unittest.TestCase):
    """P3-00A Task 1: method/task hierarchy in the primary task contract."""

    @classmethod
    def setUpClass(cls) -> None:
        if not _TASK_CONFIG.exists():
            raise FileNotFoundError(f"missing frozen contract: {_TASK_CONFIG}")
        cls.doc = _read_yaml_subset(_TASK_CONFIG)

    def test_task_identity_and_freeze(self) -> None:
        self.assertEqual(self.doc["task_name"], "source_conditioned_minimal_edit_protein_output")
        self.assertEqual(self.doc["primary_endpoint"], "protein_output")
        self.assertEqual(self.doc["contract_status"], "frozen")
        self.assertEqual(self.doc["task_version"], "p3_task_v2")

    def test_paper_primary_method_covers_full_mrna_grammar_flow(self) -> None:
        method = self.doc["paper_primary_method"]
        self.assertEqual(method["name"], "heterogeneous_grammar_constrained_mrna_edit_flow")
        self.assertEqual(set(method["scope"]), _EXPECTED_METHOD_SCOPE)

    def test_active_task_is_task_a(self) -> None:
        self.assertEqual(self.doc["active_task"], "task_a")
        self.assertEqual(self.doc["application_tasks"]["task_a"]["status"], "active_primary")

    def test_task_a_only_five_utr_substitution(self) -> None:
        task_a = self.doc["application_tasks"]["task_a"]
        self.assertEqual(task_a["allowed_actions"], ["five_utr_substitution"])
        # Operational allowed_actions describe the ACTIVE task only.
        self.assertEqual(self.doc["allowed_actions"], ["five_utr_substitution"])
        for action in self.doc["allowed_actions"] + task_a["allowed_actions"]:
            self.assertNotIn("cds", action.lower())

    def test_task_b_only_synonymous_cds_substitution(self) -> None:
        task_b = self.doc["application_tasks"]["task_b"]
        self.assertEqual(task_b["status"], "frozen_fallback")
        self.assertEqual(task_b["allowed_actions"], ["synonymous_cds_substitution"])

    def test_task_c_locked_by_default(self) -> None:
        self.assertEqual(self.doc["application_tasks"]["task_c"]["status"], "locked_extension")

    def test_task_c_unlock_requires_h1_h2_h3_in_both_regions(self) -> None:
        unlock = set(self.doc["application_tasks"]["task_c"]["unlock_requires"])
        self.assertEqual(unlock, _TASK_C_UNLOCK)

    def test_task_d_rejected_for_primary_paper(self) -> None:
        self.assertEqual(
            self.doc["application_tasks"]["task_d"]["status"], "rejected_for_primary_paper"
        )

    def test_phase_pass_and_scientific_pending_coexist(self) -> None:
        self.assertEqual(self.doc["phase_status"], "PASS")
        self.assertEqual(self.doc["scientific_validation_status"], "PENDING")

    def test_motif_policy_versioned_three_tiers(self) -> None:
        policy = self.doc["motif_policy"]
        self.assertEqual(policy["version"], "motif_policy_v1")
        for tier in ("hard_forbidden", "guarded_risk", "soft_objective"):
            self.assertIn(tier, policy)
            self.assertGreater(len(policy[tier]), 0, tier)
        hard = set(self.doc["hard_constraints"])
        self.assertIn("motif_policy_v1_hard_forbidden", hard)
        self.assertNotIn("no_forbidden_motif", hard)

    def test_extensions_are_conditional_not_blockers(self) -> None:
        self.assertEqual(self.doc["three_utr_status"], "locked_extension")
        self.assertEqual(self.doc["rl_status"], "conditional_extension")
        self.assertEqual(self.doc["synergy_status"], "conditional_extension")

    def test_forbidden_actions_cover_spec(self) -> None:
        required = {
            "nonsynonymous_cds_substitution",
            "cds_insertion",
            "cds_deletion",
            "utr_insertion",
            "utr_deletion",
            "three_utr_edit",
        }
        self.assertTrue(required <= set(self.doc["forbidden_actions"]))

    def test_allowed_forbidden_disjoint(self) -> None:
        self.assertFalse(set(self.doc["allowed_actions"]) & set(self.doc["forbidden_actions"]))

    def test_edit_budgets(self) -> None:
        self.assertEqual(self.doc["edit_budgets"], [1, 3, 5, 10])


class TestFrozenResearchContract(unittest.TestCase):
    """P3-00A Task 2: master frozen research contract."""

    @classmethod
    def setUpClass(cls) -> None:
        if not _CONTRACT.exists():
            raise FileNotFoundError(f"missing frozen research contract: {_CONTRACT}")
        cls.doc = _read_yaml_subset(_CONTRACT)

    def test_closest_baseline_registered(self) -> None:
        self.assertEqual(self.doc["closest_baseline"], "generic_flexflow_mrna")
        self.assertEqual(
            self.doc["closest_baseline_reason"],
            "same_model_class_without_grammar_constrained_edit_contract",
        )

    def test_method_scope_matches_task_contract(self) -> None:
        method = self.doc["paper_primary_method"]
        self.assertEqual(method["name"], "heterogeneous_grammar_constrained_mrna_edit_flow")
        self.assertEqual(set(method["scope"]), _EXPECTED_METHOD_SCOPE)

    def test_application_task_statuses(self) -> None:
        tasks = self.doc["application_tasks"]
        self.assertEqual(tasks["task_a"]["status"], "active_primary")
        self.assertEqual(tasks["task_b"]["status"], "frozen_fallback")
        self.assertEqual(tasks["task_c"]["status"], "locked_extension")
        self.assertEqual(set(tasks["task_c"]["unlock_requires"]), _TASK_C_UNLOCK)
        self.assertEqual(tasks["task_d"]["status"], "rejected_for_primary_paper")

    def test_rl_not_paper_blocker(self) -> None:
        self.assertEqual(self.doc["rl_is_paper_blocker"], "false")
        self.assertEqual(self.doc["rl_status"], "conditional_extension")
        self.assertEqual(self.doc["three_utr_status"], "locked_extension")
        self.assertEqual(self.doc["synergy_status"], "conditional_extension")

    def test_phase_pass_and_scientific_pending_coexist(self) -> None:
        self.assertEqual(self.doc["phase_status"], "PASS")
        self.assertEqual(self.doc["scientific_validation_status"], "PENDING")

    def test_governance_references_resolve(self) -> None:
        for key in (
            "task_contract",
            "scientific_question_doc",
            "claim_ladder_doc",
            "change_governance_doc",
            "hypothesis_preregistration",
            "task_scoring",
            "gate_matrix",
        ):
            self.assertTrue((_REPO_ROOT / self.doc[key]).exists(), self.doc[key])


class TestGateDocsStatus(unittest.TestCase):
    """P3-00A Task 3: phase PASS / scientific PENDING separation everywhere."""

    def test_status_headers_in_all_gate_docs(self) -> None:
        for rel in _GATE_DOCS:
            text = (_REPO_ROOT / rel).read_text(encoding="utf-8")
            self.assertIn("phase_status**: PASS", text, rel)
            self.assertIn("scientific_validation_status**: PENDING", text, rel)

    def test_tsv_status_comments(self) -> None:
        lines = _TSV.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[0], "# phase_status: PASS")
        self.assertTrue(lines[1].startswith("# scientific_validation_status: PENDING"))

    def test_no_doc_claims_hypotheses_validated(self) -> None:
        # P3-00 PASS must never be written as H1-H6 experimental support.
        for rel in _GATE_DOCS:
            text = (_REPO_ROOT / rel).read_text(encoding="utf-8").lower()
            self.assertNotIn("h1-h6 validated", text, rel)
            self.assertNotIn("hypotheses confirmed", text, rel)


class TestScientificLandscape(unittest.TestCase):
    REQUIRED_STUDIES = (
        "PERSIST-seq",
        "UTR-LM",
        "RiboNN",
        "Optimus 5-Prime",
        "MapUTR",
        "LinearDesign",
        "EnsembleDesign",
        "codonGPT",
        "GEMORNA",
        "mRNA-GPT",
        "ProMORNA",
        "mRNAutilus",
    )
    EXPECTED_COLUMNS = (
        "study", "year", "task", "region", "source_conditioned_or_de_novo",
        "data_type", "number_of_sequences", "experimental_endpoint", "code",
        "data_availability", "applicability_to_local_edits", "major_limitation",
    )

    @classmethod
    def setUpClass(cls) -> None:
        if not _TSV.exists():
            raise FileNotFoundError(f"missing landscape matrix: {_TSV}")
        lines = [
            line
            for line in _TSV.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        cls.header = tuple(lines[0].split("\t"))
        cls.rows = [line.split("\t") for line in lines[1:]]

    def test_header_columns(self) -> None:
        self.assertEqual(self.header, self.EXPECTED_COLUMNS)

    def test_row_widths(self) -> None:
        for i, row in enumerate(self.rows, start=2):
            self.assertEqual(len(row), 12, f"row {i} has {len(row)} columns")

    def test_minimum_study_count(self) -> None:
        self.assertGreaterEqual(len(self.rows), 13)

    def test_required_studies_present(self) -> None:
        studies = "\t".join(row[0] for row in self.rows)
        for name in self.REQUIRED_STUDIES:
            self.assertIn(name, studies)

    def test_every_row_has_limitation(self) -> None:
        for row in self.rows:
            self.assertTrue(row[11].strip(), f"study {row[0]} lacks major_limitation")


class TestGoNoGoMatrix(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not _MATRIX.exists():
            raise FileNotFoundError(f"missing gate matrix: {_MATRIX}")
        cls.matrix = json.loads(_MATRIX.read_text(encoding="utf-8"))

    def test_verdict_and_scope(self) -> None:
        self.assertEqual(self.matrix["phase"], "P3-00")
        self.assertEqual(self.matrix["verdict"], "PASS")
        self.assertEqual(self.matrix["p3_00a_verdict"], "PASS")
        self.assertFalse(self.matrix["scope_limits"]["grpo_modified"])
        self.assertFalse(self.matrix["scope_limits"]["large_scale_training_started"])
        self.assertFalse(self.matrix["scope_limits"]["full_transcript_synergy_assumed"])

    def test_phase_pass_and_scientific_pending_coexist(self) -> None:
        self.assertEqual(self.matrix["phase_status"], "PASS")
        self.assertEqual(self.matrix["scientific_validation_status"], "PENDING")

    def test_all_acceptance_criteria_pass(self) -> None:
        criteria = self.matrix["acceptance_criteria"]
        self.assertEqual(len(criteria), 6)
        for criterion in criteria:
            self.assertEqual(criterion["status"], "PASS", criterion["id"])

    def test_p3_00a_acceptance_criteria_pass(self) -> None:
        criteria = self.matrix["p3_00a_acceptance_criteria"]
        self.assertEqual(len(criteria), 6)
        for criterion in criteria:
            self.assertEqual(criterion["status"], "PASS", criterion["id"])

    def test_task_scores_corrected_totals(self) -> None:
        scores = self.matrix["task_scores"]
        totals = scores["totals"]
        self.assertEqual(totals, {"A": 41, "B": 29, "C": 29, "D": 21, "max": 45})
        for key, letter in (
            ("task_a_5utr_only", "A"),
            ("task_b_cds_only_synonymous", "B"),
            ("task_c_5utr_cds_joint", "C"),
            ("task_d_full_transcript", "D"),
        ):
            self.assertEqual(sum(scores[key]), totals[letter], key)

    def test_scoring_robustness_recorded(self) -> None:
        robustness = self.matrix["task_scores"]["robustness"]
        self.assertFalse(robustness["hard_floor"]["eligible"]["D"])
        for letter in ("A", "B", "C"):
            self.assertTrue(robustness["hard_floor"]["eligible"][letter], letter)
        loco = robustness["leave_one_criterion_out"]
        self.assertEqual(loco["a_first_in_scenarios"], loco["scenarios"])
        self.assertTrue(loco["b_c_order_swaps_recorded"])
        alt = robustness["alternative_weights"]
        self.assertEqual(alt["a_first_in_scenarios"], alt["scenarios"])

    def test_task_decision_aligned_with_contract(self) -> None:
        decision = self.matrix["task_decision"]
        self.assertIn("Task A", decision["active_primary"])
        self.assertIn("five_utr_substitution", decision["active_primary"])
        self.assertIn("Task B", decision["frozen_fallback"])
        self.assertIn("H1", decision["locked_extension"])
        self.assertIn("H2", decision["locked_extension"])
        self.assertIn("H3", decision["locked_extension"])
        self.assertIn("Task D", decision["rejected_for_primary_paper"])

    def test_closest_baseline_registered(self) -> None:
        self.assertEqual(self.matrix["closest_baseline"]["name"], "generic_flexflow_mrna")

    def test_hypotheses_falsifiable(self) -> None:
        hypotheses = self.matrix["hypotheses"]
        self.assertEqual(len(hypotheses), 6)
        for key, hyp in hypotheses.items():
            self.assertIn("no_go_action", hyp, key)
        self.assertTrue(hypotheses["H5_cross_region_synergy"]["optional"])
        self.assertTrue(hypotheses["H6_full_transcript_extension_value"]["optional"])

    def test_rl_pause_conditions_nonempty(self) -> None:
        self.assertGreaterEqual(len(self.matrix["rl_pause_conditions"]), 5)

    def test_embedded_hashes_match_files(self) -> None:
        seen = set()
        for entry in self.matrix["outputs"]:
            path = _REPO_ROOT / entry["path"]
            self.assertTrue(path.exists(), entry["path"])
            if entry["path"].endswith("go_no_go_matrix.json"):
                continue  # self-referential
            seen.add(entry["path"])
            self.assertTrue(re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), entry["path"])
            self.assertEqual(
                _sha256_canonical(path),
                entry["sha256"],
                f"{entry['path']} changed after matrix freeze; re-freeze both together",
            )
        # All frozen artifacts must be covered, not just a subset.
        expected = {
            "configs/p3_primary_task.yaml",
            "configs/p3_frozen_research_contract.yaml",
            "docs/p3_00_scientific_landscape.tsv",
            "docs/p3_00_hypothesis_preregistration.md",
            "docs/p3_00_scientific_problem_lock.md",
            "docs/p3_00_frozen_scientific_question.md",
            "docs/p3_00_claim_ladder.md",
            "docs/p3_00_change_governance.md",
            "tests/test_p3_00_task_contract.py",
        }
        self.assertEqual(seen, expected)

    def test_hashes_newline_and_relative_path_independent(self) -> None:
        # Canonical newlines: no frozen artifact may contain CRLF, and the
        # canonicalized digest must match regardless.
        for entry in self.matrix["outputs"]:
            path = _REPO_ROOT / entry["path"]
            if entry["path"].endswith("go_no_go_matrix.json"):
                continue
            raw = path.read_bytes()
            self.assertNotIn(b"\r\n", raw, entry["path"])
            self.assertEqual(
                hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(),
                hashlib.sha256(raw).hexdigest(),
                entry["path"],
            )
        # Relative-path independence: verification resolves paths against the
        # repo root, so the caller's cwd must not matter.
        cwd = os.getcwd()
        try:
            os.chdir(tempfile.gettempdir())
            for entry in self.matrix["outputs"]:
                if entry["path"].endswith("go_no_go_matrix.json"):
                    continue
                digest = _sha256_canonical(_REPO_ROOT / entry["path"])
                self.assertEqual(digest, entry["sha256"], entry["path"])
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
