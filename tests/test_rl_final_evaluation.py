"""Unit and smoke tests for the Stage 6 final-evaluation contract."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import sys
_ROOT = Path(__file__).resolve().parents[1]
_PARENT = _ROOT.parent
for _path in (str(_PARENT), str(_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mrna_editflow.eval import rl_final_evaluation as final


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_metrics() -> dict[str, object]:
    metrics = {name: 0.1 for name in final.REQUIRED_METRICS}
    metrics["bootstrap_confidence_interval"] = {"low": 0.0, "high": 0.2}
    metrics["paired_significance_test"] = {"p_value": 0.5}
    metrics["kl_trajectory"] = [0.1]
    return metrics


class FinalEvaluationContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.oracle_paths = {}
        for name in ("training", "heldout", "alternative"):
            artifact = self.root / f"{name}_oracle.bin"; artifact.write_bytes(name.encode())
            manifest = self.root / f"{name}_oracle.json"
            manifest.write_text(json.dumps({"schema_version": 1, "oracle_type": "frozen_predictor", "source": name, "independent": name != "training", "independence_statement": "held out" if name != "training" else "training", "artifact_path": artifact.name, "artifact_sha256": _sha(artifact)}))
            self.oracle_paths[name] = manifest

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_training_oracle_result_is_rejected(self) -> None:
        result = self.root / "result.json"
        result.write_text(json.dumps({"schema_version": 1, "method": "stage_a_editflow_only", "metadata": {name: "x" for name in final.RESULT_METADATA_FIELDS} | {"split_role": "test", "evaluation_split": "family_disjoint_test", "oracle_metadata": {"manifest_sha256": "training"}}, "metrics": _valid_metrics()}))
        row = final.validate_result(self.root, result, expected_method="stage_a_editflow_only", split_name="family_disjoint_test", training_oracle_sha="training", heldout_oracle_sha="heldout", alternative_oracle_sha="alternative")
        self.assertEqual(row["status"], "invalid")
        self.assertIn("training_oracle", str(row["reason"]))

    def test_non_test_role_is_rejected(self) -> None:
        result = self.root / "result.json"
        metadata = {name: "x" for name in final.RESULT_METADATA_FIELDS}
        metadata.update({"split_role": "train", "evaluation_split": "family_disjoint_test", "oracle_metadata": {"manifest_sha256": "heldout"}})
        result.write_text(json.dumps({"schema_version": 1, "method": "stage_a_editflow_only", "metadata": metadata, "metrics": _valid_metrics()}))
        row = final.validate_result(self.root, result, expected_method="stage_a_editflow_only", split_name="family_disjoint_test", training_oracle_sha="training", heldout_oracle_sha="heldout", alternative_oracle_sha="alternative")
        self.assertEqual(row["status"], "invalid")
        self.assertIn("test_role", str(row["reason"]))

    def test_default_preflight_is_honestly_incomplete(self) -> None:
        report = final.build_report(self.root)
        self.assertEqual(report["status"], "incomplete_preflight")
        self.assertFalse(report["paper_eligible"])
        self.assertIn("required_methods_unavailable", " ".join(report["block_reasons"]))
        self.assertEqual(set(report["result_metadata_contract"]), set(final.RESULT_METADATA_FIELDS))
        self.assertIn("report_runtime_s", report)
        self.assertNotIn("family_disjoint_train", report["results"]["stage_a_editflow_only"])
        self.assertIn("family_disjoint_test", report["results"]["stage_a_editflow_only"])

    def test_writers_include_all_required_failure_categories(self) -> None:
        report = final.build_report(self.root)
        failure = self.root / "failure.md"; ablation = self.root / "ablations.md"; overview = self.root / "overview.md"
        final.write_failure_markdown(report, failure)
        final.write_ablation_markdown(report, ablation)
        final.write_markdown(report, overview)
        text = failure.read_text()
        self.assertIn("Forced harmful edits", text)
        self.assertIn("Cases where local search beats GRPO", text)
        self.assertIn("no_kl", ablation.read_text())
        self.assertIn("No method comparison is claimed", overview.read_text())
        self.assertIn("reward_schema_version", overview.read_text())


if __name__ == "__main__":
    unittest.main()
