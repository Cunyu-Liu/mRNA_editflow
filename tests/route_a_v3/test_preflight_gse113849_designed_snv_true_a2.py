from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


STAGING_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    STAGING_ROOT
    / "scripts"
    / "route_a_v3"
    / "preflight_gse113849_designed_snv_true_a2.py"
)
PROTOCOL_PATH = (
    STAGING_ROOT
    / "configs"
    / "route_a_v3_gse113849_designed_snv_true_a2_preflight_v1.json"
)


def _load_subject():
    spec = importlib.util.spec_from_file_location("gse113849_preflight", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("subject module not loadable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


subject = _load_subject()


def _bound_protocol() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    bindings = protocol["bindings"]
    bindings["implementation"].update(
        {
            "status": subject.BOUND,
            "implementation_commit": "7" * 40,
            "implementation_script_sha256": hashlib.sha256(
                SCRIPT_PATH.read_bytes()
            ).hexdigest(),
            "implementation_test_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        }
    )
    return protocol


def _small_rows():
    source = "AAAA"
    return [
        {
            "gene": "GENE1",
            "clinvar_id": "Missing",
            "significance": "Missing",
            "in_acmg": "True",
            "sitetype": "UTR3",
            "wt_seq": source,
            "master_seq": "CAAA",
            "snv_pos": "0",
            "delta_logodds_true": "0.2",
            "delta_logodds_pred": "not-read",
            "delta_p_val": "0.01",
        },
        {
            "gene": "GENE1",
            "clinvar_id": "Missing",
            "significance": "Missing",
            "in_acmg": "True",
            "sitetype": "UTR3",
            "wt_seq": source,
            "master_seq": "ACAA",
            "snv_pos": "1",
            "delta_logodds_true": "-0.3",
            "delta_logodds_pred": "not-read",
            "delta_p_val": "0.02",
        },
        {
            "gene": "GENE1",
            "clinvar_id": "Missing",
            "significance": "Missing",
            "in_acmg": "True",
            "sitetype": "UTR3",
            "wt_seq": source,
            "master_seq": "AACA",
            "snv_pos": "2",
            "delta_logodds_true": "0.0",
            "delta_logodds_pred": "not-read",
            "delta_p_val": "0.5",
        },
        {
            "gene": "GENE2",
            "clinvar_id": "Missing",
            "significance": "Missing",
            "in_acmg": "False",
            "sitetype": "UTR3",
            "wt_seq": source,
            "master_seq": "AAAC",
            "snv_pos": "3",
            "delta_logodds_true": "1.1",
            "delta_logodds_pred": "not-read",
            "delta_p_val": "0.0",
        },
    ]


def _small_protocol(base_protocol):
    protocol = copy.deepcopy(base_protocol)
    protocol["expected_mechanical_replay"] = {
        "table_row_count": 4,
        "distinct_source_sequence_count": 1,
        "distinct_gene_source_group_count": 2,
        "source_sequences_assigned_to_multiple_genes_count": 1,
        "distinct_source_candidate_pair_count": 4,
        "equal_length_pair_count": 4,
        "exact_hamming_one_pair_count": 4,
        "snv_position_consistent_pair_count": 4,
        "missing_required_source_candidate_endpoint_pvalue_count": 0,
        "finite_endpoint_count": 4,
        "valid_pvalue_count": 4,
        "dense_gene_source_group_count": 1,
        "rows_in_dense_gene_source_groups": 3,
        "maximum_dense_candidate_pool_size": 3,
        "explicit_utr3_row_count": 4,
        "explicit_utr3_gene_source_group_count": 2,
        "explicit_utr3_dense_gene_source_group_count": 1,
        "explicit_utr3_rows_in_dense_gene_source_groups": 3,
        "explicit_utr3_maximum_candidate_pool_size": 3,
        "published_biological_replicate_count": 2,
        "replicate_derived_standard_error_field_count": 0,
    }
    return protocol


class GSE113849PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        cls.small_protocol = _small_protocol(cls.protocol)
        cls.rows = _small_rows()
        cls.semantics = {
            "repository_license_scope": "SOFTWARE_AND_ASSOCIATED_DOCUMENTATION"
        }
        cls.report = subject._build_report(
            cls.small_protocol,
            cls.rows,
            cls.semantics,
            "2026-08-15T13:00:00+08:00",
        )

    def test_protocol_freezes_exact_thirteen_gates_context_and_random_exclusion(self):
        subject._validate_protocol(self.protocol)
        self.assertEqual(
            tuple(self.protocol["required_fail_closed_gate_ids_exactly"]),
            subject.REQUIRED_GATE_IDS,
        )
        context = self.protocol["outcome_blind_context_freeze"]
        self.assertEqual(context["selected_rule"], subject.SELECTED_CONTEXT_RULE)
        self.assertFalse(context["outcome_or_power_may_change_selected_rule"])
        self.assertEqual(
            self.protocol["intended_universe"]["randomized_absolute_library"],
            "EXCLUDED_NOT_TRUE_A2",
        )
        expected = self.protocol["expected_mechanical_replay"]
        self.assertEqual(expected["distinct_gene_source_group_count"], 1353)
        self.assertEqual(expected["dense_gene_source_group_count"], 532)
        self.assertEqual(expected["rows_in_dense_gene_source_groups"], 12677)
        self.assertEqual(expected["published_biological_replicate_count"], 2)
        self.assertEqual(expected["replicate_derived_standard_error_field_count"], 0)
        bindings = self.protocol["bindings"]
        self.assertEqual(
            tuple(
                step["label"]
                for step in bindings["gse217518_predecessor"]["append_only_history"]
            ),
            subject.GSE217_HISTORY_LABELS,
        )
        self.assertEqual(
            tuple(
                step["commit"]
                for step in bindings["encsr854ruf_predecessor"]["append_only_history"]
            ),
            subject.ENCSR_HISTORY_COMMITS,
        )
        self.assertEqual(
            tuple(
                step["commit"]
                for step in bindings["gse232572_predecessor"]["append_only_history"]
            ),
            subject.GSE232_HISTORY_COMMITS,
        )
        self.assertEqual(bindings["gse232572_predecessor"]["status"], subject.BOUND)

    def test_production_stops_before_asset_loader_or_output(self):
        calls = {"git": 0, "asset": 0, "output": 0}

        def poison(name):
            def callback(*args, **kwargs):
                calls[name] += 1
                raise AssertionError(f"{name} crossed grouped-UNKNOWN barrier")

            return callback

        with tempfile.TemporaryDirectory() as temp_dir:
            unknown_protocol = subject._normalise_own_binding(self.protocol)
            unknown_protocol["bindings"]["gse232572_predecessor"] = {
                "status": subject.UNKNOWN,
                "append_only_history": subject.UNKNOWN,
            }
            protocol_path = Path(temp_dir) / "unknown_protocol.json"
            protocol_path.write_text(
                json.dumps(unknown_protocol, indent=2) + "\n", encoding="utf-8"
            )
            asset_dir = Path(temp_dir) / "must_not_be_read"
            output_dir = Path(temp_dir) / "must_not_exist"
            with mock.patch.object(
                subject, "_audit_repository_bindings", poison("git")
            ), mock.patch.object(
                subject, "_load_and_replay", poison("asset")
            ), mock.patch.object(subject, "_write_report", poison("output")):
                with self.assertRaises(subject.BindingNotReady) as caught:
                    subject.execute_production(
                        protocol_path=protocol_path,
                        asset_dir=asset_dir,
                        output_dir=output_dir,
                        recorded_at="2026-08-15T13:00:00+08:00",
                    )
            self.assertEqual(
                caught.exception.code,
                "ORDERED_PREDECESSOR_OR_OWN_GROUP_UNKNOWN_FAIL_BEFORE_GIT_ASSET_OUTPUT",
            )
            self.assertEqual(calls, {"git": 0, "asset": 0, "output": 0})
            self.assertFalse(asset_dir.exists())
            self.assertFalse(output_dir.exists())

    def test_partial_grouped_binding_is_rejected(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["bindings"]["gse232572_predecessor"]["status"] = subject.UNKNOWN
        with self.assertRaises(subject.ProtocolError) as caught:
            subject._validate_protocol(protocol)
        self.assertEqual(
            caught.exception.code, "GSE232572_PARTIAL_HISTORY_FORBIDDEN"
        )

    def test_clean_normalised_disk_i_and_legal_disk_b_are_both_accepted(self):
        bound = _bound_protocol()
        subject._validate_protocol(bound)
        subject._require_production_bindings(bound)
        implementation_i = subject._normalise_own_binding(bound)
        subject._validate_protocol(implementation_i)
        self.assertEqual(
            {
                implementation_i["bindings"]["implementation"][field]
                for field in subject.OWN_BINDING_FIELDS
            },
            {subject.UNKNOWN},
        )
        self.assertEqual(
            implementation_i["bindings"]["gse232572_predecessor"],
            bound["bindings"]["gse232572_predecessor"],
        )

    def test_single_entry_has_no_loader_or_public_analysis_bypass(self):
        self.assertEqual(
            set(inspect.signature(subject.execute_production).parameters),
            {"protocol_path", "asset_dir", "output_dir", "recorded_at"},
        )
        self.assertFalse(hasattr(subject, "execute_public_analysis"))
        parser_actions = {action.dest for action in subject._parser()._actions}
        self.assertNotIn("mode", parser_actions)
        self.assertNotIn("replay_loader", parser_actions)

    def _repository_audit_fixture(self, *, stale_copy: bool):
        protocol = _bound_protocol()
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_root = Path(temp_dir) / "repo"
            script_path = repo_root / subject.SCRIPT_REPO_PATH
            test_path = repo_root / subject.TEST_REPO_PATH
            protocol_path = repo_root / subject.CONFIG_REPO_PATH
            for path in (script_path, test_path, protocol_path):
                path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_bytes(SCRIPT_PATH.read_bytes())
            test_path.write_bytes(Path(__file__).read_bytes())
            disk_protocol = copy.deepcopy(protocol)
            disk_protocol["repository_authority"]["production_repo_root"] = str(
                repo_root
            )
            protocol_path.write_text(
                json.dumps(disk_protocol, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            implementation_bytes = (
                json.dumps(
                    subject._normalise_own_binding(disk_protocol),
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8")
            head = "8" * 40
            own_i = "7" * 40
            verified = []

            def fake_run_git(_repo_root, *arguments):
                return {
                    ("rev-parse", "HEAD"): head,
                    ("rev-parse", "@{upstream}"): head,
                    ("rev-parse", "--abbrev-ref", "HEAD"): subject.PRODUCTION_BRANCH,
                    ("rev-parse", "--abbrev-ref", "@{upstream}"): subject.PRODUCTION_UPSTREAM,
                    ("status", "--porcelain=v1", "--untracked-files=all"): "",
                }[arguments]

            def fake_verify(_repo_root, **kwargs):
                verified.append(
                    (
                        kwargs["label"],
                        kwargs["commit"],
                        kwargs["expected_parent"],
                        tuple(kwargs["expected_paths"]),
                    )
                )

            def fake_blob(_repo_root, commit, path):
                if commit == own_i and path == subject.CONFIG_REPO_PATH:
                    return implementation_bytes
                if commit == head and path == subject.CONFIG_REPO_PATH:
                    return protocol_path.read_bytes()
                if commit == own_i and path == subject.SCRIPT_REPO_PATH:
                    return script_path.read_bytes()
                if commit == own_i and path == subject.TEST_REPO_PATH:
                    return test_path.read_bytes()
                raise AssertionError(f"unexpected blob request: {commit}:{path}")

            executing = SCRIPT_PATH if stale_copy else script_path
            patches = (
                mock.patch.object(subject, "_run_git", fake_run_git),
                mock.patch.object(subject, "_live_origin_head", return_value=head),
                mock.patch.object(subject, "_verify_frozen_commit", fake_verify),
                mock.patch.object(subject, "_git_blob", fake_blob),
                mock.patch.object(subject, "__file__", str(executing)),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                if stale_copy:
                    with self.assertRaises(subject.ProtocolError) as caught:
                        subject._audit_repository_bindings(
                            disk_protocol, protocol_path, repo_root
                        )
                    self.assertEqual(
                        caught.exception.code, "EXECUTING_SCRIPT_IS_STALE_COPY"
                    )
                    return verified
                result = subject._audit_repository_bindings(
                    disk_protocol, protocol_path, repo_root
                )
            self.assertEqual(result["binding_commit"], head)
            return verified

    def test_legal_disk_i_b_chain_audits_in_exact_order(self):
        verified = self._repository_audit_fixture(stale_copy=False)
        self.assertEqual(
            [item[0] for item in verified],
            [
                "DEC027_AUTHORITY_A",
                "DEC027_RUNTIME_I1",
                "DEC027_RUNTIME_I2",
                "DEC027_RUNTIME_B2",
                "GSE217518_I1",
                "GSE217518_I2",
                "GSE217518_B2",
                "GSE217518_I3",
                "GSE217518_B3",
                "ENCSR854RUF_I1",
                "ENCSR854RUF_I2",
                "ENCSR854RUF_B2",
                "ENCSR854RUF_I3",
                "ENCSR854RUF_B3",
                "ENCSR854RUF_I4",
                "ENCSR854RUF_B4",
                "GSE232572_I1",
                "GSE232572_B1",
                "GSE113849_I",
                "GSE113849_B",
            ],
        )
        self.assertEqual(verified[4][2], subject.RUNTIME_B_COMMIT)
        self.assertEqual(verified[9][2], subject.GSE217_HISTORY_COMMITS[-1])
        self.assertEqual(verified[16][2], subject.ENCSR_B4_COMMIT)
        self.assertEqual(verified[18][2], subject.GSE232_B1_COMMIT)
        self.assertEqual(verified[19][2], "7" * 40)

    def test_stale_copy_is_rejected_before_asset_read(self):
        self.assertEqual(
            len(self._repository_audit_fixture(stale_copy=True)), 20
        )

    def test_fixed_reader_checks_all_asset_identities_before_parse(self):
        subject._validate_protocol(self.protocol)
        parse_count = 0

        def forbidden_parse(*args, **kwargs):
            nonlocal parse_count
            parse_count += 1
            raise AssertionError("parse crossed asset identity barrier")

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            subject, "_validate_author_semantics", forbidden_parse
        ):
            asset_dir = Path(temp_dir) / "missing-assets"
            asset_paths = {
                name: asset_dir
                / self.protocol["ordinary_public_inputs"][name]["filename"]
                for name in subject.EXPECTED_INPUT_NAMES
            }
            with self.assertRaises(subject.AssetError) as caught:
                subject._load_and_replay(
                    protocol=self.protocol, asset_paths=asset_paths
                )
        self.assertEqual(caught.exception.code, "DESIGNED_SNV_TABLE_MISSING")
        self.assertEqual(parse_count, 0)

    def test_context_validation_precedes_loader_and_endpoint_evaluation(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["outcome_blind_context_freeze"]["selected_rule"] = (
            subject.EXPLICIT_UTR3_RULE
        )
        calls = {"git": 0, "asset": 0, "output": 0}

        def poison(name):
            def callback(*args, **kwargs):
                calls[name] += 1
                raise AssertionError(f"{name} must not be called")

            return callback

        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_protocol_path = Path(temp_dir) / "invalid_protocol.json"
            invalid_protocol_path.write_text(
                json.dumps(protocol), encoding="utf-8"
            )
            with mock.patch.object(
                subject, "_audit_repository_bindings", poison("git")
            ), mock.patch.object(
                subject, "_load_and_replay", poison("asset")
            ), mock.patch.object(subject, "_write_report", poison("output")):
                with self.assertRaises(subject.ProtocolError) as caught:
                    subject.execute_production(
                        protocol_path=invalid_protocol_path,
                        asset_dir=Path(temp_dir) / "assets",
                        output_dir=Path(temp_dir) / "out",
                        recorded_at="2026-08-15T13:00:00+08:00",
                    )
            self.assertEqual(calls, {"git": 0, "asset": 0, "output": 0})
            self.assertEqual(caught.exception.code, "SELECTED_CONTEXT_RULE_CHANGED")

        with self.assertRaises(subject.ProtocolError) as direct:
            subject._validate_protocol(protocol)
        self.assertEqual(direct.exception.code, "SELECTED_CONTEXT_RULE_CHANGED")

    def test_gene_plus_source_rule_prevents_cross_gene_merge(self):
        geometry = subject._geometry(self.rows)
        self.assertEqual(geometry["distinct_source_sequence_count"], 1)
        self.assertEqual(geometry["distinct_gene_source_group_count"], 2)
        self.assertEqual(
            geometry["source_sequences_assigned_to_multiple_genes_count"], 1
        )
        self.assertEqual(geometry["dense_gene_source_group_count"], 1)
        self.assertEqual(geometry["rows_in_dense_gene_source_groups"], 3)

    def test_edit_replay_rejects_non_single_substitution_and_position_mismatch(self):
        bad_hamming = copy.deepcopy(self.rows)
        bad_hamming[0]["master_seq"] = "CCAA"
        with self.assertRaises(subject.ReplayInvariantError) as caught:
            subject._geometry(bad_hamming)
        self.assertEqual(caught.exception.code, "PAIR_NOT_EXACT_HAMMING_ONE")

        bad_position = copy.deepcopy(self.rows)
        bad_position[0]["snv_pos"] = "1"
        with self.assertRaises(subject.ReplayInvariantError) as caught:
            subject._geometry(bad_position)
        self.assertEqual(
            caught.exception.code, "SNV_POSITION_DISAGREES_WITH_SEQUENCE"
        )

    def test_endpoint_does_not_read_prediction_and_pvalue_is_not_se(self):
        endpoint = subject._endpoint_aggregates(self.rows)
        self.assertEqual(endpoint["finite_endpoint_count"], 4)
        self.assertEqual(endpoint["valid_pvalue_count"], 4)
        self.assertFalse(endpoint["prediction_field_read_or_used"])
        self.assertEqual(endpoint["replicate_derived_standard_error_field_count"], 0)

    def test_report_is_fail_closed_aggregate_only_and_zero_credit(self):
        report = self.report
        self.assertEqual(
            report["scientific_gate_status_counts"],
            {
                "FAIL": 1,
                "NOT_RUN": 1,
                "PARTIAL_OR_CONDITIONAL": 2,
                "PASS": 5,
                "UNKNOWN_NOT_ASSERTED": 4,
            },
        )
        statuses = {
            gate["gate_id"]: gate["status"] for gate in report["scientific_gates"]
        }
        self.assertEqual(statuses[subject.REQUIRED_GATE_IDS[6]], "FAIL")
        self.assertEqual(statuses[subject.REQUIRED_GATE_IDS[12]], "NOT_RUN")
        self.assertEqual(
            report["randomized_absolute_library"]["status"],
            "EXCLUDED_NOT_TRUE_A2",
        )
        self.assertEqual(
            report["current_contribution"],
            {"ordinary": 0, "a1": 0, "true_a2": 0, "canonical_records": 0},
        )
        self.assertFalse(report["qualified"])
        self.assertFalse(report["state_change"]["training_allowed"])
        self.assertFalse(report["state_change"]["gpu_work_allowed"])
        self.assertFalse(report["state_change"]["model_selection_allowed"])
        self.assertFalse(report["state_change"]["a7_allowed"])
        subject._assert_aggregate_only(report)
        rendered = json.dumps(report, allow_nan=False).lower()
        for forbidden in subject.FORBIDDEN_REPORT_KEYS:
            self.assertNotIn(f'"{forbidden}"', rendered)

    def test_nonfinite_endpoint_and_invalid_pvalue_fail(self):
        invalid_endpoint = copy.deepcopy(self.rows)
        invalid_endpoint[0]["delta_logodds_true"] = "nan"
        with self.assertRaises(subject.ReplayInvariantError) as caught:
            subject._endpoint_aggregates(invalid_endpoint)
        self.assertEqual(caught.exception.code, "ENDPOINT_NOT_FINITE")

        invalid_pvalue = copy.deepcopy(self.rows)
        invalid_pvalue[0]["delta_p_val"] = "1.1"
        with self.assertRaises(subject.ReplayInvariantError) as caught:
            subject._endpoint_aggregates(invalid_pvalue)
        self.assertEqual(
            caught.exception.code, "PVALUE_NOT_FINITE_OR_OUT_OF_RANGE"
        )

    def test_single_aggregate_output_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "result"
            output_path = subject._write_report(output_dir, self.report)
            self.assertEqual(output_path.name, subject.REPORT_FILENAME)
            self.assertEqual(
                [path.name for path in output_dir.iterdir()],
                [subject.REPORT_FILENAME],
            )
            self.assertEqual(subject._write_report(output_dir, self.report), output_path)
            different = copy.deepcopy(self.report)
            different["recorded_at"] = "2026-08-15T13:00:01+08:00"
            with self.assertRaises(subject.OutputError) as caught:
                subject._write_report(output_dir, different)
            self.assertEqual(caught.exception.code, "DIFFERENT_REPORT_ALREADY_EXISTS")


if __name__ == "__main__":
    unittest.main()
