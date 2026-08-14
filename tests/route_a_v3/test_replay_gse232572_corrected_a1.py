from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock


STAGING_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = (
    STAGING_ROOT
    / "scripts"
    / "route_a_v3"
    / "replay_gse232572_corrected_a1.py"
)
PROTOCOL_PATH = (
    STAGING_ROOT
    / "configs"
    / "route_a_v3_gse232572_corrected_a1_replay_v1.json"
)


def _load_subject():
    spec = importlib.util.spec_from_file_location("gse232572_corrected_replay", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("subject module not loadable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


subject = _load_subject()


def _bound_protocol() -> dict:
    protocol = subject._normalise_own_binding(
        json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    )
    own = protocol["bindings"]["implementation"]
    own.update(
        {
            "status": subject.BOUND_BINDING_STATUS,
            "implementation_commit": "5" * 40,
            "implementation_script_sha256": hashlib.sha256(
                SCRIPT_PATH.read_bytes()
            ).hexdigest(),
            "implementation_test_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        }
    )
    return protocol


def _expected_audit_chain(own_i: str, head: str):
    own_blobs = {
        subject.SCRIPT_REPO_PATH: hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest(),
        subject.TEST_REPO_PATH: hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return [
        (
            "DEC027_AUTHORITY_A",
            subject.AUTHORITY_COMMIT,
            subject.AUTHORITY_PARENT,
            subject.AUTHORITY_EXACT12,
            subject.AUTHORITY_BLOBS,
        ),
        (
            "DEC027_RUNTIME_I1",
            subject.RUNTIME_I1_COMMIT,
            subject.AUTHORITY_COMMIT,
            subject.RUNTIME_EXACT3,
            subject.RUNTIME_I1_BLOBS,
        ),
        (
            "DEC027_RUNTIME_I2",
            subject.RUNTIME_I2_COMMIT,
            subject.RUNTIME_I1_COMMIT,
            subject.RUNTIME_EXACT3,
            subject.RUNTIME_I2_BLOBS,
        ),
        (
            "DEC027_RUNTIME_B2",
            subject.RUNTIME_B_COMMIT,
            subject.RUNTIME_I2_COMMIT,
            (subject.RUNTIME_CONFIG_PATH,),
            subject.RUNTIME_B_BLOBS,
        ),
        (
            "GSE217518_I1",
            subject.GSE217_I1_COMMIT,
            subject.RUNTIME_B_COMMIT,
            subject.GSE217_EXACT3,
            subject.GSE217_I1_BLOBS,
        ),
        (
            "GSE217518_I2",
            subject.GSE217_I2_COMMIT,
            subject.GSE217_I1_COMMIT,
            subject.GSE217_EXACT3,
            subject.GSE217_I2_BLOBS,
        ),
        (
            "GSE217518_B2",
            subject.GSE217_B2_COMMIT,
            subject.GSE217_I2_COMMIT,
            (subject.GSE217_CONFIG_PATH,),
            subject.GSE217_B2_BLOBS,
        ),
        (
            "GSE217518_I3",
            subject.GSE217_I3_COMMIT,
            subject.GSE217_B2_COMMIT,
            subject.GSE217_EXACT3,
            subject.GSE217_I3_BLOBS,
        ),
        (
            "GSE217518_B3",
            subject.GSE217_B3_COMMIT,
            subject.GSE217_I3_COMMIT,
            (subject.GSE217_CONFIG_PATH,),
            subject.GSE217_B3_BLOBS,
        ),
        (
            "ENCSR854RUF_I1",
            subject.ENCSR_I1_COMMIT,
            subject.GSE217_B3_COMMIT,
            subject.ENCSR_EXACT3,
            subject.ENCSR_I1_BLOBS,
        ),
        (
            "ENCSR854RUF_I2",
            subject.ENCSR_I2_COMMIT,
            subject.ENCSR_I1_COMMIT,
            subject.ENCSR_EXACT3,
            subject.ENCSR_I2_BLOBS,
        ),
        (
            "ENCSR854RUF_B2",
            subject.ENCSR_B2_COMMIT,
            subject.ENCSR_I2_COMMIT,
            (subject.ENCSR_CONFIG_PATH,),
            subject.ENCSR_B2_BLOBS,
        ),
        (
            "ENCSR854RUF_I3",
            subject.ENCSR_I3_COMMIT,
            subject.ENCSR_B2_COMMIT,
            subject.ENCSR_EXACT3,
            subject.ENCSR_I3_BLOBS,
        ),
        (
            "ENCSR854RUF_B3",
            subject.ENCSR_B3_COMMIT,
            subject.ENCSR_I3_COMMIT,
            (subject.ENCSR_CONFIG_PATH,),
            subject.ENCSR_B3_BLOBS,
        ),
        (
            "ENCSR854RUF_I4",
            subject.ENCSR_I4_COMMIT,
            subject.ENCSR_B3_COMMIT,
            subject.ENCSR_EXACT3,
            subject.ENCSR_I4_BLOBS,
        ),
        (
            "ENCSR854RUF_B4",
            subject.ENCSR_B4_COMMIT,
            subject.ENCSR_I4_COMMIT,
            (subject.ENCSR_CONFIG_PATH,),
            subject.ENCSR_B4_BLOBS,
        ),
        (
            "GSE232572_I",
            own_i,
            subject.ENCSR_B4_COMMIT,
            subject.EXACT3,
            own_blobs,
        ),
        (
            "GSE232572_B",
            head,
            own_i,
            (subject.CONFIG_REPO_PATH,),
            None,
        ),
    ]


def _synthetic_exact_public_geometry() -> dict:
    pair_count = 8068
    ref_insert = "A" * 165
    alt_insert = "C" + ("A" * 164)
    matrices = {
        (1, molecule, replicate): {}
        for molecule in ("DNA", "RNA")
        for replicate in (1, 2, 3)
    }
    pairs = []
    published = {}
    for index in range(pair_count):
        ref_header = f"ref_{index}"
        alt_header = f"alt_{index}"
        key = (index,)
        pair = {
            "published_key": key,
            "ref": {
                "header": ref_header,
                "subpool_number": 1,
                "gene": "GENE",
                "source": f"source_{index}",
                "chr_pos": f"chr1:{index + 1}",
                "strand": "+",
                "orientation": "forward",
                "insert": ref_insert,
            },
            "alt": {
                "header": alt_header,
                "subpool_number": 1,
                "insert": alt_insert,
            },
        }
        pairs.append(pair)
        published[key] = {"published_lnfc": 0.18, "mpranalyze_fdr": 0.5}
        for replicate, alt_rna in ((1, 110.0), (2, 120.0), (3, 130.0)):
            matrices[(1, "DNA", replicate)][ref_header] = 100.0
            matrices[(1, "DNA", replicate)][alt_header] = 100.0
            matrices[(1, "RNA", replicate)][ref_header] = 100.0
            matrices[(1, "RNA", replicate)][alt_header] = alt_rna
    return {
        "published_count": 11929,
        "accepted_pairs": pairs,
        "accepted_before_endpoint_count": pair_count,
        "incomplete_endpoint_count": 0,
        "rejection_counts": {
            "NO_UNIQUE_SEQUENCE_PAIR": 3404,
            "AMBIGUOUS_DISTINCT_SEQUENCE_PAIRS": 457,
        },
        "matrices": matrices,
        "published": published,
    }


class CorrectedReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        cls.replay = _synthetic_exact_public_geometry()
        cls.report = subject._aggregate_public_replay(
            cls.protocol, cls.replay, "2026-08-15T00:00:00+08:00"
        )

    def test_protocol_freezes_exact_eleven_a1_gates_and_dense_gate_is_na(self):
        subject._validate_protocol(self.protocol)
        self.assertEqual(
            tuple(self.protocol["required_scientific_gate_ids_exactly"]),
            subject.REQUIRED_GATE_IDS,
        )
        self.assertNotIn(
            subject.INAPPLICABLE_DENSE_GATE_ID,
            self.protocol["required_scientific_gate_ids_exactly"],
        )
        boundary = self.protocol["a1_role_boundary"]
        self.assertEqual(
            boundary["dense_multi_candidate_true_a2_gate_applicability"],
            "NOT_APPLICABLE_FOR_A1_REPLAY",
        )
        self.assertFalse(boundary["dense_multi_candidate_gate_may_block_a1_replay"])

    def test_production_unknown_bindings_fail_before_asset_loader_and_output(self):
        calls = {"git": 0, "asset": 0, "output": 0}

        def poison(name):
            def callback(*args, **kwargs):
                calls[name] += 1
                raise AssertionError(f"{name} crossed grouped-UNKNOWN barrier")

            return callback

        with tempfile.TemporaryDirectory() as temp_dir:
            asset_dir = Path(temp_dir) / "must_not_be_read"
            output_dir = Path(temp_dir) / "must_not_exist"
            unknown_protocol = subject._normalise_own_binding(self.protocol)
            unknown_protocol_path = Path(temp_dir) / "unknown-protocol.json"
            unknown_protocol_path.write_text(
                json.dumps(unknown_protocol, indent=2) + "\n", encoding="utf-8"
            )
            with mock.patch.object(
                subject, "_audit_repository_bindings", poison("git")
            ), mock.patch.object(
                subject, "_load_public_replay", poison("asset")
            ), mock.patch.object(subject, "_write_report", poison("output")):
                with self.assertRaises(subject.BindingNotReady) as caught:
                    subject.execute_production(
                        protocol_path=unknown_protocol_path,
                        asset_dir=asset_dir,
                        output_dir=output_dir,
                        recorded_at="2026-08-15T00:00:00+08:00",
                    )
            self.assertEqual(
                caught.exception.code,
                "ORDERED_PREDECESSOR_OR_OWN_GROUP_UNKNOWN_FAIL_BEFORE_GIT_ASSET_OUTPUT",
            )
            self.assertEqual(calls, {"git": 0, "asset": 0, "output": 0})
            self.assertFalse(asset_dir.exists())
            self.assertFalse(output_dir.exists())

    def test_grouped_bindings_reject_partial_state(self):
        protocol = subject._normalise_own_binding(self.protocol)
        protocol["bindings"]["implementation"]["implementation_commit"] = "1" * 40
        with self.assertRaises(subject.ProtocolError) as caught:
            subject._validate_protocol(protocol)
        self.assertEqual(caught.exception.code, "OWN_PARTIAL_GROUP_FORBIDDEN")

    def test_frozen_predecessor_full_lifecycle_rejects_any_drift(self):
        protocol = copy.deepcopy(self.protocol)
        protocol["bindings"]["encsr854ruf_predecessor"]["b4_expected_parent"] = (
            subject.ENCSR_B3_COMMIT
        )
        with self.assertRaises(subject.ProtocolError) as caught:
            subject._validate_protocol(protocol)
        self.assertEqual(
            caught.exception.code,
            "ENCSR854RUF_FULL_LIFECYCLE_BINDING_DIFFERS",
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
            {subject.UNKNOWN_BINDING_STATUS},
        )
        self.assertEqual(
            implementation_i["bindings"]["encsr854ruf_predecessor"],
            subject.FROZEN_ENCSR_BINDING,
        )
        self.assertEqual(
            implementation_i["bindings"]["gse217518_predecessor"],
            subject.FROZEN_GSE217_BINDING,
        )

    def test_single_entry_has_no_loader_or_public_analysis_bypass(self):
        parameters = inspect.signature(subject.execute_production).parameters
        self.assertEqual(
            set(parameters), {"protocol_path", "asset_dir", "output_dir", "recorded_at"}
        )
        self.assertFalse(hasattr(subject, "execute_public_analysis"))
        parser_actions = {action.dest for action in subject._parser()._actions}
        self.assertNotIn("mode", parser_actions)
        self.assertNotIn("repo_root", parser_actions)
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
            implementation_protocol = subject._normalise_own_binding(disk_protocol)
            implementation_bytes = (
                json.dumps(implementation_protocol, indent=2, ensure_ascii=False) + "\n"
            ).encode("utf-8")
            head = "6" * 40
            own_i = "5" * 40
            verified = []

            def fake_run_git(_repo_root, *arguments):
                lookup = {
                    ("rev-parse", "HEAD"): head,
                    ("rev-parse", "@{upstream}"): head,
                    ("rev-parse", "--abbrev-ref", "HEAD"): subject.PRODUCTION_BRANCH,
                    ("rev-parse", "--abbrev-ref", "@{upstream}"): subject.PRODUCTION_UPSTREAM,
                    ("status", "--porcelain=v1", "--untracked-files=all"): "",
                }
                return lookup[arguments]

            def fake_verify(_repo_root, **kwargs):
                verified.append(
                    (
                        kwargs["label"],
                        kwargs["commit"],
                        kwargs["expected_parent"],
                        tuple(kwargs["expected_paths"]),
                        kwargs["expected_blobs"],
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
        self.assertEqual(verified, _expected_audit_chain("5" * 40, "6" * 40))

    def test_frozen_commit_audit_checks_parent_paths_and_blob_bytes(self):
        commit = "a" * 40
        parent = "b" * 40
        payload = b"bound payload\n"
        digest = hashlib.sha256(payload).hexdigest()

        def legal_git(_repo_root, *arguments):
            if arguments[0] == "rev-list":
                return f"{commit} {parent}"
            if arguments[0] == "diff-tree":
                return "bound/path.json"
            raise AssertionError(arguments)

        with mock.patch.object(subject, "_run_git", legal_git), mock.patch.object(
            subject, "_git_blob", return_value=payload
        ):
            subject._verify_frozen_commit(
                Path("."),
                label="TEST",
                commit=commit,
                expected_parent=parent,
                expected_paths=("bound/path.json",),
                expected_blobs={"bound/path.json": digest},
            )

        def wrong_parent_git(_repo_root, *arguments):
            if arguments[0] == "rev-list":
                return f"{commit} {'c' * 40}"
            return "bound/path.json"

        with mock.patch.object(subject, "_run_git", wrong_parent_git):
            with self.assertRaises(subject.ProtocolError) as caught:
                subject._verify_frozen_commit(
                    Path("."),
                    label="TEST",
                    commit=commit,
                    expected_parent=parent,
                    expected_paths=("bound/path.json",),
                    expected_blobs={"bound/path.json": digest},
                )
        self.assertEqual(caught.exception.code, "TEST_DIRECT_PARENT_DIFFERS")

    def test_stale_copy_is_rejected_before_asset_read(self):
        verified = self._repository_audit_fixture(stale_copy=True)
        self.assertEqual(len(verified), 18)

    def test_fixed_reader_checks_asset_identity_before_any_parser_import(self):
        subject._validate_protocol(self.protocol)
        parser_import_count = 0

        def fake_upstream(_repo_root, _relative_path, _sha256, label):
            names = {
                "recovery_script": "recover_gse232572_a1.py",
                "generic_helper": "reconstruct_gse232572_sequences.py",
                "recovery_config": "route_a_v3_gse232572_a1_recovery_v1.json",
            }
            return Path(names[label])

        def forbidden_parser_import(*args, **kwargs):
            nonlocal parser_import_count
            parser_import_count += 1
            raise AssertionError("parser imported before public asset identity closed")

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            subject, "_verify_upstream_file", fake_upstream
        ), mock.patch.object(subject, "_load_module", forbidden_parser_import):
            asset_dir = Path(temp_dir) / "missing-assets"
            with self.assertRaises(subject.AssetError) as caught:
                subject._load_public_replay(
                    protocol=self.protocol,
                    repo_root=Path(temp_dir),
                    fasta_paths={
                        1: asset_dir / "GSE232572_C4Sp1.fasta.gz",
                        2: asset_dir / "GSE232572_C4Sp2.fasta.gz",
                        3: asset_dir / "GSE232572_C4Sp3.fasta.gz",
                    },
                    raw_tar=asset_dir / "GSE232572_RAW.tar",
                    published_results=(
                        asset_dir / "41467_2024_46795_MOESM4_ESM.xlsx"
                    ),
                )
        self.assertEqual(caught.exception.code, "FASTA1_MISSING")
        self.assertEqual(parser_import_count, 0)

    def test_mechanical_replay_is_aggregate_only_and_does_not_change_state(self):
        report = self.report
        self.assertEqual(
            report["scientific_gate_status_counts"],
            {"NOT_RUN": 1, "PASS": 7, "UNKNOWN_NOT_ASSERTED": 3},
        )
        geometry = report["aggregate_replay_geometry"]
        self.assertEqual(geometry["published_universe_row_count"], 11929)
        self.assertEqual(geometry["accepted_reference_alternative_pair_count"], 8068)
        self.assertEqual(geometry["exact_hamming_one_pair_count"], 8068)
        self.assertEqual(geometry["biological_replicate_count"], 3)
        self.assertEqual(geometry["replicate_derived_se_defined_pair_count"], 8068)
        self.assertEqual(sum(geometry["replicate_derived_se_histogram"].values()), 8068)
        self.assertEqual(
            report["dense_multi_candidate_true_a2_gate"]["status"],
            "NOT_APPLICABLE_FOR_A1_REPLAY",
        )
        self.assertFalse(
            report["dense_multi_candidate_true_a2_gate"]["blocks_a1_replay"]
        )
        self.assertEqual(
            report["current_contribution"],
            {"ordinary": 0, "a1": 0, "true_a2": 0, "canonical_records": 0},
        )
        self.assertFalse(report["qualified"])
        self.assertFalse(report["state_change"]["training_allowed"])
        self.assertFalse(report["state_change"]["gpu_work_allowed"])
        subject._assert_aggregate_only(report)

        rendered = json.dumps(report, allow_nan=False).lower()
        for forbidden_key in subject.FORBIDDEN_REPORT_KEYS:
            self.assertNotIn(f'"{forbidden_key}"', rendered)

    def test_missing_replicate_fails_before_any_output(self):
        replay = dict(self.replay)
        matrices = dict(self.replay["matrices"])
        damaged = dict(matrices[(1, "RNA", 1)])
        damaged.pop("ref_0")
        matrices[(1, "RNA", 1)] = damaged
        replay["matrices"] = matrices
        with self.assertRaises(subject.ReplayInvariantError) as caught:
            subject._aggregate_public_replay(
                self.protocol, replay, "2026-08-15T00:00:00+08:00"
            )
        self.assertEqual(caught.exception.code, "REQUIRED_REPLICATE_COUNT_MISSING")

    def test_zero_endpoint_is_not_imputed_and_fails_frozen_coverage(self):
        replay = dict(self.replay)
        matrices = dict(self.replay["matrices"])
        damaged = dict(matrices[(1, "RNA", 1)])
        damaged["ref_0"] = 0.0
        matrices[(1, "RNA", 1)] = damaged
        replay["matrices"] = matrices
        with self.assertRaises(subject.ReplayInvariantError) as caught:
            subject._aggregate_public_replay(
                self.protocol, replay, "2026-08-15T00:00:00+08:00"
            )
        self.assertEqual(
            caught.exception.code, "DEFINED_REPLICATE_AUXILIARY_COUNT_MISMATCH"
        )

    def test_wrong_public_universe_count_fails_before_row_replay(self):
        replay = dict(self.replay)
        replay["published_count"] = 11928
        with self.assertRaises(subject.ReplayInvariantError) as caught:
            subject._aggregate_public_replay(
                self.protocol, replay, "2026-08-15T00:00:00+08:00"
            )
        self.assertEqual(caught.exception.code, "PUBLISHED_UNIVERSE_COUNT_MISMATCH")

    def test_remaining_gate_statuses_are_not_overclaimed(self):
        statuses = {
            gate["gate_id"]: gate["status"] for gate in self.report["scientific_gates"]
        }
        self.assertEqual(statuses[subject.REQUIRED_GATE_IDS[6]], "UNKNOWN_NOT_ASSERTED")
        self.assertEqual(statuses[subject.REQUIRED_GATE_IDS[7]], "PASS")
        self.assertEqual(statuses[subject.REQUIRED_GATE_IDS[8]], "UNKNOWN_NOT_ASSERTED")
        self.assertEqual(statuses[subject.REQUIRED_GATE_IDS[9]], "UNKNOWN_NOT_ASSERTED")
        self.assertEqual(statuses[subject.REQUIRED_GATE_IDS[10]], "NOT_RUN")

    def test_single_aggregate_output_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "result"
            output_path = subject._write_report(output_dir, self.report)
            self.assertEqual(output_path.name, subject.REPORT_FILENAME)
            self.assertEqual([path.name for path in output_dir.iterdir()], [subject.REPORT_FILENAME])
            self.assertEqual(subject._write_report(output_dir, self.report), output_path)
            different = copy.deepcopy(self.report)
            different["recorded_at"] = "2026-08-15T00:00:01+08:00"
            with self.assertRaises(subject.OutputError) as caught:
                subject._write_report(output_dir, different)
            self.assertEqual(caught.exception.code, "DIFFERENT_REPORT_ALREADY_EXISTS")
            self.assertEqual(output_path.read_text(encoding="utf-8"), json.dumps(
                self.report,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n")


if __name__ == "__main__":
    unittest.main()
