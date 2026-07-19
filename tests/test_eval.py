"""Offline tests for the independent evaluation suite.

Run:
    /Users/bytedance/Documents/research/editflow/.venv/bin/python \
        -m unittest mrna_editflow.tests.test_eval -v
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mrna_editflow.core.schema import MRNARecord
from mrna_editflow import train_proposal_ranker
from mrna_editflow.data.download_mrna import synthesize_corpus, write_records_jsonl
from mrna_editflow.data.split_contract import (
    build_split_manifest,
    build_split_provenance,
    load_and_verify_split_manifest,
    sha256_file,
)
from mrna_editflow.eval import (
    analyze_mo_fusion_decoded_properties,
    audit_external_sota_real_runs,
    audit_stage_a_health,
    audit_multiobjective_scaleup_claims,
    audit_protein_conditioned_codon_metrics,
    audit_region_adapter_checkpoints,
    audit_region_adapter_results,
    audit_sota_readiness,
    build_data_scaleup_readiness,
    build_downstream_table_manifest,
    build_external_sota_evidence_manifest,
    build_paper_table1_sota_landscape,
    build_paper_table2_t1_t7,
    build_paper_table3_external_baselines,
    build_paper_table4_architecture_ablation,
    build_paper_table5_scale_law_readiness,
    build_codongpt_multiseed_summary,
    build_t4_external_cds_comparison,
    build_t5_external_utr_comparison,
    build_paper_figure1_full_length_edit_flow,
    build_paper_figure2_cascade_recall_precision,
    build_paper_figure3_oracle_gap_closure,
    cascade_error_analysis,
    compare_benchmarks,
    dataset_manifest_audit,
    downstream_data_acquisition_audit,
    family_leakage_protocol,
    metrics,
    merge_multiseed_shards,
    multi_scale_sequence_spectrum_audit,
    mpra_te_predictor,
    oracle,
    run_eval,
    run_multiseed_benchmark,
    artifact_contract,
    stability_predictor,
    stage_a_downstream_eval_readiness,
    summarize_stage_a_scalelaw_sweep,
    summarize_t1_runtime,
    summarize_t2_t3_distribution_novelty,
    summarize_t4_protein_identity_cai_gc,
    summarize_t6_length_curve,
    summarize_region_adapter_comparisons,
    sota_gap_report,
    summarize_proposal_ranking,
)


def _source_records():
    return [
        MRNARecord("r1", "GCCACC", "AUGGCUAAAUAA", "GGAAUAAACU"),
        MRNARecord("r2", "ACCUCC", "AUGCCCGGGUAA", "UGCAAAUAAA"),
    ]


def _candidate_records():
    return [
        # GCU -> GCC is synonymous Ala; protein remains MAK.
        MRNARecord("r1", "GCCACC", "AUGGCCAAAUAA", "GGAAUAAACU"),
        # CCC -> CCU is synonymous Pro; UTR changes exercise budget/length.
        MRNARecord("r2", "CUCUCC", "AUGCCUGGGUAA", "UGCAAAUAAA"),
    ]


class TestScientificArtifactContract(unittest.TestCase):
    def _paper_fixture(
        self,
        tmp: str,
        *,
        role: str = "test",
        suffix: str = "a",
        functional: bool = False,
    ):
        fixture_dir = os.path.join(tmp, f"fixture_{suffix}")
        os.makedirs(fixture_dir, exist_ok=True)
        records = synthesize_corpus(12, seed=41 + len(suffix))
        records_path = os.path.join(fixture_dir, "records.jsonl")
        write_records_jsonl(records, records_path)
        role_indices = {
            "train": list(range(0, 6)),
            "val": list(range(6, 9)),
            "test": list(range(9, 12)),
        }
        role_paths = {}
        for name, indices in role_indices.items():
            path = os.path.join(fixture_dir, f"{name}.idx")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("".join(f"{index}\n" for index in indices))
            role_paths[name] = path
        cluster_path = os.path.join(fixture_dir, "clusters.json")
        with open(cluster_path, "w", encoding="utf-8") as fh:
            json.dump(list(range(12)), fh, separators=(",", ":"))
        leakage_path = os.path.join(fixture_dir, "leakage.json")
        with open(leakage_path, "w", encoding="utf-8") as fh:
            json.dump({
                "split": {"cluster_disjoint": True},
                "summary": {
                    "leakage_exact_match_count": 0,
                    "leakage_flagged_fraction": 0.0,
                    "near_neighbor_threshold_passed": True,
                },
            }, fh)
        manifest = build_split_manifest(
            dataset_id=f"synthetic_{suffix}",
            records_path=records_path,
            role_idx_paths=role_paths,
            leakage_report_path=leakage_path,
            algorithm="deterministic_minhash",
            seed=41,
            family_threshold=0.8,
            family_disjoint=True,
            exact_cross_role_matches=0,
            near_neighbor_threshold_passed=True,
            cluster_assignment_path=cluster_path,
            paper_eligible=True,
        )
        manifest_path = os.path.join(fixture_dir, "split_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)
        contract = load_and_verify_split_manifest(manifest_path)
        oracle_metadata = None
        if functional:
            oracle_artifact = os.path.join(fixture_dir, "oracle.bin")
            with open(oracle_artifact, "wb") as fh:
                fh.write(b"independent-test-oracle")
            oracle_manifest = os.path.join(fixture_dir, "oracle.json")
            with open(oracle_manifest, "w", encoding="utf-8") as fh:
                json.dump({
                    "schema_version": 1,
                    "oracle_type": "independent_test_oracle",
                    "independent": True,
                    "source": "unit_test",
                    "independence_statement": (
                        "Unit-test fixture trained independently of benchmark records."
                    ),
                    "artifact_path": oracle_artifact,
                    "artifact_sha256": sha256_file(oracle_artifact),
                }, fh)
            oracle_metadata = artifact_contract.load_and_verify_oracle_manifest(
                oracle_manifest, run_mode="paper"
            )
        metadata = artifact_contract.build_run_metadata(
            run_mode="paper",
            data_provenance=build_split_provenance(contract, role),
            config={"fixture": suffix},
            code_paths=(artifact_contract.__file__,),
            training_seed=7,
            decoder_seed=11,
            oracle=oracle_metadata,
            functional_claim=functional,
        )
        return metadata, contract, records

    def test_development_artifact_is_rejected_by_paper_builder(self):
        with tempfile.TemporaryDirectory(prefix="paper_gate_dev_") as tmp:
            path = os.path.join(tmp, "benchmark", "dev", "artifact.json")
            os.makedirs(os.path.dirname(path))
            metadata, _contract, _records = self._paper_fixture(tmp)
            metadata.update({
                "run_mode": "development",
                "claim_tier": "development_only",
                "paper_eligible": False,
                "block_reasons": ["development_mode"],
            })
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"scientific_validity": metadata}, fh)
            report = build_paper_table1_sota_landscape.build_paper_table1(
                tmp, run_mode="PAPER", artifact_paths=[path]
            )
            self.assertFalse(report["paper_eligible"])
            self.assertEqual(report["status"], "blocked")
            self.assertEqual(len(report["rejected_artifacts"]), 1)

    def test_valid_synthetic_paper_artifact_exercises_positive_builder_gate(self):
        with tempfile.TemporaryDirectory(prefix="paper_gate_valid_") as tmp:
            path = os.path.join(tmp, "benchmark", "paper", "artifact.json")
            os.makedirs(os.path.dirname(path))
            metadata, _contract, _records = self._paper_fixture(tmp)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"payload": "paper fixture"}, fh)
            artifact_contract.write_provenance_sidecar(path, metadata)
            verified = artifact_contract.verify_paper_artifact(path)
            self.assertTrue(verified["paper_eligible"])
            report = build_paper_table1_sota_landscape.build_paper_table1(
                tmp, run_mode="paper", artifact_paths=[path]
            )
            self.assertTrue(report["paper_eligible"])
            self.assertEqual(len(report["accepted_artifacts"]), 1)
            self.assertEqual(report["rows"], [])

            with self.assertRaises(artifact_contract.ArtifactNamespaceError):
                build_paper_table1_sota_landscape.main([
                    "--project-root", tmp,
                    "--run-mode", "paper",
                    "--paper-artifact", path,
                ])
            report_dir = os.path.join(tmp, "benchmark", "paper", "reports")
            out_json = os.path.join(report_dir, "table1.json")
            out_md = os.path.join(report_dir, "table1.md")
            self.assertEqual(build_paper_table1_sota_landscape.main([
                "--project-root", tmp,
                "--run-mode", "paper",
                "--paper-artifact", path,
                "--out-json", out_json,
                "--out-md", out_md,
            ]), 0)
            self.assertTrue(artifact_contract.verify_paper_artifact(out_json)["paper_eligible"])

            with open(path, "a", encoding="utf-8") as fh:
                fh.write("\n")
            with self.assertRaisesRegex(
                artifact_contract.ArtifactProvenanceError, "content SHA changed"
            ):
                artifact_contract.verify_paper_artifact(path)

    def test_paper_cli_role_and_manifest_fail_closed(self):
        with self.assertRaises(artifact_contract.RunModeError):
            artifact_contract.require_paper_cli_inputs(
                run_mode="paper",
                split_manifest=None,
                split_role="train",
                allowed_roles=("train",),
            )
        with self.assertRaises(Exception):
            artifact_contract.require_paper_cli_inputs(
                run_mode="paper",
                split_manifest="manifest.json",
                split_role="test",
                allowed_roles=("train",),
            )

    def test_heuristic_oracle_and_wrong_namespace_fail(self):
        with tempfile.TemporaryDirectory(prefix="oracle_contract_") as tmp:
            artifact = os.path.join(tmp, "oracle.bin")
            with open(artifact, "wb") as fh:
                fh.write(b"heuristic")
            manifest = os.path.join(tmp, "oracle.json")
            with open(manifest, "w", encoding="utf-8") as fh:
                json.dump({
                    "schema_version": 1,
                    "oracle_type": "heuristic_development_oracle",
                    "independent": False,
                    "source": "unit_test",
                    "artifact_path": artifact,
                    "artifact_sha256": hashlib.sha256(b"heuristic").hexdigest(),
                }, fh)
            with self.assertRaises(artifact_contract.OracleContractError):
                artifact_contract.load_and_verify_oracle_manifest(manifest, run_mode="paper")
            with self.assertRaises(artifact_contract.ArtifactNamespaceError):
                artifact_contract.validate_output_namespace(
                    os.path.join(tmp, "benchmark", "dev", "out"), "paper"
                )

    def test_checkpoint_test_provenance_mismatch_fails(self):
        import torch

        with tempfile.TemporaryDirectory(prefix="checkpoint_contract_") as tmp:
            checkpoint = os.path.join(tmp, "benchmark", "paper", "checkpoint.pt")
            os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
            metadata, contract, _records = self._paper_fixture(tmp, role="train")
            torch.save({
                "config": {"fixture": "a"},
                "scientific_validity": metadata,
            }, checkpoint)
            expected = build_split_provenance(contract, "test")
            expected["records_sha256"] = "f" * 64
            with self.assertRaises(artifact_contract.ArtifactProvenanceError):
                artifact_contract.verify_paper_checkpoint(checkpoint, expected)

    def test_programmatic_paper_eval_cannot_bypass_verified_contract(self):
        with tempfile.TemporaryDirectory(prefix="paper_api_bypass_") as tmp:
            out_dir = os.path.join(tmp, "benchmark", "paper", "eval")
            with self.assertRaisesRegex(ValueError, "VerifiedSplitContract"):
                run_eval.run_evaluation(
                    _candidate_records(),
                    sources=_source_records(),
                    out_dir=out_dir,
                    run_mode="paper",
                    verified_data_provenance={"split_role": "test"},
                )

    def test_paper_ranker_rejects_mismatched_teacher_before_model_load(self):
        import torch

        with tempfile.TemporaryDirectory(prefix="paper_ranker_mismatch_") as tmp:
            checkpoint_metadata, contract, records = self._paper_fixture(
                tmp, role="train", suffix="checkpoint"
            )
            checkpoint = os.path.join(tmp, "benchmark", "paper", "base.pt")
            os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
            torch.save({
                "config": {"fixture": "checkpoint"},
                "scientific_validity": checkpoint_metadata,
            }, checkpoint)

            teacher_metadata, _other_contract, _other_records = self._paper_fixture(
                tmp, role="train", suffix="teacher_other", functional=True
            )
            teacher = os.path.join(tmp, "benchmark", "paper", "teacher.jsonl")
            summary = os.path.join(tmp, "benchmark", "paper", "teacher_summary.json")
            with open(teacher, "w", encoding="utf-8") as fh:
                fh.write("{}\n")
            with open(summary, "w", encoding="utf-8") as fh:
                json.dump({"artifact_kind": "teacher_summary"}, fh)
            artifact_contract.write_provenance_sidecar(teacher, teacher_metadata)
            artifact_contract.write_provenance_sidecar(summary, teacher_metadata)

            with self.assertRaisesRegex(
                artifact_contract.ArtifactProvenanceError, "provenance mismatch"
            ):
                train_proposal_ranker.train_proposal_ranker(
                    records=records,
                    teacher_jsonl=teacher,
                    teacher_summary=summary,
                    base_checkpoint=checkpoint,
                    save_dir=os.path.join(tmp, "benchmark", "paper", "ranker"),
                    profile_path=os.path.join(
                        tmp, "benchmark", "paper", "ranker", "profile.jsonl"
                    ),
                    steps=1,
                    run_mode="paper",
                    split_contract=contract,
                    split_role="train",
                )

    def test_positive_paper_evaluation_requires_all_verified_inputs(self):
        import torch

        class IndependentOracleAdapter:
            def __init__(self):
                self._delegate = oracle.LocalTranslationOracle()

            def batch_score(self, records):
                return self._delegate.batch_score(records)

            def cross_validate_predictors(self, records):
                return self._delegate.cross_validate_predictors(records)

        with tempfile.TemporaryDirectory(prefix="paper_eval_positive_") as tmp:
            checkpoint_metadata, contract, records = self._paper_fixture(
                tmp, role="train", suffix="paper_eval", functional=True
            )
            checkpoint = os.path.join(tmp, "benchmark", "paper", "checkpoint.pt")
            os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
            torch.save({
                "config": {"fixture": "paper_eval"},
                "scientific_validity": checkpoint_metadata,
            }, checkpoint)
            candidates = [records[index] for index in contract.roles["test"].indices]
            out_dir = os.path.join(tmp, "benchmark", "paper", "evaluation")
            result = run_eval.run_evaluation(
                candidates,
                sources=records,
                out_dir=out_dir,
                n_bootstrap=10,
                run_mode="paper",
                split_contract=contract,
                split_role="test",
                checkpoint_path=checkpoint,
                oracle_manifest=checkpoint_metadata["oracle"]["manifest_path"],
                oracle=IndependentOracleAdapter(),
                training_seed=7,
                decoder_seed=11,
            )
            self.assertTrue(result["scientific_validity"]["paper_eligible"])
            verified = artifact_contract.verify_paper_artifact(result["json_path"])
            self.assertTrue(verified["paper_eligible"])
            self.assertTrue(verified["oracle"]["paper_permitted"])


class TestStageAHealthAudit(unittest.TestCase):
    def _write_profile(self, path: str, *, unhealthy: bool) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for step in range(1, 21):
                row = {
                    "step": step,
                    "loss": 10.0 - step * 0.1,
                    "finite_loss": not unhealthy,
                    "finite_grad": not unhealthy,
                    "grad_norm": 2.0 if not unhealthy else 1000.0,
                    "amp_enabled": True,
                    "amp_fallback_used": unhealthy,
                    "retries": 0 if not unhealthy else 4,
                    "oom_reductions": 0,
                    "samples_per_s": 2.0,
                    "batch_size": 1,
                }
                fh.write(json.dumps(row) + "\n")

    def test_healthy_and_unhealthy_profiles_are_advisory_and_read_only(self):
        with tempfile.TemporaryDirectory(prefix="stage_a_health_") as tmp:
            config = os.path.join(tmp, "config.json")
            with open(config, "w", encoding="utf-8") as fh:
                json.dump({
                    "model": {"use_aux_struct": False, "aux_loss_weight": 0.0},
                    "scientific_validity": {
                        "records_role_restricted": True,
                        "records_pretruncated": False,
                    },
                }, fh)
            healthy = os.path.join(tmp, "healthy.profile.jsonl")
            unhealthy = os.path.join(tmp, "unhealthy.profile.jsonl")
            self._write_profile(healthy, unhealthy=False)
            self._write_profile(unhealthy, unhealthy=True)
            with open(healthy, "rb") as fh:
                before_sha = hashlib.sha256(fh.read()).hexdigest()
            before = (os.stat(healthy).st_size, os.stat(healthy).st_mtime_ns, before_sha)
            healthy_report = audit_stage_a_health.audit_profile(
                healthy,
                config_path=config,
                held_out_curve_present=True,
                split_provenance_present=True,
                target_steps=20,
            )
            with open(healthy, "rb") as fh:
                after_sha = hashlib.sha256(fh.read()).hexdigest()
            after = (os.stat(healthy).st_size, os.stat(healthy).st_mtime_ns, after_sha)
            self.assertEqual(before, after)
            self.assertEqual(healthy_report["verdict"], "healthy_to_continue")
            unhealthy_report = audit_stage_a_health.audit_profile(
                unhealthy,
                config_path=config,
                held_out_curve_present=True,
                split_provenance_present=True,
            )
            self.assertEqual(unhealthy_report["verdict"], "stop_recommended")
            self.assertTrue(unhealthy_report["advisory_only"])
            self.assertFalse(unhealthy_report["biological_progress_inferred_from_training_loss"])


class TestOracle(unittest.TestCase):
    def test_oracle_is_deterministic_and_training_decoupled(self):
        pred = oracle.LocalTranslationOracle()
        a = pred.score_utr("GCCACC", "AUGGCC")
        b = pred.score_utr("GCCACC", "AUGGCC")
        self.assertEqual(a, b)
        for key in ("mrl", "te", "predictor2_mrl", "predictor2_te", "agreement"):
            self.assertTrue(math.isfinite(float(a[key])))
        self.assertGreaterEqual(a["te"], 0.0)
        self.assertLessEqual(a["te"], 1.0)

        with open(oracle.__file__, "r", encoding="utf-8") as fh:
            src = fh.read()
        self.assertNotIn("MRNAEditFormer", src)
        self.assertNotIn("mrna_editflow.models", src)

    def test_oracle_record_batch_and_cross_validation(self):
        pred = oracle.LocalTranslationOracle()
        scores = pred.batch_score(_candidate_records())
        self.assertEqual(len(scores), 2)
        self.assertEqual(scores[0]["transcript_id"], "r1")
        cv = pred.cross_validate_predictors(_candidate_records())
        self.assertEqual(cv["n"], 2)
        self.assertTrue(math.isfinite(cv["te_mae_between_predictors"]))
        self.assertGreaterEqual(cv["agreement_mean"], 0.0)


class TestRegionAdapterCheckpointAudit(unittest.TestCase):
    def test_audit_region_adapter_checkpoint_artifacts(self):
        import torch

        with tempfile.TemporaryDirectory(prefix="mef_region_audit_") as tmp:
            ckpt_dir = os.path.join(tmp, "ckpts", "region_adapter_t5_utr5_head256")
            log_dir = os.path.join(tmp, "logs")
            os.makedirs(ckpt_dir)
            os.makedirs(log_dir)
            ckpt_path = os.path.join(ckpt_dir, "stage_b_region_t5_best.pt")
            torch.save(
                {
                    "stage": "B_region",
                    "task_id": "T5",
                    "step": 7,
                    "best_loss": 1.25,
                    "region_ids": [0],
                    "adapter_bottleneck": 4,
                    "trainable_names": [
                        "adapters.utr5.down.weight",
                        "adapters.utr5.up.bias",
                    ],
                    "changed_frozen_names": [],
                },
                ckpt_path,
            )
            profile_path = os.path.join(log_dir, "region_adapter_t5_utr5_head256.profile.jsonl")
            with open(profile_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"step": 1, "stage": "B_region"}) + "\n")
                fh.write(
                    json.dumps(
                        {
                            "step": 10,
                            "stage": "B_region",
                            "task_id": "T5",
                            "region_ids": [0],
                            "finite_loss": True,
                            "finite_grad": True,
                            "loss": 2.0,
                            "grad_norm": 0.5,
                        }
                    )
                    + "\n"
                )
            out_json = os.path.join(tmp, "audit.json")
            out_md = os.path.join(tmp, "audit.md")
            payload = audit_region_adapter_checkpoints.run_audit(
                project_root=tmp,
                modes=["utr5"],
                expected_profile_step=10,
                out_json=out_json,
                out_md=out_md,
            )
            self.assertTrue(payload["summary"]["all_ok"])
            row = payload["rows"][0]
            self.assertEqual(row["stage"], "B_region")
            self.assertEqual(row["profile_step"], 10)
            self.assertTrue(row["all_trainable_adapters"])
            self.assertTrue(row["profile_reached_expected_step"])
            self.assertIn("checkpoint_sha256", row)
            with open(out_json, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["artifact_kind"], "region_adapter_checkpoint_audit")
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Region Adapter Checkpoint Audit", text)
            self.assertIn("utr5", text)


class TestRegionAdapterDecisionReport(unittest.TestCase):
    def test_summarizes_region_adapter_compare_files(self):
        def write_compare(path: str, baseline: str, run_values: dict[str, float]) -> None:
            rows = []
            for run, te in run_values.items():
                for metric, value in {
                    "delta_oracle_te_vs_source": te,
                    "legal_fraction": 1.0,
                    "mean_protein_identity": 1.0,
                    "within_budget_fraction": 1.0,
                    "reading_frame_intact_fraction": 1.0,
                }.items():
                    rows.append(
                        {
                            "run": run,
                            "metric": metric,
                            "baseline_mean": 0.01 if metric == "delta_oracle_te_vs_source" else 1.0,
                            "run_mean": value,
                            "delta": value - (0.01 if metric == "delta_oracle_te_vs_source" else 1.0),
                            "improvement": value - (0.01 if metric == "delta_oracle_te_vs_source" else 1.0),
                            "paired_p": 0.01,
                            "ci_low": 0.001,
                            "ci_high": 0.003,
                            "n_paired_seeds": 10,
                            "higher_is_better": True,
                        }
                    )
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"baseline": {"label": baseline}, "rows": rows}, fh)

        with tempfile.TemporaryDirectory(prefix="mef_region_decision_") as tmp:
            bench = os.path.join(tmp, "benchmark")
            os.makedirs(bench)
            write_compare(
                os.path.join(bench, "region_adapter_vs_hardneg_v2_top64_head256.json"),
                "hardneg_v2_top64",
                {
                    "region_adapter_utr5_top64": 0.02,
                    "region_adapter_all_top64": 0.04,
                },
            )
            write_compare(
                os.path.join(bench, "region_adapter_vs_mo_grpo_top64_head256.json"),
                "mo_grpo_top64",
                {
                    "region_adapter_utr5_top64": 0.015,
                    "region_adapter_all_top64": 0.035,
                },
            )
            out_json = os.path.join(tmp, "decision.json")
            out_md = os.path.join(tmp, "decision.md")
            payload = summarize_region_adapter_comparisons.summarize_region_adapter_comparisons(
                project_root=tmp,
                out_json=out_json,
                out_md=out_md,
            )
            self.assertEqual(payload["artifact_kind"], "region_adapter_decision_report")
            self.assertEqual(payload["summary"]["n_compare_files_found"], 2)
            self.assertEqual(payload["summary"]["n_compare_files_expected"], 5)
            self.assertEqual(len(payload["summary"]["missing_compare_files"]), 3)
            self.assertTrue(payload["summary"]["all_constraints_ok"])
            self.assertTrue(payload["summary"]["all_constraints_exact_1"])
            self.assertEqual(payload["summary"]["best_run_vs_hardneg"], "region_adapter_all_top64")
            self.assertEqual(
                payload["summary"]["best_run_by_baseline"]["hardneg_v2_top64"],
                "region_adapter_all_top64",
            )
            self.assertEqual(
                payload["summary"]["best_strict_positive_run_by_baseline"]["hardneg_v2_top64"],
                "region_adapter_all_top64",
            )
            self.assertEqual(
                payload["summary"]["best_run_by_baseline"]["mo_grpo_top64"],
                "region_adapter_all_top64",
            )
            all_row = {
                row["run"]: row for row in payload["runs"]
            }["region_adapter_all_top64"]
            self.assertEqual(all_row["constraints"]["legal_fraction"], 1.0)
            self.assertTrue(all_row["constraints_exact_1"])
            self.assertAlmostEqual(all_row["primary_vs_hardneg_delta"], 0.03)
            self.assertIn("mo_grpo_top64", all_row["primary_by_baseline"])
            self.assertAlmostEqual(
                all_row["primary_by_baseline"]["mo_grpo_top64"]["delta"],
                0.025,
            )
            self.assertEqual(
                all_row["primary_by_baseline"]["mo_grpo_top64"]["signal"],
                "strict_positive",
            )
            with open(out_json, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["summary"]["n_region_adapter_runs"], 2)
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Region Adapter Decision Report", text)
            self.assertIn("region_adapter_all_top64", text)
            self.assertIn("vs mo_grpo_top64", text)
            self.assertIn("strict_positive", text)
            self.assertIn("All constraints exactly 1: True", text)


class TestRegionAdapterResultAudit(unittest.TestCase):
    def _write_compare(self, path: str, baseline: str, run_values: dict[str, float]) -> None:
        rows = []
        for run, te in run_values.items():
            for metric, value in {
                "delta_oracle_te_vs_source": te,
                "legal_fraction": 1.0,
                "mean_protein_identity": 1.0,
                "within_budget_fraction": 1.0,
                "reading_frame_intact_fraction": 1.0,
            }.items():
                baseline_mean = 0.01 if metric == "delta_oracle_te_vs_source" else 1.0
                rows.append(
                    {
                        "run": run,
                        "metric": metric,
                        "baseline_mean": baseline_mean,
                        "run_mean": value,
                        "delta": value - baseline_mean,
                        "improvement": value - baseline_mean,
                        "paired_p": 0.01,
                        "ci_low": 0.001,
                        "ci_high": 0.003,
                        "n_paired_seeds": 10,
                        "higher_is_better": True,
                    }
                )
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"baseline": {"label": baseline}, "rows": rows}, fh)

    def test_audits_complete_region_adapter_result_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="mef_region_result_audit_") as tmp:
            bench = os.path.join(tmp, "benchmark")
            os.makedirs(bench)
            runs = {
                "region_adapter_utr5_top64": 0.020,
                "region_adapter_cds_top64": 0.025,
                "region_adapter_utr3_top64": 0.030,
                "region_adapter_all_top64": 0.040,
            }
            for label in (
                "hardneg_v2_top64",
                "mo_grpo_top64",
                "mo_scalar_top64",
                "mo_pareto_top64",
                "mo_te_only_top64",
            ):
                self._write_compare(
                    os.path.join(bench, f"region_adapter_vs_{label}_head256.json"),
                    label,
                    runs,
                )
            summarize_region_adapter_comparisons.summarize_region_adapter_comparisons(
                project_root=tmp,
                out_json=os.path.join(bench, "region_adapter_decision_report_head256.json"),
            )
            out_json = os.path.join(tmp, "audit.json")
            out_md = os.path.join(tmp, "audit.md")
            payload = audit_region_adapter_results.audit_region_adapter_results(
                project_root=tmp,
                out_json=out_json,
                out_md=out_md,
            )
            self.assertEqual(payload["artifact_kind"], "region_adapter_result_audit")
            self.assertTrue(payload["summary"]["ready_for_sota_claim_audit"])
            self.assertTrue(payload["summary"]["all_expected_compare_files_exist"])
            self.assertTrue(payload["summary"]["all_constraints_exact_1"])
            self.assertTrue(payload["summary"]["all_primary_stats_finite"])
            all_row = {
                row["run"]: row for row in payload["run_audits"]
            }["region_adapter_all_top64"]
            self.assertTrue(all_row["constraints_exact_1"])
            self.assertEqual(
                all_row["primary_by_baseline"][0]["signal"],
                "strict_positive",
            )
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Ready for SOTA claim audit: True", text)
            self.assertIn("All constraints exactly 1: True", text)

    def test_pending_region_adapter_result_audit_lists_missing_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="mef_region_result_pending_") as tmp:
            payload = audit_region_adapter_results.audit_region_adapter_results(project_root=tmp)
            self.assertFalse(payload["summary"]["ready_for_sota_claim_audit"])
            self.assertFalse(payload["summary"]["decision_report_exists"])
            self.assertEqual(payload["summary"]["n_compare_files_found"], 0)
            self.assertEqual(len(payload["summary"]["missing_artifacts"]), 6)


class TestSotaReadinessAudit(unittest.TestCase):
    def _write_region_compare(self, path: str, baseline: str) -> None:
        rows = []
        run_values = {
            "region_adapter_utr5_top64": 0.020,
            "region_adapter_cds_top64": 0.025,
            "region_adapter_utr3_top64": 0.030,
            "region_adapter_all_top64": 0.040,
        }
        for run, te in run_values.items():
            for metric, value in {
                "delta_oracle_te_vs_source": te,
                "legal_fraction": 1.0,
                "mean_protein_identity": 1.0,
                "within_budget_fraction": 1.0,
                "reading_frame_intact_fraction": 1.0,
            }.items():
                baseline_mean = 0.01 if metric == "delta_oracle_te_vs_source" else 1.0
                rows.append(
                    {
                        "run": run,
                        "metric": metric,
                        "baseline_mean": baseline_mean,
                        "run_mean": value,
                        "delta": value - baseline_mean,
                        "improvement": value - baseline_mean,
                        "paired_p": 0.01,
                        "ci_low": 0.001,
                        "ci_high": 0.003,
                        "n_paired_seeds": 10,
                        "higher_is_better": True,
                    }
                )
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"baseline": {"label": baseline}, "rows": rows}, fh)

    def _write_mo_compare(
        self,
        bench: str,
        filename: str,
        run_rows: list[tuple[str, float, float, float]],
        *,
        n_records: int,
    ) -> None:
        rows = []
        for run, run_mean, baseline_mean, paired_p in run_rows:
            rows.append(
                {
                    "run": run,
                    "metric": "delta_oracle_te_vs_source",
                    "baseline_mean": baseline_mean,
                    "run_mean": run_mean,
                    "delta": run_mean - baseline_mean,
                    "paired_p": paired_p,
                    "ci_low": 0.001,
                    "ci_high": 0.003,
                    "n_paired_seeds": 10,
                }
            )
            for metric in (
                "mean_protein_identity",
                "within_budget_fraction",
                "reading_frame_intact_fraction",
            ):
                rows.append(
                    {
                        "run": run,
                        "metric": metric,
                        "baseline_mean": 1.0,
                        "run_mean": 1.0,
                        "delta": 0.0,
                        "paired_p": 1.0,
                        "ci_low": 0.0,
                        "ci_high": 0.0,
                        "n_paired_seeds": 10,
                    }
                )
        payload = {
            "baseline": {"config": {"n_records": n_records}},
            "config_checks": [{"field": "n_records", "matches": True}],
            "rows": rows,
        }
        with open(os.path.join(bench, filename), "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    def _write_mo_summary(self, bench: str, slice_name: str, mode: str) -> None:
        if mode == "hardneg_v2":
            dirname = f"multiseed_t5_public_{slice_name}_hardneg_v2_top64"
        else:
            dirname = f"multiseed_t5_public_{slice_name}_mo_{mode}_top64"
        out_dir = os.path.join(bench, dirname)
        os.makedirs(out_dir, exist_ok=True)
        aggregate = {
            metric: {"mean": 1.0, "n": 10}
            for metric in (
                "legal_fraction",
                "mean_protein_identity",
                "within_budget_fraction",
                "reading_frame_intact_fraction",
            )
        }
        with open(os.path.join(out_dir, "multiseed_summary.json"), "w", encoding="utf-8") as fh:
            json.dump({"aggregate": aggregate}, fh)

    def _write_mo_claim_artifacts(self, tmp: str, *, include_summaries: bool = True) -> None:
        bench = os.path.join(tmp, "benchmark")
        os.makedirs(bench, exist_ok=True)
        self._write_mo_compare(
            bench,
            "compare_mo_fusion_vs_te_only_head256.json",
            [
                ("mo_scalar_top64", 0.01087, 0.00348, 0.0045),
                ("mo_pareto_top64", 0.01029, 0.00348, 0.0045),
                ("mo_grpo_top64", 0.01114, 0.00348, 0.0045),
            ],
            n_records=256,
        )
        self._write_mo_compare(
            bench,
            "compare_scalar_vs_hardneg_v2_head256.json",
            [("mo_scalar_top64", 0.01087, 0.00503, 0.0045)],
            n_records=256,
        )
        self._write_mo_compare(
            bench,
            "compare_pareto_vs_hardneg_v2_head256.json",
            [("mo_pareto", 0.01029, 0.00503, 0.0045)],
            n_records=256,
        )
        self._write_mo_compare(
            bench,
            "compare_grpo_vs_hardneg_v2_head256.json",
            [("mo_grpo", 0.01114, 0.00503, 0.0045)],
            n_records=256,
        )
        self._write_mo_compare(
            bench,
            "compare_grpo_vs_scalar_head256.json",
            [("mo_grpo", 0.01114, 0.01087, 0.4348)],
            n_records=256,
        )
        self._write_mo_compare(
            bench,
            "compare_scalar_vs_pareto_head256.json",
            [("mo_scalar", 0.01087, 0.01029, 0.2139)],
            n_records=256,
        )
        self._write_mo_compare(
            bench,
            "compare_mo_fusion_vs_te_only_head1024.json",
            [
                ("mo_scalar_top64", 0.00855, 0.00846, 0.7446),
                ("mo_pareto_top64", 0.00927, 0.00846, 0.05047),
                ("mo_grpo_top64", 0.00852, 0.00846, 0.8981),
            ],
            n_records=1024,
        )
        self._write_mo_compare(
            bench,
            "compare_mo_fusion_vs_hardneg_v2_head1024.json",
            [
                ("mo_scalar_top64", 0.00855, 0.00385, 0.0045),
                ("mo_pareto_top64", 0.00927, 0.00385, 0.0045),
                ("mo_grpo_top64", 0.00852, 0.00385, 0.0045),
            ],
            n_records=1024,
        )
        if include_summaries:
            for slice_name in ("head256", "head1024"):
                for mode in ("te_only", "scalar", "pareto", "grpo", "hardneg_v2"):
                    self._write_mo_summary(bench, slice_name, mode)

    def test_multiobjective_scaleup_claim_audit_classifies_borderline(self):
        with tempfile.TemporaryDirectory(prefix="mef_mo_claim_audit_") as tmp:
            self._write_mo_claim_artifacts(tmp)
            out_json = os.path.join(tmp, "claim.json")
            out_md = os.path.join(tmp, "claim.md")
            payload = audit_multiobjective_scaleup_claims.audit_multiobjective_scaleup_claims(
                project_root=tmp,
                out_json=out_json,
                out_md=out_md,
            )
            self.assertTrue(payload["summary"]["ready_for_full_hard_constraint_claim_audit"])
            self.assertTrue(payload["summary"]["head256_fusion_vs_te_only_all_strict"])
            self.assertFalse(payload["summary"]["head1024_vs_te_only_strict_claim_allowed"])
            self.assertEqual(payload["summary"]["head1024_vs_te_only_best_signal"], "borderline_positive")
            pareto = next(
                row
                for row in payload["comparisons"]
                if row["comparison_id"] == "head1024_mo_pareto_top64_vs_te_only"
            )
            self.assertEqual(pareto["signal"], "borderline_positive")
            self.assertEqual(
                pareto["claim_language"],
                "trend_or_borderline_only_no_strict_significance",
            )
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("head1024 vs te_only strict claim allowed: False", text)

    def _write_gc_sweep(self, bench: str) -> None:
        summary_json = os.path.join(bench, "protein_conditioned_cds_gc_sweep_head256.summary.json")
        jsonl_path = os.path.join(bench, "protein_conditioned_cds_gc_sweep_head256.jsonl")
        md_path = os.path.join(bench, "protein_conditioned_cds_gc_sweep_head256.md")
        points = []
        for rank, gc_weight in enumerate([0.0, 1.0]):
            point = {
                "gc_weight": gc_weight,
                "target_gc": 0.55,
                "summary": {
                    "n": 2,
                    "mean_designed_cai": 1.0 - 0.01 * rank,
                    "mean_designed_gc": 0.70 - 0.03 * rank,
                    "mean_abs_gc_error": 0.15 - 0.02 * rank,
                    "mean_codon_changes": 4.0,
                    "protein_identity_eq_1_fraction": 1.0,
                },
                "pareto_rank": 0,
                "is_pareto_front": True,
            }
            points.append(point)
        payload = {
            "sweep_kind": "protein_conditioned_cai_gc_pareto",
            "n_targets": 2,
            "target_gc": 0.55,
            "points": points,
            "pareto_front": points,
            "pareto_front_gc_weights": [0.0, 1.0],
            "out_jsonl": jsonl_path,
            "out_md": md_path,
            "artifact_contract": {
                "hard_constraint": "protein_identity_eq_1_fraction must remain 1.0",
            },
        }
        with open(summary_json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for gc_weight in [0.0, 1.0]:
                for idx in range(2):
                    fh.write(
                        json.dumps(
                            {
                                "transcript_id": f"r{idx}",
                                "gc_weight": gc_weight,
                                "protein_identity": 1.0,
                            }
                        )
                        + "\n"
                    )
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write("# sweep\n")

    def _write_external_sota_dry_run(self, tmp: str) -> None:
        out_dir = os.path.join(tmp, "benchmark", "external_sota", "dry_run_t5_head1024")
        os.makedirs(out_dir, exist_ok=True)
        dataset_sha = "a" * 64
        summary = {
            "status": "dry_run_complete",
            "task_id": "T5",
            "n_models": 2,
            "n_executable_ready": 0,
            "dataset": {
                "exists": True,
                "sha256": dataset_sha,
                "record_count_effective": 1024,
                "split_name": "public_head1024",
                "seed": 0,
            },
            "rows": [
                {
                    "model_name": "LinearDesign",
                    "status": "not_configured",
                    "protocol_difference": "CDS-only optimizer; not full-transcript editing.",
                    "metrics": {},
                },
                {
                    "model_name": "codonGPT",
                    "status": "not_configured",
                    "protocol_difference": "CDS synonymous generation; not UTR editing.",
                    "metrics": {},
                },
            ],
            "artifact_contract": {
                "required_real_run_metadata": [
                    "dataset.sha256",
                    "dataset.split_name",
                    "dataset.seed",
                    "dataset.record_count_effective",
                    "runtime.elapsed_s",
                    "hardware",
                ],
                "real_metric_policy": "Do not report external metrics from dry-run.",
            },
        }
        runtime = {
            "dataset_sha256": dataset_sha,
            "elapsed_s": 0.01,
            "hardware": {"label": "unit-cpu"},
        }
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh)
        with open(os.path.join(out_dir, "runtime.json"), "w", encoding="utf-8") as fh:
            json.dump(runtime, fh)
        with open(os.path.join(out_dir, "table.md"), "w", encoding="utf-8") as fh:
            fh.write("# External SOTA Dry-Run\n")

    def _write_external_sota_input_pack(self, tmp: str) -> None:
        out_dir = os.path.join(tmp, "benchmark", "external_sota", "input_pack_t5_head1024")
        os.makedirs(out_dir, exist_ok=True)
        outputs = {
            "cds_protein_jsonl": os.path.join(out_dir, "cds_protein_inputs.jsonl"),
            "utr5_jsonl": os.path.join(out_dir, "utr5_inputs.jsonl"),
            "metric_schema_json": os.path.join(out_dir, "metric_schema.json"),
            "cds_protein_jsonl_sha256": "c" * 64,
            "utr5_jsonl_sha256": "d" * 64,
            "metric_schema_json_sha256": "e" * 64,
        }
        summary = {
            "artifact_kind": "external_sota_input_pack",
            "claim_policy": "Input pack only; no external SOTA claim.",
            "ready_for_external_real_run": True,
            "ready_for_external_sota_claim": False,
            "models": {
                "cds_protein_conditioned": ["LinearDesign", "codonGPT"],
                "utr5_only": [],
            },
            "n_cds_protein_rows": 1024,
            "n_utr5_rows": 1024,
            "n_skipped_invalid_cds": 0,
            "outputs": outputs,
        }
        with open(outputs["cds_protein_jsonl"], "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"transcript_id": "r1"}) + "\n")
        with open(outputs["utr5_jsonl"], "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"transcript_id": "r1"}) + "\n")
        with open(outputs["metric_schema_json"], "w", encoding="utf-8") as fh:
            json.dump({"artifact_kind": "external_sota_metric_schema"}, fh)
        with open(os.path.join(out_dir, "table.md"), "w", encoding="utf-8") as fh:
            fh.write("# External SOTA Input Pack\n")
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh)

    def _write_external_sota_real_run_audit(self, tmp: str) -> None:
        docs = os.path.join(tmp, "docs")
        os.makedirs(docs, exist_ok=True)
        audit = {
            "artifact_kind": "external_sota_real_run_audit",
            "summary": {
                "audit_complete": True,
                "ready_for_external_real_metric_table": False,
                "ready_for_external_sota_metric_claim": False,
                "ready_for_external_sota_claim": False,
                "n_models_expected": 2,
                "n_models_measured": 0,
                "n_models_invalid": 0,
                "n_models_missing": 2,
            },
            "rows": [
                {
                    "model_name": model_name,
                    "status": "missing",
                    "task_family": task_family,
                    "expected_input_rows": 1024,
                    "n_outputs": 0,
                    "success_fraction": 0.0,
                    "hard_constraints_exact_1": False,
                    "real_metric_ready": False,
                    "real_runtime_ready": False,
                    "failure_reasons": ["summary_missing", "outputs_jsonl_missing"],
                }
                for model_name, task_family in (
                    ("LinearDesign", "cds_protein_conditioned"),
                    ("codonGPT", "cds_protein_conditioned"),
                )
            ],
        }
        with open(os.path.join(docs, "external_sota_real_run_audit.json"), "w", encoding="utf-8") as fh:
            json.dump(audit, fh)

    def _write_frozen_foundation_protocol(self, tmp: str) -> None:
        out_dir = os.path.join(tmp, "benchmark", "frozen_backbone_protocol_head256")
        os.makedirs(out_dir, exist_ok=True)
        summary = {
            "leakage_gate": {
                "enabled": True,
                "audited": True,
                "passed": True,
                "exact_match_count": 0,
            },
            "matched_budget": {
                "trainable_params_consistent": True,
                "trainable_params": 1234,
            },
            "n_real_arms": 1,
            "n_stub_arms": 2,
            "runs": [
                {
                    "backbone": "none",
                    "is_real": True,
                    "finite_loss": True,
                    "valid_quality_signal": True,
                },
                {
                    "backbone": "helix_mrna",
                    "is_real": False,
                    "finite_loss": True,
                    "valid_quality_signal": False,
                },
                {
                    "backbone": "mrnabert",
                    "is_real": False,
                    "finite_loss": True,
                    "valid_quality_signal": False,
                },
            ],
        }
        leakage = {"summary": {"flagged_fraction": 0.0, "exact_match_count": 0}}
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh)
        with open(os.path.join(out_dir, "table.md"), "w", encoding="utf-8") as fh:
            fh.write("# Frozen protocol\n")
        with open(os.path.join(out_dir, "leakage.json"), "w", encoding="utf-8") as fh:
            json.dump(leakage, fh)

    def _write_t1_t7_bundle(self, tmp: str) -> None:
        bench = os.path.join(tmp, "benchmark")
        os.makedirs(bench, exist_ok=True)

        def write_json_md(rel_json, payload, rel_md=None):
            json_path = os.path.join(tmp, rel_json)
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            if rel_md is None:
                rel_md = os.path.splitext(rel_json)[0] + ".md"
            md_path = os.path.join(tmp, rel_md)
            os.makedirs(os.path.dirname(md_path), exist_ok=True)
            with open(md_path, "w", encoding="utf-8") as fh:
                fh.write("# audit\n")

        write_json_md(
            "benchmark/t1_t7_evidence_status_head256.json",
            {
                "artifact_kind": "t1_t7_evidence_status",
                "claim_policy": "proxy only",
                "tasks": [
                    {"task": task, "status": "ready_unit"}
                    for task in ("T1", "T2", "T3", "T4", "T5", "T6", "T7")
                ],
            },
        )
        write_json_md(
            "benchmark/t1_runtime_report_head256_head1024.json",
            {
                "artifact_kind": "t1_runtime_report",
                "interpretation": {"strict_hardware_benchmark_ready": False},
                "rows": [{"label": "head256_mo_grpo"}],
            },
        )
        write_json_md(
            "benchmark/t2_t3_distribution_novelty_report_head256_head1024.json",
            {
                "artifact_kind": "t2_t3_distribution_novelty_report",
                "interpretation": {
                    "primary_head256_distribution_collapse_flag": False,
                    "primary_head256_de_novo_overclaim_flag": True,
                },
                "rows": [{"label": "head256_mo_grpo", "status": "complete"}],
            },
        )
        write_json_md(
            "benchmark/multi_scale_sequence_spectrum_head32_ranker_full1k.json",
            {
                "artifact_kind": "multi_scale_sequence_spectrum_audit",
                "summary": {"ready_for_distribution_figure_audit": True},
                "base_composition": {
                    "full": {
                        "candidate": {"A": 0.25, "C": 0.25, "G": 0.25, "U": 0.25},
                        "source": {"A": 0.25, "C": 0.25, "G": 0.25, "U": 0.25},
                    },
                    "regions": {
                        region: {
                            "candidate": {"A": 0.25, "C": 0.25, "G": 0.25, "U": 0.25},
                            "source": {"A": 0.25, "C": 0.25, "G": 0.25, "U": 0.25},
                        }
                        for region in ("five_utr", "cds", "three_utr")
                    },
                },
                "figures": {
                    "base_composition_full_svg": "figures/base_composition_full.svg",
                    "base_composition_five_utr_svg": "figures/base_composition_five_utr.svg",
                    "base_composition_cds_svg": "figures/base_composition_cds.svg",
                    "base_composition_three_utr_svg": "figures/base_composition_three_utr.svg",
                    "length_histogram_svg": "figures/length_histogram.svg",
                    "gc_histogram_svg": "figures/gc_histogram.svg",
                    "kmer_top_delta_svg": "figures/kmer_top_delta.svg",
                    "codon_pair_top_delta_svg": "figures/codon_pair_top_delta.svg",
                },
            },
        )
        write_json_md(
            "benchmark/t4_protein_identity_cai_gc_report_head256.json",
            {
                "artifact_kind": "t4_protein_identity_cai_gc_report",
                "summary": {
                    "ready": True,
                    "hard_constraints_exact_1": True,
                    "codon_level_metrics_ready": True,
                    "external_baselines_configured": False,
                    "true_mfe_structure_metric_available": False,
                },
            },
        )
        constraint_row = {
            "legal_fraction": 1.0,
            "mean_protein_identity": 1.0,
            "within_budget_fraction": 1.0,
            "reading_frame_intact_fraction": 1.0,
        }
        write_json_md(
            "benchmark/edit_budget_curve_report_head256_head1024.json",
            {
                "artifact_kind": "edit_budget_curve_report",
                "head256_mo_grpo": {"status": "complete", "rows": [constraint_row]},
                "head1024_mo_pareto": {"status": "complete", "rows": [constraint_row]},
            },
        )
        write_json_md(
            "benchmark/t6_length_curve_report_head256_head1024.json",
            {
                "artifact_kind": "t6_length_curve_report",
                "head256_stagea10k": {
                    "status": "complete",
                    "pending_target_length_deltas": [],
                    "rows": [constraint_row],
                },
                "head1024_stagea10k": {
                    "status": "complete",
                    "pending_target_length_deltas": [],
                    "rows": [constraint_row],
                },
            },
        )
        write_json_md(
            "benchmark/t7_motif_frame_report_head256.json",
            {
                "artifact_kind": "t7_motif_frame_report",
                "rows": [
                    {
                        "metric": "reading_frame_intact_fraction",
                        "run": "mo_grpo",
                        "mean": 1.0,
                    }
                ],
            },
        )
        edit_aggregate = {
            key: {"mean": 1.0}
            for key in (
                "insert_mean_protein_identity",
                "insert_reading_frame_intact_fraction",
                "insert_within_budget_fraction",
                "excise_mean_protein_identity",
                "excise_reading_frame_intact_fraction",
                "excise_within_budget_fraction",
                "insert_success_fraction",
                "excise_success_fraction",
            )
        }
        write_json_md(
            "benchmark/t7_motif_edit_benchmark_head256/summary.json",
            {
                "artifact_kind": "t7_motif_edit_benchmark",
                "aggregate": edit_aggregate,
            },
            "benchmark/t7_motif_edit_benchmark_head256/summary.md",
        )

    def test_pending_sota_readiness_lists_missing_sections(self):
        with tempfile.TemporaryDirectory(prefix="mef_sota_ready_pending_") as tmp:
            payload = audit_sota_readiness.audit_sota_readiness(project_root=tmp)
            self.assertFalse(payload["summary"]["all_ready_for_sota_claim_audit"])
            self.assertFalse(payload["summary"]["positive_sota_claim_ready"])
            self.assertIn(
                "internal_evidence_sections_incomplete",
                payload["summary"]["positive_sota_block_reasons"],
            )
            self.assertEqual(
                payload["summary"]["pending_sections"],
                [
                    "region_adapter",
                    "protein_conditioned_gc_sweep",
                    "external_sota_protocol",
                    "multiobjective_scaleup_claims",
                    "frozen_foundation_protocol",
                    "t1_t7_evidence_bundle",
                ],
            )
            self.assertGreaterEqual(len(payload["summary"]["missing_artifacts"]), 9)

    def test_complete_sota_readiness_accepts_all_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="mef_sota_ready_complete_") as tmp:
            bench = os.path.join(tmp, "benchmark")
            os.makedirs(bench)
            for label in (
                "hardneg_v2_top64",
                "mo_grpo_top64",
                "mo_scalar_top64",
                "mo_pareto_top64",
                "mo_te_only_top64",
            ):
                self._write_region_compare(
                    os.path.join(bench, f"region_adapter_vs_{label}_head256.json"),
                    label,
                )
            summarize_region_adapter_comparisons.summarize_region_adapter_comparisons(
                project_root=tmp,
                out_json=os.path.join(bench, "region_adapter_decision_report_head256.json"),
            )
            audit_region_adapter_results.audit_region_adapter_results(
                project_root=tmp,
                out_json=os.path.join(bench, "region_adapter_result_audit_head256.json"),
                out_md=os.path.join(bench, "region_adapter_result_audit_head256.md"),
            )
            self._write_gc_sweep(bench)
            from mrna_editflow.eval import audit_protein_conditioned_gc_sweep

            audit_protein_conditioned_gc_sweep.audit_protein_conditioned_gc_sweep(
                summary_json=os.path.join(
                    bench, "protein_conditioned_cds_gc_sweep_head256.summary.json"
                ),
                jsonl_path=os.path.join(
                    bench, "protein_conditioned_cds_gc_sweep_head256.jsonl"
                ),
                md_path=os.path.join(bench, "protein_conditioned_cds_gc_sweep_head256.md"),
                project_root=tmp,
                out_json=os.path.join(
                    bench, "protein_conditioned_cds_gc_sweep_head256.audit.json"
                ),
                out_md=os.path.join(
                    bench, "protein_conditioned_cds_gc_sweep_head256.audit.md"
                ),
            )
            self._write_external_sota_dry_run(tmp)
            self._write_external_sota_input_pack(tmp)
            self._write_external_sota_real_run_audit(tmp)
            self._write_mo_claim_artifacts(tmp)
            self._write_frozen_foundation_protocol(tmp)
            self._write_t1_t7_bundle(tmp)
            out_json = os.path.join(tmp, "readiness.json")
            out_md = os.path.join(tmp, "readiness.md")
            payload = audit_sota_readiness.audit_sota_readiness(
                project_root=tmp,
                out_json=out_json,
                out_md=out_md,
            )
            self.assertTrue(payload["summary"]["all_ready_for_sota_claim_audit"])
            self.assertFalse(payload["summary"]["positive_sota_claim_ready"])
            self.assertTrue(
                payload["summary"]["ready_for_internal_proxy_constrained_optimization_claim"]
            )
            self.assertFalse(payload["summary"]["ready_for_external_sota_metric_claim"])
            self.assertFalse(payload["summary"]["ready_for_full_de_novo_claim"])
            self.assertFalse(payload["summary"]["ready_for_real_te_or_stability_claim"])
            self.assertFalse(payload["summary"]["ready_for_true_scale_law_claim"])
            self.assertFalse(payload["summary"]["ready_for_wet_lab_claim"])
            self.assertIn(
                "external_sota_real_metrics_missing",
                payload["summary"]["positive_sota_block_reasons"],
            )
            self.assertIn(
                "full_de_novo_evidence_missing_or_overclaim_flagged",
                payload["summary"]["positive_sota_block_reasons"],
            )
            self.assertIn(
                "head1024_fusion_vs_strong_te_only_not_strict",
                payload["summary"]["positive_sota_block_reasons"],
            )
            self.assertEqual(payload["summary"]["pending_sections"], [])
            self.assertEqual(payload["summary"]["n_sections_ready"], 6)
            self.assertTrue(
                payload["sections"]["external_sota_protocol"]["audit"]["summary"]["protocol_ready"]
            )
            self.assertTrue(
                payload["sections"]["external_sota_protocol"]["audit"]["summary"]["input_pack_ready"]
            )
            self.assertTrue(
                payload["sections"]["external_sota_protocol"]["audit"]["summary"]["real_run_audit_complete"]
            )
            self.assertTrue(
                payload["sections"]["external_sota_protocol"]["audit"]["summary"]["model_set_consistent"]
            )
            self.assertEqual(
                payload["sections"]["external_sota_protocol"]["audit"]["summary"][
                    "n_real_run_models_measured"
                ],
                0,
            )
            self.assertTrue(
                payload["sections"]["multiobjective_scaleup_claims"]["audit"]["summary"][
                    "ready_for_full_hard_constraint_claim_audit"
                ]
            )
            self.assertTrue(
                payload["sections"]["frozen_foundation_protocol"]["audit"]["summary"]["protocol_ready"]
            )
            self.assertTrue(
                payload["sections"]["t1_t7_evidence_bundle"]["audit"]["summary"]["bundle_ready"]
            )
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("mRNA-EditFlow SOTA Readiness Audit", text)
            self.assertIn("All ready for SOTA claim audit: True", text)
            self.assertIn("Positive SOTA claim ready: False", text)
            self.assertIn("Full de novo claim ready: False", text)
            self.assertIn("external_sota_protocol", text)
            self.assertIn("multiobjective_scaleup_claims", text)
            self.assertIn("frozen_foundation_protocol", text)
            self.assertIn("t1_t7_evidence_bundle", text)


class TestPaperTable2T1T7(unittest.TestCase):
    def _write_json(self, root, rel_path, payload):
        path = os.path.join(root, rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        return path

    def test_table2_builds_proxy_claim_table_without_overclaiming(self):
        with tempfile.TemporaryDirectory(prefix="mef_table2_") as tmp:
            self._write_json(
                tmp,
                "benchmark/t1_t7_evidence_status_head256.json",
                {
                    "tasks": [
                        {
                            "task": "T1",
                            "evidence": {
                                "legal_fraction": 1.0,
                                "mean_protein_identity": 1.0,
                                "within_budget_fraction": 1.0,
                                "reading_frame_intact_fraction": 1.0,
                                "delta_oracle_te_vs_source": 0.01114,
                                "mean_oracle_te": 0.79097,
                                "mean_oracle_mrl": 8.39,
                            },
                        }
                    ]
                },
            )
            cmp_row = {
                "metric": "delta_oracle_te_vs_source",
                "run": "mo_grpo_top64",
                "paired_p": 0.0045,
                "n_paired_seeds": 10,
            }
            self._write_json(
                tmp,
                "benchmark/compare_mo_fusion_vs_te_only_head256.json",
                {"rows": [cmp_row]},
            )
            self._write_json(
                tmp,
                "benchmark/compare_grpo_vs_hardneg_v2_head256.json",
                {"rows": [{**cmp_row, "run": "mo_grpo"}]},
            )
            self._write_json(
                tmp,
                "benchmark/t2_t3_distribution_novelty_report_head256_head1024.json",
                {
                    "rows": [
                        {
                            "label": "head256_mo_grpo",
                            "T2_distribution": {
                                "kmer_js": {"mean": 0.00001},
                                "codon_usage_kl": {"mean": 0.0},
                                "combined_gc_length_distance": {"mean": 0.002},
                                "candidate_mean_length": {"mean": 101.0},
                                "source_mean_length": {"mean": 101.0},
                            },
                            "T3_novelty_diversity": {
                                "mean_novelty": {"mean": 0.003},
                                "exact_source_match_fraction": {"mean": 0.01},
                                "unique_fraction": {"mean": 0.99},
                                "pairwise_diversity": {"mean": 0.5},
                            },
                        }
                    ]
                },
            )
            self._write_json(
                tmp,
                "benchmark/t4_protein_identity_cai_gc_report_head256.json",
                {
                    "summary": {
                        "hard_constraints_exact_1": True,
                        "codon_level_metrics_ready": True,
                    },
                    "codon_lattice_dp": {"mean_delta_cai": 0.02},
                    "protein_conditioned_cds": {"mean_designed_vs_native_cai_delta": 0.20},
                },
            )
            constraint_row = {
                "budget": 3,
                "delta_oracle_te_vs_source": 0.011,
                "mean_edit_distance": 3.0,
                "within_budget_fraction": 1.0,
                "legal_fraction": 1.0,
                "mean_protein_identity": 1.0,
                "reading_frame_intact_fraction": 1.0,
            }
            self._write_json(
                tmp,
                "benchmark/edit_budget_curve_report_head256_head1024.json",
                {
                    "head256_mo_grpo": {
                        "rows": [
                            constraint_row,
                            {**constraint_row, "budget": 10, "delta_oracle_te_vs_source": 0.028},
                        ]
                    }
                },
            )
            len_row = {
                "mean_abs_length_error": 0.25,
                "legal_fraction": 1.0,
                "mean_protein_identity": 1.0,
                "within_budget_fraction": 1.0,
                "reading_frame_intact_fraction": 1.0,
            }
            self._write_json(
                tmp,
                "benchmark/t6_length_curve_report_head256_head1024.json",
                {
                    "head256_stagea10k": {"rows": [len_row]},
                    "head1024_stagea10k": {"rows": [{**len_row, "mean_abs_length_error": 0.5}]},
                },
            )
            self._write_json(
                tmp,
                "benchmark/t7_motif_frame_report_head256.json",
                {
                    "rows": [
                        {"run": "mo_grpo", "metric": "reading_frame_intact_fraction", "mean": 1.0},
                        {"run": "mo_grpo", "metric": "uAUG_presence_fraction", "mean": 0.34},
                    ],
                    "comparisons_vs_mo_te_only": [
                        {
                            "run": "mo_grpo",
                            "metric": "uAUG_presence_fraction",
                            "paired_p": 0.04048,
                            "n_paired_seeds": 10,
                            "delta": 0.0078,
                            "direction_note": "lower_is_safer",
                        }
                    ],
                },
            )
            edit_metrics = {
                "insert_success_fraction": {"mean": 0.99},
                "excise_success_fraction": {"mean": 0.99},
            }
            self._write_json(
                tmp,
                "benchmark/t7_motif_edit_benchmark_head256/summary.json",
                {"aggregate": edit_metrics},
            )

            report = build_paper_table2_t1_t7.build_paper_table2(tmp)
            self.assertEqual(report["artifact_kind"], "paper_table2_t1_t7_main_results")
            self.assertTrue(report["summary"]["all_tasks_present"])
            self.assertEqual(report["summary"]["paired_p_available_tasks"], ["T1", "T7"])
            self.assertFalse(report["summary"]["ready_for_external_sota_claim"])
            self.assertFalse(report["summary"]["ready_for_wet_lab_claim"])
            self.assertFalse(report["summary"]["ready_for_full_de_novo_claim"])
            rows = {row["task"]: row for row in report["rows"]}
            self.assertIn("not de novo", rows["T3"]["claim_language"])
            self.assertIn("do not claim uAUG safety improvement", rows["T7"]["claim_language"])

            out_json = os.path.join(tmp, "table2.json")
            out_md = os.path.join(tmp, "table2.md")
            build_paper_table2_t1_t7.write_report_json(report, out_json)
            build_paper_table2_t1_t7.write_report_markdown(report, out_md)
            with open(out_json, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["summary"]["n_tasks"], 7)
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Paper Table 2", text)
            self.assertIn("External SOTA claim ready: `False`", text)
            self.assertIn("p=0.00450", text)


class TestPaperTable1SotaLandscape(unittest.TestCase):
    def test_table1_classifies_methods_and_refuses_external_metric_claims(self):
        with tempfile.TemporaryDirectory(prefix="mef_table1_") as tmp:
            docs = os.path.join(tmp, "docs")
            dry = os.path.join(tmp, "benchmark", "external_sota", "dry_run_t5_head1024")
            os.makedirs(docs, exist_ok=True)
            os.makedirs(dry, exist_ok=True)
            with open(os.path.join(docs, "mrna_dataset_survey.md"), "w", encoding="utf-8") as fh:
                fh.write(
                    "GEMORNA mRNA-GPT ProMORNA RNAGenScape codonGPT UTailoR "
                    "LinearDesign split leakage license"
                )
            with open(os.path.join(dry, "summary.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "rows": [
                            {
                                "model_name": "LinearDesign",
                                "status": "not_configured",
                            },
                            {
                                "model_name": "codonGPT",
                                "status": "not_configured",
                            },
                        ]
                    },
                    fh,
                )

            report = build_paper_table1_sota_landscape.build_paper_table1(tmp)
            self.assertEqual(report["artifact_kind"], "paper_table1_sota_landscape")
            self.assertEqual(report["summary"]["n_methods"], 12)
            self.assertTrue(report["summary"]["ready_for_landscape_table"])
            self.assertFalse(report["summary"]["ready_for_external_metric_claim"])
            self.assertFalse(report["summary"]["ready_for_wet_lab_claim"])
            categories = report["summary"]["category_counts"]
            self.assertGreaterEqual(categories["full-length de novo generation"], 2)
            self.assertGreaterEqual(categories["CDS-only structure/codon optimization"], 2)
            rows = {row["method"]: row for row in report["rows"]}
            self.assertEqual(rows["LinearDesign"]["dry_run_status"], "not_configured")
            self.assertEqual(rows["mRNA-GPT"]["category"], "full-length de novo generation")
            self.assertEqual(
                rows["mRNA-GPT"]["claim_language"],
                "landscape_only_no_measured_external_metric",
            )

            out_json = os.path.join(tmp, "table1.json")
            out_md = os.path.join(tmp, "table1.md")
            build_paper_table1_sota_landscape.write_report_json(report, out_json)
            build_paper_table1_sota_landscape.write_report_markdown(report, out_md)
            with open(out_json, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["summary"]["n_executable_ready"], 0)
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Paper Table 1", text)
            self.assertIn("landscape/protocol", text)
            self.assertIn("external metric claim ready: `False`", text)
            self.assertIn("full-length de novo generation", text)


class TestExternalSotaRealRunAudit(unittest.TestCase):
    def _sha256(self, path):
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _write_input_pack(self, tmp):
        from mrna_editflow.baselines.external_sota_input_pack import build_external_sota_input_pack

        records_jsonl = os.path.join(tmp, "records.jsonl")
        write_records_jsonl(_source_records(), records_jsonl)
        return build_external_sota_input_pack(
            records_jsonl=records_jsonl,
            out_dir=os.path.join(tmp, "benchmark", "external_sota", "input_pack_t5_head1024"),
            limit=2,
            split_name="unit_external_real",
            seed=11,
        )

    def test_external_real_run_audit_records_missing_measured_outputs(self):
        with tempfile.TemporaryDirectory(prefix="mef_external_real_missing_") as tmp:
            self._write_input_pack(tmp)
            payload = audit_external_sota_real_runs.audit_external_sota_real_runs(
                project_root=tmp,
                models=["LinearDesign", "UTailoR"],
            )
            self.assertEqual(payload["artifact_kind"], "external_sota_real_run_audit")
            self.assertTrue(payload["summary"]["audit_complete"])
            self.assertEqual(payload["summary"]["n_models_expected"], 2)
            self.assertEqual(payload["summary"]["n_models_measured"], 0)
            self.assertEqual(payload["summary"]["n_models_missing"], 2)
            self.assertFalse(payload["summary"]["ready_for_external_real_metric_table"])
            self.assertFalse(payload["summary"]["ready_for_external_sota_claim"])
            self.assertEqual({row["status"] for row in payload["rows"]}, {"missing"})

    def test_external_real_run_audit_accepts_complete_cds_model_contract(self):
        with tempfile.TemporaryDirectory(prefix="mef_external_real_measured_") as tmp:
            pack = self._write_input_pack(tmp)
            outputs = pack["outputs"]
            summary_path = outputs["summary_json"]
            real_dir = os.path.join(tmp, "benchmark", "external_sota", "real_runs_t5_head1024", "LinearDesign")
            os.makedirs(real_dir, exist_ok=True)
            cds_inputs = []
            with open(outputs["cds_protein_jsonl"], "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        cds_inputs.append(json.loads(line))
            rows = []
            for row in cds_inputs:
                rows.append(
                    {
                        "transcript_id": row["transcript_id"],
                        "model_name": "LinearDesign",
                        "designed_cds": row["native_cds"],
                        "wall_clock_s": 0.01,
                        "valid_cds": True,
                        "protein_identity": 1.0,
                        "protein_identity_exact_1": True,
                        "cai": 0.75,
                        "gc": 0.5,
                        "gc3": 0.5,
                        "codon_usage_kl_vs_native": 0.0,
                        "codon_pair_kl_vs_native": 0.0,
                    }
                )
            outputs_jsonl = os.path.join(real_dir, "cds_outputs.jsonl")
            with open(outputs_jsonl, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
            real_summary = {
                "artifact_kind": "external_sota_real_run_summary",
                "model_name": "LinearDesign",
                "task_family": "cds_protein_conditioned",
                "protocol_fidelity": "unit_complete_protocol",
                "protocol_fidelity_sufficient_for_sota_reproduction": True,
                "input_pack": {
                    "summary_sha256": self._sha256(summary_path),
                    "cds_protein_jsonl_sha256": outputs["cds_protein_jsonl_sha256"],
                    "utr5_jsonl_sha256": outputs["utr5_jsonl_sha256"],
                },
                "dataset": {
                    "records_jsonl_sha256": pack["dataset"]["records_jsonl_sha256"],
                    "split_name": pack["dataset"]["split_name"],
                    "seed": pack["dataset"]["seed"],
                },
                "runtime": {"elapsed_s": 0.25},
                "hardware": {"label": "unit-cpu"},
                "executable": {"path": "/bin/echo", "version": "unit-version"},
                "n_inputs": len(cds_inputs),
                "n_outputs": len(rows),
                "n_failures": 0,
                "mean_wall_clock_s": 0.01,
                "valid_cds_fraction": 1.0,
                "protein_identity_exact_1_fraction": 1.0,
                "mean_cai": 0.75,
                "mean_gc": 0.5,
                "mean_gc3": 0.5,
            }
            with open(os.path.join(real_dir, "summary.json"), "w", encoding="utf-8") as fh:
                json.dump(real_summary, fh, indent=2, sort_keys=True)

            out_json = os.path.join(tmp, "audit.json")
            out_md = os.path.join(tmp, "audit.md")
            payload = audit_external_sota_real_runs.audit_external_sota_real_runs(
                project_root=tmp,
                models=["LinearDesign"],
                out_json=out_json,
                out_md=out_md,
            )
            self.assertTrue(payload["summary"]["audit_complete"])
            self.assertTrue(payload["summary"]["ready_for_external_real_metric_table"])
            self.assertTrue(payload["summary"]["ready_for_external_sota_metric_claim"])
            self.assertFalse(payload["summary"]["ready_for_external_sota_claim"])
            self.assertEqual(payload["rows"][0]["status"], "measured")
            self.assertTrue(payload["rows"][0]["hard_constraints_exact_1"])
            self.assertEqual(payload["rows"][0]["failure_reasons"], [])
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("External SOTA Real-Run Audit", text)
            self.assertIn("Measured models: `1` / `1`", text)

    def test_utrgan_audit_prefers_complete_paper_default_variant(self):
        with tempfile.TemporaryDirectory(prefix="mef_utrgan_variant_") as tmp:
            pack = self._write_input_pack(tmp)
            outputs = pack["outputs"]
            with open(
                outputs["utr5_jsonl"],
                "r",
                encoding="utf-8",
            ) as fh:
                inputs = [json.loads(line) for line in fh if line.strip()]
            real_root = os.path.join(
                tmp,
                "benchmark/external_sota/real_runs_t5_head1024",
            )

            def write_variant(directory_name, protocol, sufficient):
                out_dir = os.path.join(real_root, directory_name)
                os.makedirs(out_dir, exist_ok=True)
                rows = [
                    {
                        "transcript_id": row["transcript_id"],
                        "model_name": "UTRGAN",
                        "designed_five_utr": row["native_five_utr"],
                        "wall_clock_s": 0.01,
                        "cds_unchanged": True,
                        "three_utr_unchanged": True,
                        "protein_identity_exact_1": True,
                        "te_proxy": 0.8,
                        "te_proxy_delta_vs_native": 0.0,
                        "uaug_count": 0.0,
                        "kozak_score": 0.5,
                        "start_accessibility_proxy": 0.6,
                    }
                    for row in inputs
                ]
                with open(
                    os.path.join(out_dir, "utr5_outputs.jsonl"),
                    "w",
                    encoding="utf-8",
                ) as fh:
                    for row in rows:
                        fh.write(json.dumps(row, sort_keys=True) + "\n")
                summary = {
                    "artifact_kind": "external_sota_real_run_summary",
                    "model_name": "UTRGAN",
                    "task_family": "utr5_only",
                    "protocol_fidelity": protocol,
                    "protocol_fidelity_sufficient_for_sota_reproduction": (
                        sufficient
                    ),
                    "input_pack": {
                        "summary_sha256": self._sha256(
                            outputs["summary_json"]
                        ),
                        "cds_protein_jsonl_sha256": outputs[
                            "cds_protein_jsonl_sha256"
                        ],
                        "utr5_jsonl_sha256": outputs[
                            "utr5_jsonl_sha256"
                        ],
                    },
                    "dataset": {
                        "records_jsonl_sha256": pack["dataset"][
                            "records_jsonl_sha256"
                        ],
                        "split_name": pack["dataset"]["split_name"],
                        "seed": pack["dataset"]["seed"],
                    },
                    "runtime": {"elapsed_s": 1.0},
                    "hardware": {"label": "unit-cpu"},
                    "executable": {
                        "path": "/bin/echo",
                        "version": "unit-version",
                    },
                    "n_inputs": len(inputs),
                    "n_outputs": len(rows),
                    "n_failures": 0,
                    "mean_wall_clock_s": 0.01,
                    "cds_unchanged_fraction": 1.0,
                    "three_utr_unchanged_fraction": 1.0,
                    "protein_identity_exact_1_fraction": 1.0,
                    "mean_te_proxy_delta_vs_native": 0.0,
                }
                with open(
                    os.path.join(out_dir, "summary.json"),
                    "w",
                    encoding="utf-8",
                ) as fh:
                    json.dump(summary, fh, indent=2, sort_keys=True)

            write_variant(
                "UTRGAN",
                "official_code_budgeted_10_steps_vs_paper_default_10000",
                False,
            )
            canonical = (
                audit_external_sota_real_runs.audit_external_sota_real_runs(
                    project_root=tmp,
                    models=["UTRGAN"],
                )
            )
            self.assertEqual(
                canonical["rows"][0]["selected_evidence_variant"],
                "canonical",
            )
            write_variant(
                "UTRGAN_paper10000",
                "official_code_paper_default_10000_steps",
                True,
            )
            paper = (
                audit_external_sota_real_runs.audit_external_sota_real_runs(
                    project_root=tmp,
                    models=["UTRGAN"],
                )
            )
            row = paper["rows"][0]
            self.assertEqual(
                row["selected_evidence_variant"],
                "paper_default_10000_steps",
            )
            self.assertEqual(row["status"], "measured")
            self.assertTrue(
                row[
                    "protocol_fidelity_sufficient_for_sota_reproduction"
                ]
            )


class TestExternalSotaEvidenceManifest(unittest.TestCase):
    def test_required_bundle_digest_ignores_root_and_dynamic_run_files(self):
        with tempfile.TemporaryDirectory(prefix="mef_manifest_a_") as root_a:
            with tempfile.TemporaryDirectory(
                prefix="mef_manifest_b_"
            ) as root_b:
                for root in (root_a, root_b):
                    with open(
                        os.path.join(root, "README.md"),
                        "w",
                        encoding="utf-8",
                    ) as fh:
                        fh.write("same required evidence\n")
                active = os.path.join(
                    root_b,
                    "benchmark/external_sota/real_runs_t5_head1024/"
                    "EnsembleDesign/progress.jsonl",
                )
                os.makedirs(os.path.dirname(active), exist_ok=True)
                with open(active, "w", encoding="utf-8") as fh:
                    fh.write('{"completed": 1}\n')

                report_a = (
                    build_external_sota_evidence_manifest
                    .build_external_sota_evidence_manifest(root_a)
                )
                report_b = (
                    build_external_sota_evidence_manifest
                    .build_external_sota_evidence_manifest(root_b)
                )
                digest_a = report_a["summary"][
                    "required_bundle_digest_sha256"
                ]
                digest_b = report_b["summary"][
                    "required_bundle_digest_sha256"
                ]
                self.assertEqual(digest_a, digest_b)
                self.assertEqual(len(digest_a), 64)
                self.assertEqual(
                    report_a["summary"]["n_active_external_files_present"],
                    0,
                )
                self.assertEqual(
                    report_b["summary"]["n_active_external_files_present"],
                    1,
                )

                with open(
                    os.path.join(root_b, "README.md"),
                    "w",
                    encoding="utf-8",
                ) as fh:
                    fh.write("changed required evidence\n")
                changed = (
                    build_external_sota_evidence_manifest
                    .build_external_sota_evidence_manifest(root_b)
                )
                self.assertNotEqual(
                    digest_a,
                    changed["summary"]["required_bundle_digest_sha256"],
                )


class TestPaperTable3ExternalBaselines(unittest.TestCase):
    def _write_table3_dry_run_fixture(self, tmp):
        out_dir = os.path.join(tmp, "benchmark", "external_sota", "dry_run_t5_head1024")
        os.makedirs(out_dir, exist_ok=True)
        dataset_sha = "b" * 64
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "status": "dry_run_complete",
                    "task_id": "T5",
                    "n_models": 1,
                    "n_executable_ready": 0,
                    "n_not_configured": 1,
                    "dataset": {
                        "exists": True,
                        "sha256": dataset_sha,
                        "record_count_effective": 2,
                        "record_count_total": 2,
                        "split_name": "unit_split",
                        "seed": 3,
                    },
                    "hardware": {
                        "label": "unit-host",
                        "hostname": "unit",
                        "machine": "arm64",
                    },
                    "artifact_contract": {
                        "required_real_run_metadata": [
                            "dataset.sha256",
                            "dataset.split_name",
                            "dataset.seed",
                            "dataset.record_count_effective",
                            "runtime.elapsed_s",
                            "hardware",
                        ],
                        "real_metric_policy": "Do not report real metrics.",
                    },
                    "rows": [
                        {
                            "model_name": "LinearDesign",
                            "status": "not_configured",
                            "family": "dynamic-programming/lattice parsing mRNA optimizer",
                            "executable": None,
                            "executable_source": None,
                            "command_candidates": ["LINEARDESIGN_BIN", "lineardesign"],
                            "candidate_audit": [
                                {"candidate": "LINEARDESIGN_BIN", "status": "env_unset"},
                                {"candidate": "lineardesign", "status": "path_not_found"},
                            ],
                            "expected_inputs": "protein",
                            "expected_outputs": "CDS",
                            "protocol_difference": "CDS-only optimizer.",
                        }
                    ],
                },
                fh,
            )
        with open(os.path.join(out_dir, "runtime.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "dataset_sha256": dataset_sha,
                    "elapsed_s": 0.25,
                    "hardware": {"label": "unit-host"},
                },
                fh,
            )
        with open(os.path.join(out_dir, "table.md"), "w", encoding="utf-8") as fh:
            fh.write("# External SOTA Dry-Run\n")
        return out_dir

    def test_table3_summarizes_dry_run_without_real_metric_claims(self):
        with tempfile.TemporaryDirectory(prefix="mef_table3_") as tmp:
            self._write_table3_dry_run_fixture(tmp)

            report = build_paper_table3_external_baselines.build_paper_table3(tmp)
            self.assertEqual(report["artifact_kind"], "paper_table3_external_baseline_readiness")
            self.assertTrue(report["summary"]["metadata_contract_ok"])
            self.assertTrue(report["summary"]["ready_for_protocol_table"])
            self.assertFalse(report["summary"]["ready_for_real_metric_table"])
            self.assertFalse(report["summary"]["ready_for_external_sota_claim"])
            self.assertFalse(report["summary"]["input_pack_ready"])
            self.assertFalse(report["input_pack"]["present"])
            self.assertFalse(report["real_run_audit"]["present"])
            self.assertFalse(report["summary"]["real_run_audit_complete"])
            self.assertEqual(report["summary"]["n_executable_ready"], 0)
            self.assertEqual(report["rows"][0]["claim_language"], "not_configured_no_external_metric_claim")
            self.assertIn("LINEARDESIGN_BIN=env_unset", report["rows"][0]["candidate_audit_text"])

            out_json = os.path.join(tmp, "table3.json")
            out_md = os.path.join(tmp, "table3.md")
            build_paper_table3_external_baselines.write_report_json(report, out_json)
            build_paper_table3_external_baselines.write_report_markdown(report, out_md)
            with open(out_json, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertFalse(loaded["summary"]["ready_for_real_metric_table"])
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Paper Table 3", text)
            self.assertIn("real metric table ready: `False`", text)
            self.assertIn("ready for external real run: `False`", text)
            self.assertIn("External Real-Run Audit", text)
            self.assertIn("LINEARDESIGN_BIN=env_unset", text)

    def test_table3_reports_external_input_pack_when_present(self):
        with tempfile.TemporaryDirectory(prefix="mef_table3_pack_") as tmp:
            self._write_table3_dry_run_fixture(tmp)
            pack_dir = os.path.join(tmp, "benchmark", "external_sota", "input_pack_t5_head1024")
            os.makedirs(pack_dir, exist_ok=True)
            pack_summary = {
                "artifact_kind": "external_sota_input_pack",
                "ready_for_external_real_run": True,
                "ready_for_external_sota_claim": False,
                "models": {
                    "cds_protein_conditioned": ["LinearDesign"],
                    "utr5_only": [],
                },
                "n_cds_protein_rows": 2,
                "n_utr5_rows": 2,
                "n_skipped_invalid_cds": 0,
            }
            with open(os.path.join(pack_dir, "summary.json"), "w", encoding="utf-8") as fh:
                json.dump(pack_summary, fh, sort_keys=True)

            report = build_paper_table3_external_baselines.build_paper_table3(tmp)
            self.assertTrue(report["summary"]["input_pack_ready"])
            self.assertTrue(report["input_pack"]["present"])
            self.assertTrue(report["input_pack"]["ready_for_external_real_run"])
            self.assertEqual(report["input_pack"]["n_cds_protein_rows"], 2)
            self.assertRegex(report["input_pack"]["sha256"], r"^[0-9a-f]{64}$")
            self.assertFalse(report["real_run_audit"]["present"])
            self.assertFalse(report["summary"]["model_set_consistent"])
            self.assertFalse(report["summary"]["ready_for_external_sota_claim"])

            out_md = os.path.join(tmp, "table3_pack.md")
            build_paper_table3_external_baselines.write_report_markdown(report, out_md)
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("External Input Pack", text)
            self.assertIn("ready for external real run: `True`", text)
            self.assertIn("CDS/protein-conditioned rows: `2`", text)
            self.assertIn("real metric table ready: `False`", text)

    def test_table3_reports_real_run_audit_when_present(self):
        with tempfile.TemporaryDirectory(prefix="mef_table3_real_audit_") as tmp:
            self._write_table3_dry_run_fixture(tmp)
            docs = os.path.join(tmp, "docs")
            os.makedirs(docs, exist_ok=True)
            audit = {
                "artifact_kind": "external_sota_real_run_audit",
                "summary": {
                    "audit_complete": True,
                    "ready_for_external_real_metric_table": False,
                    "ready_for_external_sota_metric_claim": False,
                    "ready_for_external_sota_claim": False,
                    "n_models_expected": 1,
                    "n_models_measured": 0,
                    "n_models_invalid": 0,
                    "n_models_missing": 1,
                },
                "rows": [
                    {
                        "model_name": "LinearDesign",
                        "status": "missing",
                        "task_family": "cds_protein_conditioned",
                        "expected_input_rows": 2,
                        "n_outputs": 0,
                        "success_fraction": 0.0,
                        "hard_constraints_exact_1": False,
                        "real_metric_ready": False,
                        "real_runtime_ready": False,
                        "failure_reasons": ["summary_missing", "outputs_jsonl_missing"],
                    }
                ],
            }
            with open(os.path.join(docs, "external_sota_real_run_audit.json"), "w", encoding="utf-8") as fh:
                json.dump(audit, fh, sort_keys=True)

            report = build_paper_table3_external_baselines.build_paper_table3(tmp)
            self.assertTrue(report["real_run_audit"]["present"])
            self.assertTrue(report["summary"]["real_run_audit_complete"])
            self.assertEqual(report["real_run_audit"]["n_models_missing"], 1)
            self.assertEqual(report["rows"][0]["real_run_status"], "missing")
            self.assertFalse(report["rows"][0]["real_metric_ready"])
            self.assertFalse(report["summary"]["ready_for_real_metric_table"])
            self.assertFalse(report["summary"]["model_set_consistent"])


class TestPaperTable4ArchitectureAblation(unittest.TestCase):
    def _write_json(self, root, rel_path, payload):
        path = os.path.join(root, rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        return path

    def test_table4_keeps_negative_and_non_significant_ablation_language(self):
        with tempfile.TemporaryDirectory(prefix="mef_table4_") as tmp:
            run = {
                "run": "region_adapter_all_top64",
                "constraints_exact_1": True,
                "primary_by_baseline": {
                    "hardneg_v2_top64": {
                        "delta": -0.01,
                        "paired_p": 0.0045,
                        "run_mean": -0.001,
                        "baseline_mean": 0.005,
                        "signal": "not_positive",
                    }
                },
            }
            decision = {
                "summary": {"best_run_vs_hardneg": "region_adapter_all_top64"},
                "runs": [run],
            }
            self._write_json(tmp, "benchmark/region_adapter_decision_report_head256.json", decision)
            self._write_json(tmp, "benchmark/region_adapter_decision_report_head1024.json", decision)
            audit = {"summary": {"all_constraints_exact_1": True}}
            self._write_json(tmp, "benchmark/region_adapter_result_audit_head256.json", audit)
            self._write_json(tmp, "benchmark/region_adapter_result_audit_head1024.json", audit)
            self._write_json(
                tmp,
                "benchmark/codon_lattice_dp_head256.json",
                {
                    "summary": {
                        "protein_identity_fraction": 1.0,
                        "mean_delta_cai": 0.02,
                        "mean_delta_gc": 0.01,
                        "mean_codon_changes": 3.0,
                    }
                },
            )
            proposal_base = {
                "aggregate": {
                    "oracle_best_in_model_top_k_fraction": 0.03,
                    "mean_model_regret": 0.04,
                }
            }
            proposal_ranker = {
                "aggregate": {
                    "oracle_best_in_model_top_k_fraction": 0.42,
                    "mean_model_regret": 0.03,
                }
            }
            proposal_source = {
                "aggregate": {
                    "oracle_best_in_model_top_k_fraction": 0.75,
                    "mean_model_regret": 0.031,
                }
            }
            self._write_json(tmp, "benchmark/proposal_ranking_t5_base_full1k_head64.json", proposal_base)
            self._write_json(tmp, "benchmark/proposal_ranking_t5_ranker_full1k_final_head64.json", proposal_ranker)
            self._write_json(tmp, "benchmark/proposal_ranking_t5_sourceaware_hybrid_teacher_head64.json", proposal_source)
            self._write_json(
                tmp,
                "benchmark/cascade_sourceaware_to_sequential_head64_k64.json",
                {"aggregate": {"oracle_best_in_recall_top_k_fraction": 0.80}},
            )
            self._write_json(
                tmp,
                "benchmark/compare_t5_head256_cascade_vs_seq_top64.json",
                {
                    "rows": [
                        {
                            "metric": "delta_oracle_te_vs_source",
                            "run": "cascade_top64",
                            "delta": 0.001,
                            "paired_p": 0.10,
                        }
                    ]
                },
            )
            self._write_json(
                tmp,
                "benchmark/compare_cascade_10krecall_vs_hardneg_v2.json",
                {
                    "rows": [
                        {
                            "metric": "delta_oracle_te_vs_source",
                            "run": "cascade_10krecall_hardneg_top64",
                            "delta": -0.0005,
                            "paired_p": 0.36,
                            "run_mean": 0.0045,
                        }
                    ]
                },
            )
            self._write_json(
                tmp,
                "benchmark/cascade_error_analysis_head256_top64.json",
                {"aggregate": {"win_record_fraction": 0.55}},
            )

            report = build_paper_table4_architecture_ablation.build_paper_table4(tmp)
            self.assertEqual(report["artifact_kind"], "paper_table4_architecture_ablation")
            self.assertTrue(report["summary"]["table_ready_for_architecture_draft"])
            self.assertFalse(report["summary"]["positive_sota_claim_ready"])
            self.assertTrue(report["summary"]["negative_results_included"])
            signals = {row["signal"] for row in report["rows"]}
            self.assertIn("negative_ablation", signals)
            self.assertIn("trend_not_significant", signals)
            self.assertIn("not_positive", signals)
            rows = {row["module"]: row for row in report["rows"]}
            self.assertIn("failed/negative", rows["Region adapters / FiLM-style conditioning"]["claim_language"])
            self.assertIn("does not by itself prove top-1 TE", rows["Source-aware teacher / recall ranker"]["claim_language"])

            out_json = os.path.join(tmp, "table4.json")
            out_md = os.path.join(tmp, "table4.md")
            build_paper_table4_architecture_ablation.write_report_json(report, out_json)
            build_paper_table4_architecture_ablation.write_report_markdown(report, out_md)
            with open(out_json, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["summary"]["n_modules"], 5)
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Paper Table 4", text)
            self.assertIn("positive SOTA claim ready: `False`", text)
            self.assertIn("negative_ablation", text)


class TestPaperTable5ScaleLawReadiness(unittest.TestCase):
    def _write_json(self, root, rel_path, payload):
        path = os.path.join(root, rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        return path

    def _constraint_row(self, **extra):
        row = {
            "delta_oracle_te_vs_source": 0.0,
            "legal_fraction": 1.0,
            "mean_protein_identity": 1.0,
            "reading_frame_intact_fraction": 1.0,
            "within_budget_fraction": 1.0,
        }
        row.update(extra)
        return row

    def test_table5_reports_readiness_without_true_scale_law_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_json(
                tmp,
                "docs/multiobjective_scaleup_claim_audit_head256_head1024.json",
                {
                    "summary": {
                        "available_compare_constraints_exact_1": True,
                        "head1024_vs_hardneg_v2_all_strict": True,
                        "head1024_vs_te_only_best_signal": "borderline_positive",
                        "head1024_vs_te_only_strict_claim_allowed": False,
                        "summary_constraints_complete": True,
                    },
                    "comparisons": [
                        {
                            "comparison_id": "head256_mo_grpo_top64_vs_te_only",
                            "delta": 0.0076,
                            "n_records": 256,
                            "paired_p": 0.0045,
                            "signal": "strict_positive",
                        },
                        {
                            "comparison_id": "head1024_mo_pareto_top64_vs_te_only",
                            "delta": 0.0008,
                            "n_records": 1024,
                            "paired_p": 0.0504,
                            "signal": "borderline_positive",
                        },
                        {
                            "comparison_id": "head1024_mo_pareto_top64_vs_hardneg_v2",
                            "delta": 0.0054,
                            "n_records": 1024,
                            "paired_p": 0.0045,
                            "signal": "strict_positive",
                        },
                    ],
                },
            )
            self._write_json(
                tmp,
                "benchmark/t1_runtime_report_head256_head1024.json",
                {
                    "rows": [
                        {
                            "config": {"n_records": 256},
                            "context": {"delta_oracle_te_vs_source": {"mean": 0.011}},
                            "label": "head256_mo_grpo",
                            "runtime": {
                                "measured_records_per_s_total": 2.1,
                                "observed_elapsed_scope": "complete_seed_runtime",
                                "seed_total_s": {"mean": 120.0},
                            },
                        },
                        {
                            "config": {"n_records": 1024},
                            "context": {"delta_oracle_te_vs_source": {"mean": 0.009}},
                            "label": "head1024_mo_pareto",
                            "runtime": {
                                "measured_records_per_s_total": 8.4,
                                "observed_elapsed_scope": "complete_run_wall_clock",
                                "seed_total_s": {"mean": 121.0},
                            },
                        },
                    ]
                },
            )
            self._write_json(
                tmp,
                "benchmark/edit_budget_curve_report_head256_head1024.json",
                {
                    "head256_mo_grpo": {
                        "rows": [
                            self._constraint_row(budget=1, delta_oracle_te_vs_source=0.004),
                            self._constraint_row(budget=3, delta_oracle_te_vs_source=0.011),
                            self._constraint_row(budget=10, delta_oracle_te_vs_source=0.028),
                        ],
                        "status": "complete",
                    },
                    "head1024_mo_pareto": {
                        "rows": [
                            self._constraint_row(budget=1, delta_oracle_te_vs_source=0.003),
                            self._constraint_row(budget=3, delta_oracle_te_vs_source=0.009),
                            self._constraint_row(budget=10, delta_oracle_te_vs_source=0.024),
                        ],
                        "status": "complete",
                    },
                },
            )
            length_rows = [
                self._constraint_row(
                    target_length_delta=delta,
                    mean_abs_length_error=0.0 if delta == 0 else 0.5,
                    delta_oracle_te_vs_source=-0.02 if delta > 0 else 0.001,
                )
                for delta in (-30, -15, 0, 15, 30)
            ]
            self._write_json(
                tmp,
                "benchmark/t6_length_curve_report_head256_head1024.json",
                {
                    "head256_stagea10k": {"rows": length_rows, "status": "complete"},
                    "head1024_stagea10k": {"rows": length_rows, "status": "complete"},
                },
            )

            report = build_paper_table5_scale_law_readiness.build_paper_table5(tmp)
            self.assertEqual(report["artifact_kind"], "paper_table5_scale_law_readiness")
            self.assertTrue(report["summary"]["table_ready_for_scale_law_readiness_draft"])
            self.assertFalse(report["summary"]["ready_for_true_scale_law_claim"])
            self.assertFalse(report["summary"]["ready_for_monotonic_scale_law_claim"])
            self.assertTrue(report["summary"]["yield_contraction_flag"])
            self.assertEqual(report["summary"]["head1024_vs_te_only_best_signal"], "borderline_positive")
            self.assertEqual(
                report["summary"]["missing_required_axes"],
                ["training_data_size", "model_size", "training_steps"],
            )
            signals = {row["signal"] for row in report["rows"]}
            self.assertIn("observed_yield_contraction", signals)
            self.assertIn("required_scale_law_axes_missing", signals)
            self.assertTrue(report["summary"]["hard_constraints_exact_1"])

            out_json = os.path.join(tmp, "table5.json")
            out_md = os.path.join(tmp, "table5.md")
            build_paper_table5_scale_law_readiness.write_report_json(report, out_json)
            build_paper_table5_scale_law_readiness.write_report_markdown(report, out_md)
            with open(out_json, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertFalse(loaded["summary"]["ready_for_true_scale_law_claim"])
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Paper Table 5", text)
            self.assertIn("ready for true scale-law claim: `False`", text)
            self.assertIn("required_scale_law_axes_missing", text)

    def test_table5_detects_queued_controlled_scalelaw_sweep_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_json(
                tmp,
                "docs/multiobjective_scaleup_claim_audit_head256_head1024.json",
                {
                    "summary": {
                        "available_compare_constraints_exact_1": True,
                        "head1024_vs_hardneg_v2_all_strict": True,
                        "head1024_vs_te_only_best_signal": "borderline_positive",
                        "head1024_vs_te_only_strict_claim_allowed": False,
                        "summary_constraints_complete": True,
                    },
                    "comparisons": [
                        {
                            "comparison_id": "head256_mo_grpo_top64_vs_te_only",
                            "delta": 0.0076,
                            "paired_p": 0.0045,
                            "signal": "strict_positive",
                        },
                        {
                            "comparison_id": "head1024_mo_pareto_top64_vs_te_only",
                            "delta": 0.0008,
                            "paired_p": 0.0504,
                            "signal": "borderline_positive",
                        },
                        {
                            "comparison_id": "head1024_mo_pareto_top64_vs_hardneg_v2",
                            "delta": 0.0054,
                            "paired_p": 0.0045,
                            "signal": "strict_positive",
                        },
                    ],
                },
            )
            self._write_json(
                tmp,
                "benchmark/t1_runtime_report_head256_head1024.json",
                {
                    "rows": [
                        {
                            "config": {"n_records": 256},
                            "context": {"delta_oracle_te_vs_source": {"mean": 0.011}},
                            "label": "head256_mo_grpo",
                            "runtime": {"measured_records_per_s_total": 2.0, "seed_total_s": {"mean": 10.0}},
                        },
                        {
                            "config": {"n_records": 1024},
                            "context": {"delta_oracle_te_vs_source": {"mean": 0.009}},
                            "label": "head1024_mo_pareto",
                            "runtime": {"measured_records_per_s_total": 8.0, "seed_total_s": {"mean": 11.0}},
                        },
                    ]
                },
            )
            self._write_json(
                tmp,
                "benchmark/edit_budget_curve_report_head256_head1024.json",
                {
                    "head256_mo_grpo": {"rows": [self._constraint_row(budget=3, delta_oracle_te_vs_source=0.011)]},
                    "head1024_mo_pareto": {"rows": [self._constraint_row(budget=3, delta_oracle_te_vs_source=0.009)]},
                },
            )
            length_rows = [
                self._constraint_row(
                    target_length_delta=delta,
                    mean_abs_length_error=0.0,
                    delta_oracle_te_vs_source=0.0,
                )
                for delta in (-30, -15, 0, 15, 30)
            ]
            self._write_json(
                tmp,
                "benchmark/t6_length_curve_report_head256_head1024.json",
                {
                    "head256_stagea10k": {"rows": length_rows},
                    "head1024_stagea10k": {"rows": length_rows},
                },
            )
            self._write_json(
                tmp,
                "benchmark/stage_a_scalelaw_test/plan.json",
                {
                    "artifact_kind": "stage_a_scalelaw_sweep_plan",
                    "axes": {
                        "data_sizes": [256, 1024],
                        "model_sizes": ["tiny", "small"],
                        "step_counts": [200, 500],
                        "seeds": [0],
                    },
                    "claim_policy": "queued only",
                    "n_runs": 8,
                    "runs": [],
                },
            )
            progress_path = os.path.join(tmp, "benchmark/stage_a_scalelaw_test/progress.jsonl")
            with open(progress_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"event": "load_gate_wait", "loadavg": "101.0"}) + "\n")

            report = build_paper_table5_scale_law_readiness.build_paper_table5(tmp)
            self.assertTrue(report["summary"]["controlled_sweep_plan_ready"])
            self.assertFalse(report["summary"]["controlled_sweep_complete"])
            self.assertEqual(report["summary"]["controlled_sweep_n_runs"], 8)
            self.assertEqual(report["summary"]["controlled_sweep_last_event"], "load_gate_wait")
            self.assertEqual(report["summary"]["missing_required_axes"], [])
            self.assertEqual(
                report["summary"]["incomplete_required_axes"],
                ["training_data_size", "model_size", "training_steps"],
            )
            signals = {row["signal"] for row in report["rows"]}
            self.assertIn("required_scale_law_axes_queued_incomplete", signals)
            self.assertEqual(
                report["controlled_sweep_audit"]["axes"]["data_sizes"],
                [256, 1024],
            )


class TestDatasetManifestAudit(unittest.TestCase):
    def _write_json(self, root, rel_path, payload):
        path = os.path.join(root, rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        return path

    def test_manifest_audit_distinguishes_complete_and_missing_split_stats(self):
        with tempfile.TemporaryDirectory(prefix="mef_manifest_audit_") as tmp:
            records_path = os.path.join(tmp, "data", "processed", "records.jsonl")
            os.makedirs(os.path.dirname(records_path), exist_ok=True)
            write_records_jsonl([MRNARecord("r1", "AAA", "AUGGCUUAA", "UUU")], records_path)
            import hashlib

            with open(records_path, "rb") as fh:
                records_sha = hashlib.sha256(fh.read()).hexdigest()
            complete_manifest = {
                "dataset": {
                    "name": "gencode_human_transcripts",
                    "registry_url": "https://example.test/gencode/",
                    "files": [
                        {
                            "public_url": "https://example.test/gencode.fa.gz",
                            "sha256": "b" * 64,
                        }
                    ],
                },
                "records_path": records_path,
                "records_sha256": records_sha,
                "raw_summary": {"n_records": 2},
                "clean_summary": {"n_records": 1},
                "cleaning_drop_counts": {"total": 2, "kept": 1, "bad": 1},
            }
            self._write_json(
                tmp,
                "data/processed/gencode_human_transcripts.data_manifest.json",
                complete_manifest,
            )
            self._write_json(
                tmp,
                "benchmark/gencode_family_leakage_protocol/report.json",
                {
                    "artifact_kind": "family_leakage_protocol",
                    "summary": {
                        "external_records_provided": True,
                        "split_ready": True,
                        "synthetic_smoke_only": False,
                    },
                    "split": {
                        "n_train": 1,
                        "n_val": 0,
                        "n_test": 0,
                        "n_clusters": 1,
                    },
                },
            )
            incomplete = dict(complete_manifest)
            incomplete["dataset"] = {
                "name": "refseq_human_rna",
                "registry_url": "https://example.test/refseq/",
                "files": [{"public_url": "https://example.test/refseq.gbff.gz", "sha256": "c" * 64}],
            }
            incomplete.pop("split_stats", None)
            self._write_json(
                tmp,
                "data/processed/refseq_human_rna.data_manifest.json",
                incomplete,
            )

            report = dataset_manifest_audit.build_dataset_manifest_audit(tmp)
            self.assertEqual(report["artifact_kind"], "dataset_manifest_audit")
            rows = {row["dataset"]: row for row in report["rows"]}
            self.assertTrue(rows["gencode_human_transcripts"]["complete"])
            self.assertTrue(rows["gencode_human_transcripts"]["split_sidecar"]["split_stats_ready"])
            self.assertFalse(rows["refseq_human_rna"]["complete"])
            self.assertIn("split_stats", rows["refseq_human_rna"]["missing_fields"])
            self.assertIn("mpra_te", report["summary"]["incomplete_datasets"])
            self.assertFalse(report["summary"]["all_required_dataset_manifests_complete"])

            out_json = os.path.join(tmp, "audit.json")
            out_md = os.path.join(tmp, "audit.md")
            dataset_manifest_audit.write_report_json(report, out_json)
            dataset_manifest_audit.write_report_markdown(report, out_md)
            with open(out_json, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["summary"]["n_manifests_present"], 2)
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Dataset Manifest Audit", text)
            self.assertIn("split_stats", text)

    def test_manifest_audit_reports_pending_split_sidecar_without_completing(self):
        with tempfile.TemporaryDirectory(prefix="mef_manifest_pending_") as tmp:
            records_path = os.path.join(tmp, "data", "processed", "records.jsonl")
            os.makedirs(os.path.dirname(records_path), exist_ok=True)
            write_records_jsonl([MRNARecord("r1", "AAA", "AUGGCUUAA", "UUU")], records_path)
            import hashlib

            with open(records_path, "rb") as fh:
                records_sha = hashlib.sha256(fh.read()).hexdigest()
            self._write_json(
                tmp,
                "data/processed/gencode_human_transcripts.data_manifest.json",
                {
                    "dataset": {
                        "name": "gencode_human_transcripts",
                        "registry_url": "https://example.test/gencode/",
                        "files": [{"public_url": "https://example.test/gencode.fa.gz", "sha256": "b" * 64}],
                    },
                    "records_path": records_path,
                    "records_sha256": records_sha,
                    "raw_summary": {"n_records": 2},
                    "clean_summary": {"n_records": 1},
                    "cleaning_drop_counts": {"total": 2, "kept": 1, "bad": 1},
                },
            )
            self._write_json(
                tmp,
                "benchmark/gencode_family_leakage_protocol/status.json",
                {
                    "artifact_kind": "gencode_family_leakage_protocol_status",
                    "status": "queued_or_running",
                    "progress": {"last_event": "load_gate_wait", "last_loadavg": "101.0"},
                },
            )
            report = dataset_manifest_audit.build_dataset_manifest_audit(tmp)
            rows = {row["dataset"]: row for row in report["rows"]}
            gencode = rows["gencode_human_transcripts"]
            self.assertFalse(gencode["complete"])
            self.assertIn("split_stats", gencode["missing_fields"])
            self.assertTrue(gencode["split_sidecar"]["split_stats_pending"])
            self.assertEqual(gencode["split_sidecar"]["last_event"], "load_gate_wait")
            self.assertIn("gencode_human_transcripts", report["summary"]["pending_split_sidecars"])


class TestDownstreamTableManifestBuilder(unittest.TestCase):
    def test_builds_mpra_and_stability_manifests_that_pass_dataset_audit(self):
        with tempfile.TemporaryDirectory(prefix="mef_downstream_manifest_") as tmp:
            raw_dir = os.path.join(tmp, "data", "raw")
            out_dir = os.path.join(tmp, "data", "processed")
            os.makedirs(raw_dir, exist_ok=True)
            mpra_path = os.path.join(raw_dir, "sample_mpra.csv")
            stability_path = os.path.join(raw_dir, "sample_half_life.tsv")
            with open(mpra_path, "w", encoding="utf-8") as fh:
                fh.write("sample_id,sequence,mrl,split\n")
                fh.write("m0,GCCACCAAA,8.0,train\n")
                fh.write("m1,GCCACCGGG,9.0,val\n")
                fh.write("m2,AAAAACCCC,4.0,test\n")
            with open(stability_path, "w", encoding="utf-8") as fh:
                fh.write("sample_id\tsequence\thalf_life\tsplit\n")
                fh.write("h0\tAAUAAAGGGG\t9.0\ttrain\n")
                fh.write("h1\tAUUUAUUUU\t3.0\tval\n")
                fh.write("h2\tAAUAAACCCC\t8.0\ttest\n")

            mpra_result = build_downstream_table_manifest.build_downstream_table_manifest(
                dataset_name="mpra_te",
                input_path=mpra_path,
                out_dir=out_dir,
                source_url="https://example.test/sample_mpra.csv",
                license_text="test fixture",
            )
            stability_result = build_downstream_table_manifest.build_downstream_table_manifest(
                dataset_name="stability_half_life",
                input_path=stability_path,
                out_dir=out_dir,
                source_url="https://example.test/sample_half_life.tsv",
                license_text="test fixture",
            )

            self.assertTrue(os.path.exists(mpra_result.records_path))
            self.assertTrue(os.path.exists(stability_result.manifest_path))
            self.assertEqual(mpra_result.n_raw, 3)
            self.assertEqual(stability_result.n_clean, 3)
            self.assertEqual(len(mpra_result.manifest_sha256), 64)

            report = dataset_manifest_audit.build_dataset_manifest_audit(tmp)
            rows = {row["dataset"]: row for row in report["rows"]}
            self.assertTrue(rows["mpra_te"]["complete"])
            self.assertTrue(rows["stability_half_life"]["complete"])
            self.assertTrue(rows["mpra_te"]["records"]["sha256_matches"])
            self.assertEqual(rows["mpra_te"]["missing_fields"], [])
            self.assertEqual(rows["stability_half_life"]["missing_fields"], [])
            self.assertFalse(report["summary"]["all_required_dataset_manifests_complete"])
            self.assertIn("gencode_human_transcripts", report["summary"]["incomplete_datasets"])

            with open(mpra_result.manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            self.assertFalse(manifest["ready_for_real_te_or_stability_claim"])
            self.assertTrue(manifest["split_stats"]["official_split_ready"])

    def test_requires_official_split_by_default(self):
        with tempfile.TemporaryDirectory(prefix="mef_downstream_manifest_split_") as tmp:
            raw_dir = os.path.join(tmp, "data", "raw")
            os.makedirs(raw_dir, exist_ok=True)
            mpra_path = os.path.join(raw_dir, "sample_mpra.csv")
            with open(mpra_path, "w", encoding="utf-8") as fh:
                fh.write("sample_id,sequence,mrl\n")
                fh.write("m0,GCCACCAAA,8.0\n")
                fh.write("m1,GCCACCGGG,9.0\n")

            with self.assertRaisesRegex(ValueError, "official train/val/test split"):
                build_downstream_table_manifest.build_downstream_table_manifest(
                    dataset_name="mpra_te",
                    input_path=mpra_path,
                    out_dir=os.path.join(tmp, "data", "processed"),
                    source_url="https://example.test/sample_mpra.csv",
                )


class TestDownstreamDataAcquisitionAudit(unittest.TestCase):
    def test_empty_project_keeps_all_real_downstream_gates_closed(self):
        with tempfile.TemporaryDirectory(prefix="mef_downstream_acq_empty_") as tmp:
            report = downstream_data_acquisition_audit.build_downstream_data_acquisition_audit(tmp)
            self.assertEqual(report["artifact_kind"], "downstream_data_acquisition_audit")
            self.assertEqual(report["summary"]["n_datasets"], 2)
            self.assertEqual(report["summary"]["n_source_tables_present"], 0)
            self.assertEqual(report["summary"]["n_manifests_complete"], 0)
            self.assertFalse(report["summary"]["ready_for_real_te_or_stability_claim"])
            rows = {row["dataset"]: row for row in report["rows"]}
            self.assertEqual(rows["mpra_te"]["status"], "needs_source_table_download")
            self.assertIn("source_table", rows["mpra_te"]["missing_gates"])

            out_json = os.path.join(tmp, "audit.json")
            out_md = os.path.join(tmp, "audit.md")
            downstream_data_acquisition_audit.write_report_json(report, out_json)
            downstream_data_acquisition_audit.write_report_markdown(report, out_md)
            with open(out_json, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["summary"]["n_schema_ready"], 0)
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Downstream Data Acquisition Audit", text)
            self.assertIn("Ready for real TE/stability claim: `False`", text)

    def test_manifest_ready_table_still_needs_predictor_and_leakage_gates(self):
        with tempfile.TemporaryDirectory(prefix="mef_downstream_acq_real_") as tmp:
            raw_dir = os.path.join(tmp, "data", "raw")
            out_dir = os.path.join(tmp, "data", "processed")
            os.makedirs(raw_dir, exist_ok=True)
            mpra_path = os.path.join(raw_dir, "sample_mpra.csv")
            stability_path = os.path.join(raw_dir, "sample_half_life.tsv")
            with open(mpra_path, "w", encoding="utf-8") as fh:
                fh.write("sample_id,sequence,mrl,split\n")
                fh.write("m0,GCCACCAAA,8.0,train\n")
                fh.write("m1,GCCACCGGG,9.0,val\n")
                fh.write("m2,AAAAACCCC,4.0,test\n")
            with open(stability_path, "w", encoding="utf-8") as fh:
                fh.write("sample_id\tsequence\thalf_life\tsplit\n")
                fh.write("h0\tAAUAAAGGGG\t9.0\ttrain\n")
                fh.write("h1\tAUUUAUUUU\t3.0\tval\n")
                fh.write("h2\tAAUAAACCCC\t8.0\ttest\n")
            build_downstream_table_manifest.build_downstream_table_manifest(
                dataset_name="mpra_te",
                input_path=mpra_path,
                out_dir=out_dir,
                source_url="https://example.test/sample_mpra.csv",
            )
            build_downstream_table_manifest.build_downstream_table_manifest(
                dataset_name="stability_half_life",
                input_path=stability_path,
                out_dir=out_dir,
                source_url="https://example.test/sample_half_life.tsv",
            )
            manifest_audit = dataset_manifest_audit.build_dataset_manifest_audit(tmp)
            dataset_manifest_audit.write_report_json(
                manifest_audit,
                os.path.join(tmp, "docs", "dataset_manifest_audit.json"),
            )
            data_scale = build_data_scaleup_readiness.build_data_scaleup_readiness(tmp)
            build_data_scaleup_readiness.write_report_json(
                data_scale,
                os.path.join(tmp, "docs", "data_scaleup_readiness.json"),
            )

            report = downstream_data_acquisition_audit.build_downstream_data_acquisition_audit(tmp)
            rows = {row["dataset"]: row for row in report["rows"]}
            self.assertEqual(report["summary"]["n_source_tables_present"], 2)
            self.assertEqual(report["summary"]["n_schema_ready"], 2)
            self.assertEqual(report["summary"]["n_manifests_complete"], 2)
            self.assertEqual(report["summary"]["n_predictor_audits_ready"], 0)
            self.assertFalse(report["summary"]["ready_for_real_te_or_stability_claim"])
            self.assertEqual(rows["mpra_te"]["status"], "manifest_ready_needs_heldout_predictor")
            self.assertIn("heldout_predictor_report", rows["mpra_te"]["missing_gates"])
            self.assertIn("leakage_documentation", rows["stability_half_life"]["missing_gates"])


class TestDataScaleupReadiness(unittest.TestCase):
    def _write_json(self, root, rel_path, payload):
        path = os.path.join(root, rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        return path

    def test_reports_manifest_queue_and_missing_scaleup_without_overclaiming(self):
        with tempfile.TemporaryDirectory(prefix="mef_data_scaleup_") as tmp:
            for rel in ("data/prepare_mpra.py", "data/dedup_split.py", "data/leakage_audit.py"):
                path = os.path.join(tmp, rel)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("# test fixture\n")
            self._write_json(
                tmp,
                "data/processed/gencode_human_transcripts.data_manifest.json",
                {
                    "dataset": {"name": "gencode_human_transcripts"},
                    "records_path": "/remote/mrna_editflow/data/processed/gencode_human_transcripts.records.jsonl",
                    "records_sha256": "a" * 64,
                    "raw_summary": {"n_records": 100},
                    "clean_summary": {"n_records": 80},
                    "cleaning_drop_counts": {"kept": 80, "total": 100},
                },
            )
            self._write_json(
                tmp,
                "benchmark/refseq_public_build_test/status.json",
                {
                    "artifact_kind": "refseq_public_build_status",
                    "claim_policy": "RefSeq queue only",
                    "status": "queued_or_running",
                    "raw": {"exists": False, "path": "data/raw/human.1.rna.gbff.gz"},
                    "records": {
                        "exists": False,
                        "path": "data/processed/refseq_human_rna.records.jsonl",
                    },
                    "manifest": {
                        "exists": False,
                        "path": "data/processed/refseq_human_rna.data_manifest.json",
                    },
                    "progress": {
                        "last_event": "load_gate_wait",
                        "last_loadavg": "101.0",
                        "n_events": 3,
                    },
                },
            )
            self._write_json(
                tmp,
                "benchmark/stage_a_scalelaw_test/status.json",
                {
                    "artifact_kind": "stage_a_scalelaw_sweep_status",
                    "claim_policy": "queued sweep only",
                    "plan": {
                        "axes": {
                            "data_sizes": [256, 1024],
                            "model_sizes": ["tiny", "small"],
                            "step_counts": [200, 500],
                            "seeds": [0],
                        },
                        "source_record_count": 80,
                        "source_records_sha256": "a" * 64,
                    },
                    "summary": {
                        "n_runs": 8,
                        "n_complete": 0,
                        "n_incomplete": 8,
                        "status_counts": {"queued": 8},
                        "last_event": "load_gate_wait",
                        "last_loadavg": "101.0",
                        "n_load_gate_wait_events": 4,
                    },
                },
            )
            out_json = os.path.join(tmp, "docs", "data_scaleup_readiness.json")
            out_md = os.path.join(tmp, "docs", "data_scaleup_readiness.md")
            payload = build_data_scaleup_readiness.build_data_scaleup_readiness(tmp)
            build_data_scaleup_readiness.write_report_json(payload, out_json)
            build_data_scaleup_readiness.write_report_markdown(payload, out_md)

            summary = payload["summary"]
            self.assertFalse(summary["ready_for_data_scale_claim"])
            self.assertFalse(summary["ready_for_refseq_scaleup_claim"])
            self.assertTrue(summary["gencode_manifest_ready"])
            self.assertFalse(summary["gencode_records_local_exists"])
            self.assertFalse(summary["refseq_official_corpus_ready"])
            self.assertEqual(summary["refseq_build_status"], "queued_or_running")
            self.assertFalse(summary["stage_a_controlled_sweep_complete"])
            self.assertFalse(summary["stage_a_downstream_eval_ready"])
            self.assertEqual(
                summary["stage_a_downstream_eval_status"],
                "missing_stage_a_downstream_eval_readiness",
            )
            self.assertFalse(summary["mpra_real_data_ready"])
            self.assertFalse(summary["family_leakage_ready"])
            self.assertIn("refseq_official_corpus", summary["missing_or_incomplete"])
            self.assertIn("stage_a_downstream_evaluation", summary["missing_or_incomplete"])
            self.assertEqual(
                payload["gencode_manifest_audit"]["status"],
                "manifest_ready_records_not_local",
            )
            self.assertEqual(
                payload["family_split_leakage_audit"]["status"],
                "blocked_on_refseq_records",
            )
            with open(out_json, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["artifact_kind"], "data_scaleup_readiness")
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Data Scale-Up Readiness", text)
            self.assertIn("Ready for data scale claim: `False`", text)
            self.assertIn("RefSeq parser/build queue readiness only", text)

    def test_real_gencode_family_split_is_not_collapsed_to_smoke(self):
        with tempfile.TemporaryDirectory(prefix="mef_data_scaleup_gencode_split_") as tmp:
            for rel in ("data/dedup_split.py", "data/leakage_audit.py"):
                path = os.path.join(tmp, rel)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("# test fixture\n")
            split_path = os.path.join(
                tmp,
                "benchmark",
                "gencode_family_leakage_protocol",
                "splits",
                "train.idx",
            )
            os.makedirs(os.path.dirname(split_path), exist_ok=True)
            with open(split_path, "w", encoding="utf-8") as fh:
                fh.write("0\n1\n")
            self._write_json(
                tmp,
                "benchmark/family_leakage_protocol_smoke/report.json",
                {
                    "artifact_kind": "family_leakage_protocol",
                    "summary": {
                        "synthetic_smoke_only": True,
                        "external_records_provided": False,
                        "split_ready": True,
                        "ready_for_family_leakage_audit": False,
                    },
                },
            )
            self._write_json(
                tmp,
                "benchmark/gencode_family_leakage_protocol/report.json",
                {
                    "artifact_kind": "family_leakage_protocol",
                    "summary": {
                        "synthetic_smoke_only": False,
                        "external_records_provided": True,
                        "external_reference_provided": False,
                        "split_ready": True,
                        "ready_for_family_leakage_audit": False,
                        "ready_for_family_disjoint_leakage_claim": False,
                    },
                },
            )

            payload = build_data_scaleup_readiness.build_data_scaleup_readiness(tmp)
            summary = payload["summary"]
            split = payload["family_split_leakage_audit"]

            self.assertFalse(summary["ready_for_data_scale_claim"])
            self.assertFalse(summary["family_leakage_ready"])
            self.assertTrue(summary["family_split_protocol_ready"])
            self.assertTrue(summary["family_leakage_protocol_real_report_present"])
            self.assertFalse(summary["family_leakage_protocol_synthetic_smoke_only"])
            self.assertEqual(
                split["status"],
                "blocked_on_refseq_records_real_gencode_split_ready",
            )
            self.assertIn(
                "benchmark/gencode_family_leakage_protocol/report.json",
                split["real_protocol_reports"],
            )
            self.assertIn(
                "benchmark/gencode_family_leakage_protocol/splits/train.idx",
                split["split_files"],
            )

    def test_downstream_real_tables_are_schema_audited_without_overclaiming(self):
        with tempfile.TemporaryDirectory(prefix="mef_downstream_tables_") as tmp:
            for rel in (
                "data/prepare_mpra.py",
                "eval/mpra_te_predictor.py",
                "eval/stability_predictor.py",
                "data/dedup_split.py",
                "data/leakage_audit.py",
            ):
                path = os.path.join(tmp, rel)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("# test fixture\n")

            mpra_path = os.path.join(tmp, "data", "raw", "sample_mpra_mrl.csv")
            stability_path = os.path.join(tmp, "data", "raw", "sample_half_life.tsv")
            os.makedirs(os.path.dirname(mpra_path), exist_ok=True)
            with open(mpra_path, "w", encoding="utf-8") as fh:
                fh.write("sequence,mrl,split\n")
                fh.write("ACGUACGU,1.0,train\n")
                fh.write("CGUACGUA,1.2,val\n")
                fh.write("GCAUGCAU,1.4,test\n")
            with open(stability_path, "w", encoding="utf-8") as fh:
                fh.write("sequence\thalf_life\tsplit\n")
                fh.write("AUGGCUUAA\t4.0\ttrain\n")
                fh.write("AUGCCCUAA\t4.5\tval\n")
                fh.write("AUGGGGUAA\t5.0\ttest\n")

            payload = build_data_scaleup_readiness.build_data_scaleup_readiness(tmp)
            summary = payload["summary"]
            downstream = payload["mpra_te_stability_audit"]

            self.assertTrue(summary["mpra_real_data_ready"])
            self.assertTrue(summary["mpra_input_table_ready"])
            self.assertTrue(summary["stability_input_table_ready"])
            self.assertTrue(summary["downstream_input_tables_ready"])
            self.assertFalse(summary["ready_for_real_te_or_stability_claim"])
            self.assertFalse(summary["mpra_te_predictor_audit_ready"])
            self.assertFalse(summary["stability_predictor_audit_ready"])
            self.assertIn("real_mpra_te_stability_data", summary["missing_or_incomplete"])
            self.assertEqual(
                downstream["mpra_input_table"]["status"],
                "schema_and_official_split_ready",
            )
            self.assertEqual(
                downstream["stability_input_table"]["status"],
                "schema_and_official_split_ready",
            )
            self.assertEqual(downstream["mpra_input_table"]["selected_table"], "data/raw/sample_mpra_mrl.csv")
            self.assertEqual(
                downstream["stability_input_table"]["selected_table"],
                "data/raw/sample_half_life.tsv",
            )


class TestFamilyLeakageProtocol(unittest.TestCase):
    def test_synthetic_family_leakage_smoke_keeps_claims_closed(self):
        with tempfile.TemporaryDirectory(prefix="mef_family_leakage_") as tmp:
            out_split = os.path.join(tmp, "splits")
            report = family_leakage_protocol.run_family_leakage_protocol(
                out_split_dir=out_split,
                n_synthetic=32,
                seed=13,
                use_mmseqs="never",
            )
            self.assertEqual(report["artifact_kind"], "family_leakage_protocol")
            self.assertTrue(report["summary"]["synthetic_smoke_only"])
            self.assertTrue(report["summary"]["split_ready"])
            self.assertFalse(report["summary"]["ready_for_family_leakage_audit"])
            self.assertFalse(report["summary"]["ready_for_family_disjoint_leakage_claim"])
            self.assertTrue(os.path.exists(os.path.join(out_split, "train.idx")))
            self.assertTrue(os.path.exists(os.path.join(out_split, "val.idx")))
            self.assertTrue(os.path.exists(os.path.join(out_split, "test.idx")))

            out_json = os.path.join(tmp, "report.json")
            out_md = os.path.join(tmp, "report.md")
            family_leakage_protocol.write_report_json(report, out_json)
            family_leakage_protocol.write_report_markdown(report, out_md)
            with open(out_json, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["summary"]["n_records"], 32)
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Family Leakage Protocol", text)
            self.assertIn("ready for leakage-free claim: `False`", text)

    def test_external_records_and_reference_can_open_audit_ready_gate(self):
        with tempfile.TemporaryDirectory(prefix="mef_family_leakage_real_") as tmp:
            records = synthesize_corpus(48, seed=101)
            refs = synthesize_corpus(48, seed=909)
            records_path = os.path.join(tmp, "records.jsonl")
            refs_path = os.path.join(tmp, "refs.jsonl")
            write_records_jsonl(records, records_path)
            write_records_jsonl(refs, refs_path)
            report = family_leakage_protocol.run_family_leakage_protocol(
                records_path=records_path,
                reference_path=refs_path,
                out_split_dir=os.path.join(tmp, "splits"),
                seed=5,
                use_mmseqs="never",
                kmer=9,
            )
            self.assertFalse(report["summary"]["synthetic_smoke_only"])
            self.assertTrue(report["summary"]["external_records_provided"])
            self.assertTrue(report["summary"]["external_reference_provided"])
            self.assertTrue(report["summary"]["split_ready"])
            self.assertTrue(report["summary"]["ready_for_family_leakage_audit"])
            self.assertFalse(report["summary"]["ready_for_family_disjoint_leakage_claim"])
            self.assertEqual(report["summary"]["leakage_exact_match_count"], 0)


class TestMPRATEPredictorProtocol(unittest.TestCase):
    def test_external_mpra_fixture_reports_split_metrics_without_sota_claim(self):
        with tempfile.TemporaryDirectory(prefix="mef_mpra_te_") as tmp:
            csv_path = os.path.join(tmp, "mpra.csv")
            rows = [
                ("GCCACCAAAACCC", 8.0, "train"),
                ("GCCACCGGGGCCC", 9.0, "train"),
                ("AAAAAACCCCUUU", 4.0, "train"),
                ("CCCCGGGGCCCCG", 6.0, "train"),
                ("GCCACCACACACG", 8.5, "val"),
                ("UUUUUAAAAAUUU", 3.0, "val"),
                ("GCCACCGCGCGCG", 9.5, "test"),
                ("AUAUAUAUAUAUA", 2.5, "test"),
            ]
            with open(csv_path, "w", encoding="utf-8") as fh:
                fh.write("sample_id,sequence,mrl,split\n")
                for i, (seq, mrl, split) in enumerate(rows):
                    fh.write(f"s{i},{seq},{mrl},{split}\n")

            report, predictions = mpra_te_predictor.run_mpra_te_predictor(
                input_path=csv_path,
                ridge_alpha=0.1,
                min_test_n=2,
            )
            self.assertEqual(report["artifact_kind"], "mpra_te_predictor_protocol")
            self.assertTrue(report["summary"]["external_input_provided"])
            self.assertTrue(report["summary"]["official_split_present"])
            self.assertTrue(report["summary"]["ready_for_mpra_te_predictor_audit"])
            self.assertFalse(report["summary"]["ready_for_real_te_or_stability_claim"])
            self.assertFalse(report["summary"]["ready_for_wet_lab_design_claim"])
            self.assertEqual(report["metrics"]["test"]["n"], 2)
            self.assertTrue(math.isfinite(float(report["metrics"]["test"]["mae"])))
            self.assertEqual(len(predictions), len(rows))

            out_json = os.path.join(tmp, "report.json")
            out_md = os.path.join(tmp, "report.md")
            out_pred = os.path.join(tmp, "predictions.jsonl")
            mpra_te_predictor.write_report_json(report, out_json)
            mpra_te_predictor.write_report_markdown(report, out_md)
            mpra_te_predictor.write_predictions_jsonl(predictions, out_pred)
            with open(out_json, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["model"]["model_family"], "feature_ridge_regression")
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("MPRA TE Predictor Protocol", text)
            self.assertIn("ready for real TE/stability claim: `False`", text)
            with open(out_pred, "r", encoding="utf-8") as fh:
                pred_rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(pred_rows), len(rows))

    def test_synthetic_mpra_smoke_is_not_real_te_evidence(self):
        report, _predictions = mpra_te_predictor.run_mpra_te_predictor(
            n_synthetic=12,
            seed=7,
            min_test_n=1,
        )
        self.assertTrue(report["summary"]["synthetic_smoke_only"])
        self.assertFalse(report["summary"]["external_input_provided"])
        self.assertFalse(report["summary"]["ready_for_mpra_te_predictor_audit"])
        self.assertFalse(report["summary"]["ready_for_real_te_or_stability_claim"])
        self.assertGreaterEqual(report["summary"]["n_train"], 1)


class TestStabilityPredictorProtocol(unittest.TestCase):
    def test_external_stability_fixture_reports_split_metrics_without_claim(self):
        with tempfile.TemporaryDirectory(prefix="mef_stability_") as tmp:
            csv_path = os.path.join(tmp, "stability.csv")
            rows = [
                ("AAUAAAGGGGCCCCAAUAAA", 9.0, "train"),
                ("AUUUACCCCAAAAUUUU", 3.0, "train"),
                ("GGGGCCCCGGGG", 4.0, "train"),
                ("AAUAAACCCCAAUAAA", 8.0, "train"),
                ("AUUUAUUUUAUUUA", 2.0, "val"),
                ("AAUAAAGGAAUAAA", 8.5, "val"),
                ("AAAAAAAUAAAGGG", 7.5, "test"),
                ("AUUUAUUUUGGGG", 2.5, "test"),
            ]
            with open(csv_path, "w", encoding="utf-8") as fh:
                fh.write("sample_id,sequence,half_life,split\n")
                for i, (seq, half_life, split) in enumerate(rows):
                    fh.write(f"h{i},{seq},{half_life},{split}\n")

            report, predictions = stability_predictor.run_stability_predictor(
                input_path=csv_path,
                ridge_alpha=0.1,
                min_test_n=2,
            )
            self.assertEqual(report["artifact_kind"], "stability_predictor_protocol")
            self.assertTrue(report["summary"]["external_input_provided"])
            self.assertTrue(report["summary"]["official_split_present"])
            self.assertTrue(report["summary"]["ready_for_stability_predictor_audit"])
            self.assertFalse(report["summary"]["ready_for_real_te_or_stability_claim"])
            self.assertEqual(report["input"]["target_name"], "half_life")
            self.assertEqual(report["metrics"]["test"]["n"], 2)
            self.assertTrue(math.isfinite(float(report["metrics"]["test"]["rmse"])))
            self.assertEqual(len(predictions), len(rows))

            out_json = os.path.join(tmp, "report.json")
            out_md = os.path.join(tmp, "report.md")
            out_pred = os.path.join(tmp, "predictions.jsonl")
            stability_predictor.write_report_json(report, out_json)
            stability_predictor.write_report_markdown(report, out_md)
            stability_predictor.write_predictions_jsonl(predictions, out_pred)
            with open(out_json, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["model"]["model_family"], "feature_ridge_regression")
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Stability Predictor Protocol", text)
            self.assertIn("ready for real TE/stability claim: `False`", text)
            with open(out_pred, "r", encoding="utf-8") as fh:
                pred_rows = [json.loads(line) for line in fh if line.strip()]
            self.assertEqual(len(pred_rows), len(rows))

    def test_synthetic_stability_smoke_is_not_real_stability_evidence(self):
        report, _predictions = stability_predictor.run_stability_predictor(
            n_synthetic=16,
            seed=11,
            min_test_n=1,
        )
        self.assertTrue(report["summary"]["synthetic_smoke_only"])
        self.assertFalse(report["summary"]["external_input_provided"])
        self.assertFalse(report["summary"]["ready_for_stability_predictor_audit"])
        self.assertFalse(report["summary"]["ready_for_real_te_or_stability_claim"])
        self.assertGreaterEqual(report["summary"]["n_train"], 1)


class TestPaperFigure1FullLengthEditFlow(unittest.TestCase):
    def _write_text(self, root, rel_path, text):
        path = os.path.join(root, rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def test_figure1_spec_keeps_full_length_and_claim_boundaries_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_text(tmp, "README.md", "5'UTR + CDS + 3'UTR edit-flow")
            self._write_text(tmp, "sample.py", "# decoding entrypoint\n")
            self._write_text(tmp, "eval/run_multiseed_benchmark.py", "# benchmark entrypoint\n")
            self._write_text(tmp, "benchmark/t1_t7_evidence_status_head256.json", "{}\n")

            report = build_paper_figure1_full_length_edit_flow.build_paper_figure1(tmp)
            self.assertEqual(report["artifact_kind"], "paper_figure1_full_length_edit_flow")
            self.assertTrue(report["summary"]["ready_for_algorithm_figure_draft"])
            self.assertFalse(report["summary"]["ready_for_full_de_novo_claim"])
            self.assertFalse(report["summary"]["ready_for_wet_lab_claim"])
            self.assertTrue(report["summary"]["hard_constraints_visible"])
            self.assertEqual(report["summary"]["full_length_segments"], ["5'UTR", "CDS", "3'UTR"])
            self.assertTrue(report["summary"]["source_files_ready"])
            self.assertEqual(report["summary"]["n_nodes"], 9)
            self.assertEqual(report["summary"]["n_edges"], 9)
            labels = {node["label"] for node in report["nodes"]}
            self.assertIn("Hard-constraint masks", labels)
            self.assertIn("Multi-objective ranker / fusion", labels)
            self.assertIn("T1-T7 evaluation gates", labels)
            self.assertIn("must not depict", report["claim_policy"])
            self.assertIn("unconstrained de novo", report["claim_policy"])
            self.assertIn("constrained local-optimization claim only", report["caption"])
            self.assertIn("5'UTR + CDS + 3'UTR", report["caption"])
            self.assertIn("flowchart LR", report["mermaid"])

            out_json = os.path.join(tmp, "figure1.json")
            out_md = os.path.join(tmp, "figure1.md")
            build_paper_figure1_full_length_edit_flow.write_report_json(report, out_json)
            build_paper_figure1_full_length_edit_flow.write_report_markdown(report, out_md)
            with open(out_json, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertFalse(loaded["summary"]["ready_for_wet_lab_claim"])
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Paper Figure 1", text)
            self.assertIn("```mermaid", text)
            self.assertIn("ready for full de novo claim: `False`", text)
            self.assertIn("CDS protein identity", text)


class TestPaperFigure2CascadeRecallPrecision(unittest.TestCase):
    def _write_json(self, root, rel_path, payload):
        path = os.path.join(root, rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        return path

    def _compare_payload(self, run, delta, paired_p, run_mean=0.0, baseline_mean=0.0):
        rows = [
            {
                "baseline_mean": baseline_mean,
                "delta": delta,
                "metric": "delta_oracle_te_vs_source",
                "n_paired_seeds": 10,
                "paired_p": paired_p,
                "run": run,
                "run_mean": run_mean,
            }
        ]
        for metric in (
            "legal_fraction",
            "within_budget_fraction",
            "mean_protein_identity",
            "reading_frame_intact_fraction",
        ):
            rows.append(
                {
                    "baseline_mean": 1.0,
                    "delta": 0.0,
                    "metric": metric,
                    "n_paired_seeds": 10,
                    "paired_p": 1.0,
                    "run": run,
                    "run_mean": 1.0,
                }
            )
        return {"rows": rows}

    def test_figure2_spec_keeps_cascade_trend_non_significant(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_json(
                tmp,
                "benchmark/proposal_ranking_t5_ranker_full1k_final_head64.json",
                {"aggregate": {"oracle_best_in_model_top_k_fraction": 0.42623}},
            )
            self._write_json(
                tmp,
                "benchmark/proposal_ranking_t5_sourceaware_hybrid_teacher_head64.json",
                {
                    "aggregate": {
                        "mean_model_regret": 0.03088,
                        "oracle_best_in_model_top_k_fraction": 0.75410,
                    }
                },
            )
            self._write_json(
                tmp,
                "benchmark/cascade_sourceaware_to_sequential_head64_k64.json",
                {
                    "aggregate": {
                        "mean_cascade_regret": 0.02788,
                        "mean_precision_full_regret": 0.02798,
                        "n_candidates": 54222,
                        "n_records": 64,
                        "oracle_best_in_recall_top_k_fraction": 0.80328,
                    }
                },
            )
            self._write_json(
                tmp,
                "benchmark/compare_t5_head256_cascade_vs_seq_top64.json",
                self._compare_payload("cascade_top64", 0.00092, 0.09545, 0.00482, 0.00391),
            )
            self._write_json(
                tmp,
                "benchmark/compare_t5_head256_hardneg_v2_top64.json",
                self._compare_payload("hardneg_v2_top64", 0.00112, 0.02049, 0.00503, 0.00391),
            )
            self._write_json(
                tmp,
                "benchmark/compare_t5_head256_hardneg_v2_vs_cascade_top64.json",
                self._compare_payload("hardneg_v2_top64", 0.00021, 0.75262, 0.00503, 0.00482),
            )
            self._write_json(
                tmp,
                "benchmark/compare_cascade_10krecall_vs_hardneg_v2.json",
                self._compare_payload("cascade_10krecall_hardneg_top64", -0.00048, 0.36632, 0.00455, 0.00503),
            )
            self._write_json(
                tmp,
                "benchmark/cascade_error_analysis_head256_top64.json",
                {"aggregate": {"mean_cascade_gain": 0.00092, "win_record_fraction": 0.54688}},
            )

            report = build_paper_figure2_cascade_recall_precision.build_paper_figure2(tmp)
            self.assertEqual(report["artifact_kind"], "paper_figure2_cascade_recall_precision")
            self.assertTrue(report["summary"]["ready_for_cascade_figure_draft"])
            self.assertTrue(report["summary"]["recall_precision_roles_visible"])
            self.assertFalse(report["summary"]["cascade_te_significant_vs_sequential"])
            self.assertTrue(report["summary"]["hardneg_v2_significant_vs_sequential"])
            self.assertTrue(report["summary"]["hardneg_v2_direct_default"])
            self.assertFalse(report["summary"]["ready_for_cascade_positive_claim"])
            self.assertTrue(report["summary"]["hard_constraints_exact_1"])
            self.assertTrue(report["summary"]["source_files_ready"])
            self.assertEqual(report["diagnostics"]["cascade_vs_seq_p"], 0.09545)
            self.assertEqual(report["diagnostics"]["sourceaware_recall"], 0.75410)
            labels = {node["label"] for node in report["nodes"]}
            self.assertIn("Source-aware recall ranker", labels)
            self.assertIn("Precision reranker", labels)
            self.assertIn("Hard-negative v2 direct top64", labels)
            self.assertIn("non-significant TE trend", report["caption"])
            self.assertIn("flowchart LR", report["mermaid"])

            out_json = os.path.join(tmp, "figure2.json")
            out_md = os.path.join(tmp, "figure2.md")
            build_paper_figure2_cascade_recall_precision.write_report_json(report, out_json)
            build_paper_figure2_cascade_recall_precision.write_report_markdown(report, out_md)
            with open(out_json, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertFalse(loaded["summary"]["ready_for_cascade_positive_claim"])
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Paper Figure 2", text)
            self.assertIn("cascade TE significant vs sequential: `False`", text)
            self.assertIn("Hard-negative v2 direct default: `True`", text)


class TestPaperFigure3OracleGapClosure(unittest.TestCase):
    def _write_json(self, root, rel_path, payload):
        path = os.path.join(root, rel_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        return path

    def _proposal_payload(self, model_top, recall):
        return {
            "aggregate": {
                "mean_model_regret": 1.0 - model_top,
                "mean_model_top_te": model_top,
                "mean_oracle_top_te": 1.0,
                "mean_source_te": 0.5,
                "n_candidates": 100,
                "n_records": 10,
                "oracle_best_in_model_top_k_fraction": recall,
            }
        }

    def test_figure3_keeps_negative_and_incomplete_oracle_gap_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = {
                "benchmark/proposal_ranking_t5_base_full1k_head64.json": (0.4, 0.03),
                "benchmark/proposal_ranking_t5_ranker_full1k_final_head64.json": (0.55, 0.42),
                "benchmark/proposal_ranking_t5_utr_teacher_head64.json": (0.52, 0.70),
                "benchmark/proposal_ranking_t5_hybrid_teacher_head64.json": (0.53, 0.48),
                "benchmark/proposal_ranking_t5_full1k_then_utr_teacher_head64.json": (0.58, 0.41),
                "benchmark/proposal_ranking_t5_sourceaware_hybrid_teacher_head64.json": (0.54, 0.75),
                "benchmark/proposal_ranking_t5_cascade_hardneg_teacher_head64.json": (0.8, 0.74),
            }
            for rel_path, (model_top, recall) in values.items():
                self._write_json(tmp, rel_path, self._proposal_payload(model_top, recall))

            report = build_paper_figure3_oracle_gap_closure.build_paper_figure3(tmp)
            self.assertEqual(report["artifact_kind"], "paper_figure3_oracle_gap_closure")
            self.assertTrue(report["summary"]["ready_for_oracle_gap_figure_draft"])
            self.assertTrue(report["summary"]["source_oracle_consistent"])
            self.assertEqual(report["summary"]["best_run_id"], "cascade_hardneg_v2")
            self.assertAlmostEqual(report["summary"]["best_closure_fraction"], 0.6)
            self.assertTrue(report["summary"]["negative_closure_present"])
            self.assertFalse(report["summary"]["oracle_gap_fully_closed"])
            self.assertFalse(report["summary"]["ready_for_oracle_sota_claim"])
            self.assertFalse(report["summary"]["ready_for_wet_lab_claim"])
            base = next(point for point in report["points"] if point["run_id"] == "stage_a_base")
            self.assertLess(base["closure_fraction"], 0.0)
            self.assertEqual(len(report["chart_spec"]["data"]["values"]), 7)
            self.assertIn("Negative closure", report["caption"])

            out_json = os.path.join(tmp, "figure3.json")
            out_md = os.path.join(tmp, "figure3.md")
            build_paper_figure3_oracle_gap_closure.write_report_json(report, out_json)
            build_paper_figure3_oracle_gap_closure.write_report_markdown(report, out_md)
            with open(out_json, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertFalse(loaded["summary"]["ready_for_oracle_sota_claim"])
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Paper Figure 3", text)
            self.assertIn("negative closure present: `True`", text)
            self.assertIn("oracle gap fully closed: `False`", text)
            self.assertIn("Vega-Lite", text)


class TestStageAScaleLawSweepStatus(unittest.TestCase):
    def _write_json(self, path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

    def test_summarizes_queued_and_partial_sweep_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            sweep = os.path.join(tmp, "benchmark", "stage_a_scalelaw_test")
            log_root = os.path.join(tmp, "logs", "stage_a_scalelaw_test")
            ckpt_root = os.path.join(tmp, "ckpts", "stage_a_scalelaw_test")
            os.makedirs(sweep, exist_ok=True)
            runs = []
            for run_id, rows in (("data256_tiny_steps2_seed0", 2), ("data1024_small_steps4_seed0", 0)):
                profile_path = os.path.join(log_root, f"{run_id}.profile.jsonl")
                checkpoint_path = os.path.join(ckpt_root, run_id, "stage_a_best.pt")
                metadata_path = os.path.join(sweep, "metadata", f"{run_id}.json")
                log_path = os.path.join(log_root, f"{run_id}.train.log")
                os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
                self._write_json(metadata_path, {"status": "planned", "run_id": run_id})
                if rows:
                    os.makedirs(os.path.dirname(profile_path), exist_ok=True)
                    with open(profile_path, "w", encoding="utf-8") as fh:
                        for step in range(1, rows + 1):
                            fh.write(json.dumps({"step": step, "loss": 1.0 / step, "samples_per_s": 3.0}) + "\n")
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    with open(checkpoint_path, "w", encoding="utf-8") as fh:
                        fh.write("ckpt")
                runs.append(
                    {
                        "run_id": run_id,
                        "data_size": 256 if "256" in run_id else 1024,
                        "model_size": "tiny" if "tiny" in run_id else "small",
                        "steps": rows if rows else 4,
                        "seed": 0,
                        "profile_path": profile_path,
                        "checkpoint_path": checkpoint_path,
                        "metadata_path": metadata_path,
                        "log_path": log_path,
                    }
                )
            self._write_json(
                os.path.join(sweep, "plan.json"),
                {
                    "artifact_kind": "stage_a_scalelaw_sweep_plan",
                    "axes": {"data_sizes": [256, 1024], "model_sizes": ["tiny", "small"], "step_counts": [2, 4], "seeds": [0]},
                    "n_runs": 2,
                    "source_record_count": 10,
                    "source_records_sha256": "abc",
                    "runs": runs,
                },
            )
            with open(os.path.join(sweep, "progress.jsonl"), "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"event": "plan_ready"}) + "\n")
                fh.write(json.dumps({"event": "load_gate_wait", "loadavg": "101.0"}) + "\n")

            report = summarize_stage_a_scalelaw_sweep.summarize_sweep(sweep)
            self.assertEqual(report["artifact_kind"], "stage_a_scalelaw_sweep_status")
            self.assertEqual(report["summary"]["n_runs"], 2)
            self.assertEqual(report["summary"]["n_complete"], 1)
            self.assertEqual(report["summary"]["n_incomplete"], 1)
            self.assertEqual(report["summary"]["last_event"], "load_gate_wait")
            self.assertEqual(report["summary"]["last_loadavg"], "101.0")
            self.assertFalse(report["summary"]["ready_for_scale_law_claim"])
            statuses = {row["run_id"]: row["status"] for row in report["runs"]}
            self.assertEqual(statuses["data256_tiny_steps2_seed0"], "complete")
            self.assertEqual(statuses["data1024_small_steps4_seed0"], "queued")

            out_json = os.path.join(sweep, "status.json")
            out_md = os.path.join(sweep, "status.md")
            summarize_stage_a_scalelaw_sweep.write_report_json(report, out_json)
            summarize_stage_a_scalelaw_sweep.write_report_markdown(report, out_md)
            with open(out_json, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["summary"]["status_counts"]["complete"], 1)
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Stage A Scale-Law Sweep Status", text)
            self.assertIn("ready for scale-law claim: `False`", text)


class TestStageADownstreamEvalReadiness(unittest.TestCase):
    def _write_json(self, path, payload):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

    def _write_status(self, root, runs):
        sweep = os.path.join(root, "benchmark", "stage_a_scalelaw_test")
        os.makedirs(sweep, exist_ok=True)
        payload = {
            "artifact_kind": "stage_a_scalelaw_sweep_status",
            "summary": {
                "n_runs": len(runs),
                "n_complete": sum(1 for row in runs if row["status"] == "complete"),
                "n_incomplete": sum(1 for row in runs if row["status"] != "complete"),
                "ready_for_scale_law_claim": False,
            },
            "runs": runs,
        }
        self._write_json(os.path.join(sweep, "status.json"), payload)
        return sweep

    def test_downstream_readiness_blocks_on_incomplete_training(self):
        with tempfile.TemporaryDirectory(prefix="mef_stage_a_downstream_") as tmp:
            self._write_status(
                tmp,
                [
                    {
                        "run_id": "data256_tiny_steps200_seed0",
                        "data_size": 256,
                        "model_size": "tiny",
                        "steps": 200,
                        "seed": 0,
                        "status": "queued",
                        "checkpoint_exists": False,
                        "paths": {"checkpoint": os.path.join(tmp, "missing.pt")},
                    }
                ],
            )
            report = stage_a_downstream_eval_readiness.build_stage_a_downstream_eval_readiness(tmp)
            self.assertEqual(report["artifact_kind"], "stage_a_downstream_eval_readiness")
            self.assertEqual(report["summary"]["status"], "blocked_on_stage_a_sweep")
            self.assertEqual(report["summary"]["n_training_complete"], 0)
            self.assertFalse(report["summary"]["ready_for_stage_a_downstream_eval_claim"])
            self.assertFalse(report["summary"]["ready_for_true_scale_law_claim"])
            self.assertIn("stage_a_training_complete", report["summary"]["missing_or_incomplete"])
            self.assertEqual(report["rows"][0]["status"], "blocked_on_training")

    def test_downstream_readiness_requires_aggregate_and_keeps_true_claim_closed(self):
        with tempfile.TemporaryDirectory(prefix="mef_stage_a_downstream_ready_") as tmp:
            run_id = "data256_tiny_steps200_seed0"
            checkpoint = os.path.join(tmp, "ckpts", run_id, "stage_a_best.pt")
            os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
            with open(checkpoint, "w", encoding="utf-8") as fh:
                fh.write("ckpt")
            self._write_status(
                tmp,
                [
                    {
                        "run_id": run_id,
                        "data_size": 256,
                        "model_size": "tiny",
                        "steps": 200,
                        "seed": 0,
                        "status": "complete",
                        "checkpoint_exists": True,
                        "paths": {"checkpoint": checkpoint},
                    }
                ],
            )
            run_dir = os.path.join(tmp, "benchmark", "stage_a_scalelaw_downstream", run_id)
            self._write_json(os.path.join(run_dir, "proposal_ranking_t5.json"), {"ok": True})
            self._write_json(os.path.join(run_dir, "t1_t7_eval_summary.json"), {"ok": True})
            self._write_json(os.path.join(run_dir, "runtime_audit.json"), {"ok": True})
            self._write_json(
                os.path.join(tmp, "docs", "stage_a_scalelaw_downstream_eval_summary.json"),
                {"ok": True},
            )

            report = stage_a_downstream_eval_readiness.build_stage_a_downstream_eval_readiness(tmp)
            self.assertEqual(report["summary"]["status"], "trend_audit_missing")
            self.assertEqual(report["summary"]["n_training_complete"], 1)
            self.assertEqual(report["summary"]["n_downstream_ready"], 1)
            self.assertTrue(report["summary"]["ready_for_stage_a_downstream_eval_claim"])
            self.assertFalse(report["summary"]["ready_for_true_scale_law_claim"])
            self.assertIn("scale_law_trend_audit", report["summary"]["missing_or_incomplete"])

            out_json = os.path.join(tmp, "report.json")
            out_md = os.path.join(tmp, "report.md")
            stage_a_downstream_eval_readiness.write_report_json(report, out_json)
            stage_a_downstream_eval_readiness.write_report_markdown(report, out_md)
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Stage A Downstream Evaluation Readiness", text)
            self.assertIn("ready for true scale-law claim: `False`", text)


class TestMetrics(unittest.TestCase):
    def test_edit_distance_keeps_true_levenshtein_semantics(self):
        self.assertEqual(metrics.edit_distance("ACGU", "CGUA"), 2)
        self.assertEqual(metrics.edit_distance("ACGU", "ACGA"), 1)

    def test_legality_kozak_accessibility_and_mfe_metrics(self):
        records = _candidate_records()
        legal = metrics.legality_metrics(records)
        self.assertEqual(legal["valid_cds_fraction"], 1.0)
        self.assertEqual(legal["frame_intact_fraction"], 1.0)

        malformed = records + [MRNARecord("bad", "NN", "AUGAAU", "")]
        legal2 = metrics.legality_metrics(malformed)
        self.assertLess(legal2["legal_fraction"], 1.0)

        kozak = metrics.kozak_uaug_stats(records)
        self.assertGreaterEqual(kozak["mean_kozak_score"], 0.0)
        self.assertLessEqual(kozak["mean_kozak_score"], 1.0)
        self.assertEqual(kozak["mean_uaug_count"], 0.0)

        struct = metrics.start_accessibility_mfe_metrics(records)
        self.assertTrue(math.isfinite(struct["mean_start_accessibility"]))
        self.assertTrue(math.isfinite(struct["mean_mfe_proxy"]))
        self.assertGreaterEqual(struct["mean_start_accessibility"], 0.0)
        self.assertLessEqual(struct["mean_start_accessibility"], 1.0)

    def test_distribution_metrics_are_finite(self):
        cands = _candidate_records()
        srcs = _source_records()
        self.assertGreaterEqual(metrics.kmer_js_distance(cands, srcs, k=2), 0.0)
        self.assertGreaterEqual(metrics.codon_usage_kl(cands, srcs), 0.0)
        gc_len = metrics.gc_length_distribution_distance(cands, srcs)
        self.assertTrue(math.isfinite(gc_len["combined_gc_length_distance"]))
        self.assertGreaterEqual(metrics.embedding_frechet_proxy(cands, srcs), 0.0)

    def test_diversity_novelty_and_cai(self):
        cands = _candidate_records()
        srcs = _source_records()
        div = metrics.diversity_novelty_metrics(cands, srcs)
        self.assertGreater(div["unique_fraction"], 0.0)
        self.assertGreaterEqual(div["pairwise_diversity"], 0.0)
        self.assertGreater(div["mean_novelty"], 0.0)
        self.assertTrue(div["pairwise_diversity_exact"])
        self.assertEqual(div["pairwise_pairs_total"], 1)
        self.assertEqual(div["pairwise_pairs_evaluated"], 1)

        cai = metrics.cai_metrics(cands)
        self.assertGreater(cai["mean_cai"], 0.0)
        self.assertLessEqual(cai["mean_cai"], 1.0)

    def test_diversity_uses_deterministic_pair_subsampling_at_scale(self):
        records = [
            MRNARecord(f"r{i}", "A" * (20 + i), "AUGAAAUAA", "C" * (10 + (i % 3)))
            for i in range(24)
        ]
        div = metrics.diversity_novelty_metrics(records, sources=records, max_pairwise_pairs=17)
        self.assertFalse(div["pairwise_diversity_exact"])
        self.assertEqual(div["pairwise_pairs_total"], 276)
        self.assertEqual(div["pairwise_pairs_evaluated"], 17)
        self.assertEqual(div["mean_novelty"], 0.0)
        self.assertEqual(div["exact_source_match_fraction"], 1.0)
        self.assertEqual(div["novelty_source_comparisons"], 0)
        self.assertTrue(div["novelty_exact"])

    def test_diversity_can_cap_novelty_source_search(self):
        exact = metrics.diversity_novelty_metrics(
            ["AAAA"],
            sources=["UUUU", "AAAAC"],
            max_novelty_sources=0,
        )
        capped = metrics.diversity_novelty_metrics(
            ["AAAA"],
            sources=["UUUU", "AAAAC"],
            max_novelty_sources=1,
        )
        self.assertTrue(exact["novelty_exact"])
        self.assertFalse(capped["novelty_exact"])
        self.assertEqual(capped["novelty_source_comparisons"], 1)
        self.assertGreater(capped["mean_novelty"], exact["mean_novelty"])

    def test_exact_novelty_warm_starts_from_paired_source(self):
        candidates = [
            "A" * 100 + "C" + "G" * 100,
            "A" * 200 + "C" + "G" * 200,
            "A" * 300 + "C" + "G" * 300,
        ]
        paired_sources = [
            "A" * 100 + "U" + "G" * 100,
            "A" * 200 + "U" + "G" * 200,
            "A" * 300 + "U" + "G" * 300,
        ]
        exact = metrics.diversity_novelty_metrics(
            candidates,
            sources=paired_sources,
            max_novelty_sources=0,
        )
        self.assertTrue(exact["novelty_exact"])
        expected = sum(1.0 / len(seq) for seq in candidates) / len(candidates)
        self.assertAlmostEqual(exact["mean_novelty"], expected)
        self.assertEqual(exact["novelty_source_comparisons"], len(candidates))

    def test_t4_protein_identity_metric(self):
        cands = _candidate_records()
        srcs = _source_records()
        ident = metrics.protein_identity_metrics(cands, srcs)
        self.assertEqual(ident["mean_protein_identity"], 1.0)
        bad = MRNARecord("bad", "GCCACC", "AUGGACAAAUAA", "GGAAUAAACU")
        self.assertLess(metrics.protein_identity(bad, srcs[0]), 1.0)

    def test_t5_edit_budget_metric(self):
        cands = _candidate_records()
        srcs = _source_records()
        loose = metrics.edit_budget_metrics(cands, srcs, max_edits=3)
        self.assertEqual(loose["within_budget_fraction"], 1.0)
        tight = metrics.edit_budget_metrics(cands, srcs, max_edits=1)
        self.assertEqual(tight["within_budget_fraction"], 0.5)
        self.assertEqual(tight["over_budget_count"], 1)

    def test_t6_length_control_curve_metric(self):
        recs = [
            {
                "transcript_id": "a",
                "five_utr": "AAA",
                "cds": "AUGAAAUAA",
                "three_utr": "CCC",
                "target_length": 15,
            },
            {
                "transcript_id": "b",
                "five_utr": "AAAA",
                "cds": "AUGAAAUAA",
                "three_utr": "CCCC",
                "target_length": 19,
            },
        ]
        curve = metrics.length_control_curve(recs)
        self.assertEqual(curve["n"], 2)
        self.assertEqual(curve["mean_abs_length_error"], 1.0)
        self.assertGreaterEqual(len(curve["curve"]), 1)

    def test_t7_motif_and_reading_frame_metrics(self):
        motif = metrics.motif_metrics(_candidate_records())
        self.assertEqual(motif["polyA_signal_presence_fraction"], 1.0)
        self.assertEqual(motif["reading_frame_intact_fraction"], 1.0)
        self.assertEqual(motif["uAUG_presence_fraction"], 0.0)
        detected = metrics.detect_motifs(_candidate_records()[0])
        self.assertGreater(detected["polyA_signal"], 0)


class TestRunEval(unittest.TestCase):
    def test_bootstrap_ci_is_finite(self):
        ci = run_eval.bootstrap_ci([0.1, 0.2, 0.3, 0.4], seeds=[1, 2, 3, 4, 5], n_bootstrap=20)
        self.assertTrue(math.isfinite(ci["mean"]))
        self.assertTrue(math.isfinite(ci["low"]))
        self.assertTrue(math.isfinite(ci["high"]))
        self.assertLessEqual(ci["low"], ci["high"])

    def test_run_eval_outputs_paper_table_and_json(self):
        cands = _candidate_records()
        srcs = _source_records()
        with tempfile.TemporaryDirectory(prefix="mef_eval_test_") as tmp:
            summary = run_eval.run_evaluation(
                cands,
                sources=srcs,
                task_id="T5",
                out_dir=tmp,
                seeds=[10, 11, 12, 13, 14],
                n_bootstrap=20,
                max_edits=3,
                target_lengths=[len(r.seq) for r in cands],
            )
            table = os.path.join(tmp, "paper_table.md")
            data = os.path.join(tmp, "eval_summary.json")
            self.assertTrue(os.path.exists(table))
            self.assertTrue(os.path.exists(data))
            self.assertEqual(summary["task_metrics"]["T5"]["within_budget_fraction"], 1.0)
            self.assertEqual(summary["task_metrics"]["T4"]["mean_protein_identity"], 1.0)
            self.assertEqual(summary["task_metrics"]["T7"]["reading_frame_intact_fraction"], 1.0)
            self.assertEqual(summary["eval_config"]["max_pairwise_pairs"], 64)
            with open(table, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("mRNA-EditFlow Benchmark Summary", text)
            self.assertIn("T5", text)
            with open(data, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self.assertIn("bootstrap_ci", payload)
            self.assertIn("paired_significance", payload)

    def test_multiseed_benchmark_writes_aggregate_table(self):
        with tempfile.TemporaryDirectory(prefix="mef_multiseed_test_") as tmp:
            result = run_multiseed_benchmark.run_multiseed_benchmark(
                _source_records(),
                checkpoint_path=None,
                out_dir=tmp,
                task_id="T5",
                seeds=[0, 1, 2, 3, 4],
                limit=2,
                edit_budget=2,
                guidance_scale=1.0,
                target_start_accessibility=0.6,
                n_bootstrap=20,
                max_pairwise_pairs=8,
            )
            self.assertTrue(os.path.exists(result["json_path"]))
            self.assertTrue(os.path.exists(result["table_path"]))
            self.assertIn("within_budget_fraction", result["aggregate"])
            self.assertIn("delta_oracle_te_vs_source", result["aggregate"])
            self.assertEqual(result["config"]["guidance_scale"], 1.0)
            self.assertEqual(result["config"]["target_start_accessibility"], 0.6)
            self.assertEqual(result["config"]["max_novelty_sources"], 0)
            self.assertFalse(result["config"]["resume"])
            self.assertEqual(result["config"]["decoder_seeds"], [0, 1, 2, 3, 4])
            self.assertFalse(result["seed_semantics"]["decoder_seed_is_independent_training_run"])
            self.assertFalse(result["seed_semantics"]["paper_statistical_claim_eligible"])
            self.assertTrue(os.path.exists(result["progress_jsonl"]))
            self.assertEqual(result["aggregate"]["within_budget_fraction"]["n"], 5)
            with open(result["table_path"], "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("mRNA-EditFlow Multi-Seed Benchmark", text)
            self.assertIn("mean_oracle_te", text)

            resumed = run_multiseed_benchmark.run_multiseed_benchmark(
                _source_records(),
                checkpoint_path=None,
                out_dir=tmp,
                task_id="T5",
                seeds=[0, 1, 2, 3, 4],
                limit=2,
                edit_budget=2,
                guidance_scale=1.0,
                target_start_accessibility=0.6,
                n_bootstrap=20,
                max_pairwise_pairs=8,
                resume=True,
            )
            self.assertTrue(resumed["config"]["resume"])
            self.assertTrue(all(row["resumed"] for row in resumed["per_seed"]))
            self.assertEqual(
                resumed["aggregate"]["delta_oracle_te_vs_source"]["mean"],
                result["aggregate"]["delta_oracle_te_vs_source"]["mean"],
            )
            with open(resumed["progress_jsonl"], "r", encoding="utf-8") as fh:
                events = [json.loads(line)["event"] for line in fh if line.strip()]
            self.assertGreaterEqual(events.count("seed_resumed"), 5)
            with self.assertRaisesRegex(ValueError, "stale resume"):
                run_multiseed_benchmark.run_multiseed_benchmark(
                    _source_records(),
                    checkpoint_path=None,
                    out_dir=tmp,
                    task_id="T5",
                    seeds=[0, 1, 2, 3, 4],
                    limit=2,
                    edit_budget=3,
                    guidance_scale=1.0,
                    target_start_accessibility=0.6,
                    n_bootstrap=20,
                    max_pairwise_pairs=8,
                    resume=True,
                )
            self.assertIn("benchmark_complete", events)

    def test_compare_benchmarks_paired_delta_table(self):
        with tempfile.TemporaryDirectory(prefix="mef_compare_test_") as tmp:
            def write_summary(name, values):
                path = os.path.join(tmp, f"{name}.json")
                payload = {
                    "config": {"name": name},
                    "aggregate": {
                        "mean_oracle_te": {
                            "mean": sum(values) / len(values),
                            "std": 0.0,
                            "low": min(values),
                            "high": max(values),
                            "n": len(values),
                        }
                    },
                    "per_seed": [
                        {"seed": i, "metrics": {"mean_oracle_te": float(v)}}
                        for i, v in enumerate(values)
                    ],
                }
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh)
                return path

            base = compare_benchmarks.load_benchmark_run("base", write_summary("base", [1, 1, 1, 1, 1]))
            run = compare_benchmarks.load_benchmark_run("better", write_summary("better", [2, 2, 2, 2, 2]))
            result = compare_benchmarks.compare_benchmarks(
                base,
                [run],
                metrics=["mean_oracle_te"],
                n_bootstrap=20,
                n_permutations=20,
            )
            row = result["rows"][0]
            self.assertEqual(row["delta"], 1.0)
            self.assertEqual(row["improvement"], 1.0)
            self.assertEqual(row["n_paired_seeds"], 5)
            md = os.path.join(tmp, "compare.md")
            compare_benchmarks.write_comparison_table(result, md)
            with open(md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Benchmark Comparison", text)
            self.assertIn("mean_oracle_te", text)

    def test_paper_comparison_uses_training_seed_as_inference_unit(self):
        with tempfile.TemporaryDirectory(prefix="mef_compare_nested_") as tmp:
            def write_nested(name, training_seeds, offset):
                path = os.path.join(tmp, f"{name}.json")
                rows = []
                for training_seed in training_seeds:
                    for decoder_seed in (0, 1):
                        rows.append({
                            "training_seed": training_seed,
                            "decoder_seed": decoder_seed,
                            "metrics": {
                                "mean_oracle_te": float(training_seed + decoder_seed + offset)
                            },
                        })
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump({
                        "config": {"run_mode": "paper"},
                        "aggregate": {},
                        "per_seed": rows,
                    }, fh)
                return path

            baseline = compare_benchmarks.load_benchmark_run(
                "base", write_nested("base", [0, 1, 2], 0.0)
            )
            candidate = compare_benchmarks.load_benchmark_run(
                "candidate", write_nested("candidate", [0, 1, 2], 1.0)
            )
            rows = compare_benchmarks.compare_run_to_baseline(
                baseline,
                candidate,
                ["mean_oracle_te"],
                n_bootstrap=20,
                n_permutations=20,
            )
            self.assertEqual(rows[0]["n_paired_seeds"], 3)
            self.assertEqual(
                rows[0]["inference_unit"], "training_seed_with_nested_decoder_means"
            )
            self.assertTrue(rows[0]["paper_significance_eligible"])

            insufficient = compare_benchmarks.load_benchmark_run(
                "insufficient", write_nested("insufficient", [1, 2], 1.0)
            )
            with self.assertRaisesRegex(ValueError, "three matched independent training seeds"):
                compare_benchmarks.compare_run_to_baseline(
                    baseline,
                    insufficient,
                    ["mean_oracle_te"],
                    n_bootstrap=20,
                    n_permutations=20,
                )

    def test_compare_benchmarks_can_require_matching_config(self):
        with tempfile.TemporaryDirectory(prefix="mef_compare_cfg_test_") as tmp:
            def write_summary(name, top_k):
                path = os.path.join(tmp, f"{name}.json")
                payload = {
                    "config": {
                        "task_id": "T5",
                        "edit_budget": 3,
                        "proposal_top_k": top_k,
                        "n_records": 2,
                        "seeds": [0, 1, 2, 3, 4],
                    },
                    "aggregate": {"mean_oracle_te": {"mean": 1.0, "n": 5}},
                    "per_seed": [
                        {"seed": seed, "metrics": {"mean_oracle_te": 1.0 + 0.1 * seed}}
                        for seed in range(5)
                    ],
                }
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh)
                return path

            fields = ["task_id", "edit_budget", "proposal_top_k", "n_records", "seeds"]
            base = compare_benchmarks.load_benchmark_run("base", write_summary("base", 32))
            matched = compare_benchmarks.load_benchmark_run("matched", write_summary("matched", 32))
            result = compare_benchmarks.compare_benchmarks(
                base,
                [matched],
                metrics=["mean_oracle_te"],
                n_bootstrap=20,
                n_permutations=20,
                require_matching_config=fields,
            )
            self.assertEqual(result["required_matching_config"], fields)
            self.assertTrue(all(row["matches"] for row in result["config_checks"]))

            mismatched = compare_benchmarks.load_benchmark_run("mismatched", write_summary("mismatched", 8))
            with self.assertRaisesRegex(ValueError, "proposal_top_k"):
                compare_benchmarks.compare_benchmarks(
                    base,
                    [mismatched],
                    metrics=["mean_oracle_te"],
                    n_bootstrap=20,
                    n_permutations=20,
                    require_matching_config=fields,
                )

    def test_compare_benchmarks_uses_effective_top_k_for_cascade_guard(self):
        with tempfile.TemporaryDirectory(prefix="mef_compare_cascade_cfg_test_") as tmp:
            def write_summary(name, config):
                path = os.path.join(tmp, f"{name}.json")
                base_config = {
                    "task_id": "T5",
                    "edit_budget": 3,
                    "n_records": 2,
                    "seeds": [0, 1, 2, 3, 4],
                    "target_length_delta": 0,
                    "proposal_temperature": 1.0,
                    "guidance_scale": 0.0,
                    "target_te": None,
                    "target_start_accessibility": None,
                    "max_pairwise_pairs": 16,
                    "max_novelty_sources": 0,
                }
                base_config.update(config)
                payload = {
                    "config": base_config,
                    "aggregate": {"mean_oracle_te": {"mean": 1.0, "n": 5}},
                    "per_seed": [
                        {"seed": seed, "metrics": {"mean_oracle_te": 1.0 + 0.1 * seed}}
                        for seed in range(5)
                    ],
                }
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh)
                return path

            base = compare_benchmarks.load_benchmark_run(
                "single",
                write_summary("single", {"checkpoint_path": "precision.pt", "proposal_top_k": 64}),
            )
            cascade = compare_benchmarks.load_benchmark_run(
                "cascade",
                write_summary(
                    "cascade",
                    {
                        "checkpoint_path": "precision.pt",
                        "proposal_top_k": 8,
                        "cascade_recall_checkpoint_path": "recall.pt",
                        "cascade_recall_top_k": 64,
                    },
                ),
            )
            result = compare_benchmarks.compare_benchmarks(
                base,
                [cascade],
                metrics=["mean_oracle_te"],
                n_bootstrap=20,
                n_permutations=20,
                require_matching_config=compare_benchmarks.DEFAULT_REQUIRED_MATCHING_CONFIG,
            )
            self.assertEqual(base.config["effective_proposal_top_k"], 64)
            self.assertEqual(cascade.config["effective_proposal_top_k"], 64)
            self.assertTrue(all(row["matches"] for row in result["config_checks"]))

            narrow_cascade = compare_benchmarks.load_benchmark_run(
                "cascade_k32",
                write_summary(
                    "cascade_k32",
                    {
                        "checkpoint_path": "precision.pt",
                        "proposal_top_k": 64,
                        "cascade_recall_checkpoint_path": "recall.pt",
                        "cascade_recall_top_k": 32,
                    },
                ),
            )
            with self.assertRaisesRegex(ValueError, "effective_proposal_top_k"):
                compare_benchmarks.compare_benchmarks(
                    base,
                    [narrow_cascade],
                    metrics=["mean_oracle_te"],
                    n_bootstrap=20,
                    n_permutations=20,
                    require_matching_config=compare_benchmarks.DEFAULT_REQUIRED_MATCHING_CONFIG,
                )

    def test_cascade_error_analysis_reports_per_record_win_loss(self):
        with tempfile.TemporaryDirectory(prefix="mef_cascade_error_test_") as tmp:
            baseline_dir = os.path.join(tmp, "baseline")
            cascade_dir = os.path.join(tmp, "cascade")
            sources = [
                MRNARecord("r1", "GCCACC", "AUGGCUAAAUAA", "AAUAAA"),
                MRNARecord("r2", "AUGCCC", "AUGCCCGGGUAA", "CCCCCC"),
            ]
            write_records_jsonl(sources, os.path.join(baseline_dir, "sources.jsonl"))
            write_records_jsonl(sources, os.path.join(cascade_dir, "sources.jsonl"))

            def write_seed(run_dir, seed, te_values):
                seed_dir = os.path.join(run_dir, f"seed_{seed:03d}")
                os.makedirs(seed_dir, exist_ok=True)
                payload = {
                    "per_record_metrics": {
                        "oracle_ensemble_te": te_values,
                        "source_oracle_ensemble_te": [0.5, 0.7],
                    }
                }
                with open(os.path.join(seed_dir, "eval_summary.json"), "w", encoding="utf-8") as fh:
                    json.dump(payload, fh)

            write_seed(baseline_dir, 0, [0.55, 0.75])
            write_seed(baseline_dir, 1, [0.56, 0.74])
            write_seed(cascade_dir, 0, [0.60, 0.72])
            write_seed(cascade_dir, 1, [0.59, 0.73])

            out_json = os.path.join(tmp, "analysis.json")
            out_jsonl = os.path.join(tmp, "records.jsonl")
            out_md = os.path.join(tmp, "analysis.md")
            result = cascade_error_analysis.run_cascade_error_analysis(
                baseline_dir=baseline_dir,
                cascade_dir=cascade_dir,
                out_json=out_json,
                out_jsonl=out_jsonl,
                out_md=out_md,
                baseline_label="seq",
                cascade_label="cascade",
                top_n=1,
            )
            self.assertTrue(os.path.exists(out_json))
            self.assertTrue(os.path.exists(out_jsonl))
            self.assertTrue(os.path.exists(out_md))
            self.assertEqual(result["config"]["common_seeds"], [0, 1])
            self.assertAlmostEqual(result["aggregate"]["mean_cascade_gain"], 0.01)
            self.assertAlmostEqual(result["aggregate"]["win_record_fraction"], 0.5)
            self.assertIn("groups", result)
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Cascade Error Analysis", text)
            self.assertIn("Top Cascade Wins", text)

    def test_summarise_proposal_jsonl_recovers_ranking_gap(self):
        with tempfile.TemporaryDirectory(prefix="mef_prop_summary_") as tmp:
            path = os.path.join(tmp, "candidates.jsonl")
            rows = [
                {
                    "transcript_id": "r1",
                    "model_rank": 1,
                    "oracle_rank": 2,
                    "source_te": 0.5,
                    "oracle_te": 0.55,
                },
                {
                    "transcript_id": "r1",
                    "model_rank": 2,
                    "oracle_rank": 1,
                    "source_te": 0.5,
                    "oracle_te": 0.70,
                },
            ]
            with open(path, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
            out_json = os.path.join(tmp, "summary.json")
            summary = summarize_proposal_ranking.summarise_proposal_jsonl(
                path,
                out_json=out_json,
                top_k=1,
            )
            self.assertTrue(os.path.exists(out_json))
            self.assertEqual(summary["aggregate"]["n_records"], 1)
            self.assertEqual(summary["aggregate"]["n_candidates"], 2)
            self.assertAlmostEqual(summary["aggregate"]["mean_model_regret"], 0.15)
            self.assertEqual(summary["aggregate"]["oracle_best_in_model_top_k_fraction"], 0.0)

    def test_sota_gap_report_merges_measured_and_external_evidence(self):
        with tempfile.TemporaryDirectory(prefix="mef_sota_gap_") as tmp:
            bench = os.path.join(tmp, "benchmark")
            docs = os.path.join(tmp, "docs")
            os.makedirs(bench, exist_ok=True)
            os.makedirs(docs, exist_ok=True)

            def write_comparison(path, run_label, run_mean, baseline_mean, n_records):
                payload = {
                    "baseline": {"config": {"n_records": n_records}},
                    "rows": [
                        {
                            "run": run_label,
                            "metric": "delta_oracle_te_vs_source",
                            "baseline_mean": baseline_mean,
                            "run_mean": run_mean,
                            "delta": run_mean - baseline_mean,
                            "ci_low": 0.01,
                            "ci_high": 0.03,
                            "paired_p": 0.01,
                            "n_paired_seeds": 5,
                        }
                    ],
                }
                with open(os.path.join(bench, path), "w", encoding="utf-8") as fh:
                    json.dump(payload, fh)

            write_comparison("t5_ranker_comparison.json", "ranker_full1k_top32", 0.02, -0.01, 32)
            write_comparison("t5_ranker_full1k_head256_comparison.json", "ranker_full1k_top32", 0.03, -0.01, 256)
            write_comparison("t5_guidance_comparison.json", "all_proposal_te_guided", 0.10, -0.01, 32)
            write_comparison("compare_mo_fusion_vs_te_only_head1024.json", "mo_grpo_top64", 0.025, 0.020, 1024)
            write_comparison(
                "region_adapter_vs_hardneg_v2_top64_head256.json",
                "region_adapter_all_top64",
                0.04,
                0.02,
                256,
            )
            write_comparison(
                "region_adapter_vs_mo_grpo_top64_head256.json",
                "region_adapter_all_top64",
                0.035,
                0.030,
                256,
            )
            with open(os.path.join(bench, "proposal_ranking_t5_base_full1k_head64.json"), "w", encoding="utf-8") as fh:
                json.dump({"aggregate": {"mean_model_regret": 0.05, "oracle_best_in_model_top_k_fraction": 0.10}}, fh)
            with open(os.path.join(bench, "proposal_ranking_t5_ranker_full1k_head64.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "aggregate": {
                            "mean_model_regret": 0.03,
                            "oracle_best_in_model_top_k_fraction": 0.40,
                            "n_records": 64,
                        }
                    },
                    fh,
                )
            with open(os.path.join(bench, "proposal_ranking_t5_utr_teacher_head64.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "aggregate": {
                            "mean_model_regret": 0.025,
                            "oracle_best_in_model_top_k_fraction": 0.70,
                            "n_records": 64,
                        }
                    },
                    fh,
                )
            with open(os.path.join(bench, "proposal_ranking_t5_hybrid_teacher_head64.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "aggregate": {
                            "mean_model_regret": 0.024,
                            "oracle_best_in_model_top_k_fraction": 0.60,
                            "n_records": 64,
                        }
                    },
                    fh,
                )
            with open(os.path.join(bench, "proposal_ranking_t5_full1k_then_utr_teacher_head64.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "aggregate": {
                            "mean_model_regret": 0.023,
                            "oracle_best_in_model_top_k_fraction": 0.50,
                            "n_records": 64,
                        }
                    },
                    fh,
                )
            with open(os.path.join(bench, "proposal_ranking_t5_sourceaware_hybrid_teacher_head64.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "aggregate": {
                            "mean_model_regret": 0.022,
                            "oracle_best_in_model_top_k_fraction": 0.75,
                            "n_records": 64,
                        }
                    },
                    fh,
                )
            with open(os.path.join(bench, "cascade_sourceaware_to_sequential_head64.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "aggregate": {
                            "mean_cascade_regret": 0.021,
                            "mean_recall_model_regret": 0.030,
                            "oracle_best_in_recall_top_k_fraction": 0.76,
                            "n_records": 64,
                        }
                    },
                    fh,
                )
            with open(os.path.join(bench, "cascade_sourceaware_to_sequential_head64_k64.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "aggregate": {
                            "mean_cascade_regret": 0.020,
                            "mean_precision_full_regret": 0.021,
                            "oracle_best_in_recall_top_k_fraction": 0.81,
                            "n_records": 64,
                        }
                    },
                    fh,
                )
            with open(os.path.join(bench, "proposal_ranking_t5_stage_a10k_head1024.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "aggregate": {
                            "mean_model_regret": 0.019,
                            "oracle_best_in_model_top_k_fraction": 0.82,
                            "n_records": 1024,
                        }
                    },
                    fh,
                )
            with open(os.path.join(bench, "proposal_ranking_t5_base_stage_a10k_head64.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "aggregate": {
                            "mean_model_regret": 0.026,
                            "oracle_best_in_model_top_k_fraction": 0.71,
                            "n_records": 64,
                        }
                    },
                    fh,
                )
            with open(os.path.join(bench, "proposal_ranking_t5_ranker_stage_a10k_head64.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "aggregate": {
                            "mean_model_regret": 0.018,
                            "oracle_best_in_model_top_k_fraction": 0.86,
                            "n_records": 64,
                        }
                    },
                    fh,
                )
            with open(os.path.join(bench, "codon_lattice_dp_head256.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "summary": {
                            "n": 3,
                            "mean_source_cai": 0.50,
                            "mean_optimized_cai": 0.70,
                            "mean_source_gc": 0.40,
                            "mean_optimized_gc": 0.55,
                        }
                    },
                    fh,
                )
            with open(os.path.join(bench, "leakage_ranker_head256_vs_gencode.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "summary": {
                            "n_query": 3,
                            "flagged_fraction": 1.0,
                            "exact_match_count": 3,
                        }
                    },
                    fh,
                )
            with open(os.path.join(bench, "utr_local_search_head256.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "summary": {
                            "n": 3,
                            "mean_source_te": 0.55,
                            "mean_optimized_te": 0.61,
                        }
                    },
                    fh,
                )
            with open(os.path.join(bench, "utr_teacher_head256.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "summary": {
                            "n_records": 3,
                            "mean_source_te": 0.55,
                            "mean_best_candidate_te": 0.59,
                        }
                    },
                    fh,
                )
            dry_run_dir = os.path.join(bench, "external_sota", "dry_run_t5_head1024")
            os.makedirs(dry_run_dir, exist_ok=True)
            with open(os.path.join(dry_run_dir, "summary.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "status": "dry_run_complete",
                        "task_id": "T5",
                        "n_models": 1,
                        "n_executable_ready": 0,
                        "dataset": {
                            "sha256": "abc123",
                            "split_name": "unit_split",
                            "seed": 7,
                            "record_count_effective": 2,
                            "record_count_total": 3,
                        },
                        "hardware": {
                            "label": "unit-cpu",
                            "hostname": "unit-host",
                            "machine": "arm64",
                        },
                        "rows": [
                            {
                                "model_name": "LinearDesign",
                                "status": "not_configured",
                                "command_candidates": ["LINEARDESIGN_BIN", "lineardesign"],
                                "candidate_audit": [
                                    {"candidate": "LINEARDESIGN_BIN", "status": "env_unset"},
                                    {"candidate": "lineardesign", "status": "path_not_found"},
                                ],
                                "notes": "Install LinearDesign before claiming real metrics.",
                            }
                        ],
                    },
                    fh,
                )
            with open(os.path.join(docs, "mrna_dataset_survey.md"), "w", encoding="utf-8") as fh:
                fh.write(
                    "# mRNA Dataset Survey\n\n"
                    "GEMORNA uses region-specific mRNA data.\n"
                    "mRNA-GPT uses 10M full-length mRNAs.\n"
                    "ProMORNA uses 6M protein-mRNA pairs.\n"
                    "RNAGenScape uses 3 real mRNA datasets.\n"
                    "codonGPT uses Ensembl CDS data.\n"
                    "Every protocol needs leakage-free split, license, and public split audit.\n"
                )
            with open(os.path.join(docs, "sota_readiness_audit_head256.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "artifact_kind": "sota_readiness_audit",
                        "summary": {
                            "all_ready_for_sota_claim_audit": False,
                            "positive_sota_claim_ready": False,
                            "ready_for_internal_proxy_constrained_optimization_claim": True,
                            "ready_for_external_sota_metric_claim": False,
                            "ready_for_full_de_novo_claim": False,
                            "ready_for_real_te_or_stability_claim": False,
                            "ready_for_true_scale_law_claim": False,
                            "ready_for_wet_lab_claim": False,
                            "positive_sota_block_reasons": [
                                "external_sota_real_metrics_missing",
                                "full_de_novo_evidence_missing_or_overclaim_flagged",
                            ],
                            "allowed_claim_scope": (
                                "Constrained local full-length mRNA optimization/reranking "
                                "with proxy/offline T1-T7 evidence."
                            ),
                            "n_sections_ready": 1,
                            "n_sections_expected": 3,
                            "pending_sections": [
                                "region_adapter",
                                "protein_conditioned_gc_sweep",
                            ],
                            "claim_policy": "Do not turn readiness into a positive SOTA claim.",
                        },
                        "sections": {
                            "region_adapter": {
                                "ready": False,
                                "audit": {
                                    "summary": {
                                        "n_compare_files_found": 0,
                                        "n_compare_files_expected": 5,
                                        "all_constraints_exact_1": False,
                                    }
                                },
                            },
                            "protein_conditioned_gc_sweep": {
                                "ready": False,
                                "audit": {
                                    "summary": {
                                        "n_points": 0,
                                        "all_points_identity_exact_1": False,
                                        "pareto_metadata_ok": False,
                                    }
                                },
                            },
                            "external_sota_protocol": {
                                "ready": True,
                                "audit": {
                                    "summary": {
                                        "protocol_ready": True,
                                        "n_models": 4,
                                        "n_executable_ready": 0,
                                    }
                                },
                            },
                        },
                    },
                    fh,
                )

            report = sota_gap_report.build_sota_gap_report(tmp)
            self.assertEqual(report["oracle_gap_health_check"]["status"], "measured")
            self.assertAlmostEqual(report["oracle_gap_health_check"]["remaining_gap"], 0.07)
            self.assertEqual(report["external_sota_dry_run"]["status"], "dry_run_complete")
            self.assertEqual(report["external_sota_dry_run"]["n_executable_ready"], 0)
            self.assertEqual(report["dataset_survey_audit"]["status"], "ready")
            self.assertTrue(report["dataset_survey_audit"]["protocol_ready"])
            self.assertEqual(report["dataset_survey_audit"]["missing_methods"], [])
            self.assertIn("mRNA-GPT", report["dataset_survey_audit"]["covered_methods"])
            self.assertIn("sha256", report["dataset_survey_audit"])
            self.assertEqual(report["sota_readiness_audit"]["summary"]["n_sections_ready"], 1)
            self.assertEqual(
                report["sota_readiness_audit"]["summary"]["pending_sections"],
                ["region_adapter", "protein_conditioned_gc_sweep"],
            )
            self.assertGreaterEqual(len(report["measured_evidence"]), 4)
            self.assertIn(
                "CDS codon-lattice DP CAI",
                {row["name"] for row in report["measured_evidence"]},
            )
            self.assertIn(
                "Foundation leakage audit head256 vs GENCODE",
                {row["name"] for row in report["measured_evidence"]},
            )
            self.assertIn(
                "UTR local-search TE baseline",
                {row["name"] for row in report["measured_evidence"]},
            )
            self.assertIn(
                "UTR one-step teacher headroom",
                {row["name"] for row in report["measured_evidence"]},
            )
            self.assertIn(
                "UTR-teacher head64 oracle-best-in-model-top32 fraction",
                {row["name"] for row in report["measured_evidence"]},
            )
            self.assertIn(
                "Hybrid-teacher head64 proposal-ranking regret",
                {row["name"] for row in report["measured_evidence"]},
            )
            self.assertIn(
                "Full-then-UTR head64 proposal-ranking regret",
                {row["name"] for row in report["measured_evidence"]},
            )
            self.assertIn(
                "Source-aware hybrid head64 oracle-best-in-model-top32 fraction",
                {row["name"] for row in report["measured_evidence"]},
            )
            self.assertIn(
                "Cascade source-aware->sequential head64 regret (k=64)",
                {row["name"] for row in report["measured_evidence"]},
            )
            self.assertIn(
                "Stage A 10k head1024 proposal-ranking regret",
                {row["name"] for row in report["measured_evidence"]},
            )
            self.assertIn(
                "Stage A 10k teacher-ranker head64 proposal-ranking regret",
                {row["name"] for row in report["measured_evidence"]},
            )
            self.assertIn(
                "Multi-objective grpo-fusion ranker top64 scale-up (vs single-TE control) (n=1024)",
                {row["name"] for row in report["measured_evidence"]},
            )
            self.assertIn(
                "Region-specialized all-region adapters top64 (vs hardneg_v2) (n=256)",
                {row["name"] for row in report["measured_evidence"]},
            )
            self.assertIn(
                "Region-specialized all-region adapters top64 (vs MO-GRPO) (n=256)",
                {row["name"] for row in report["measured_evidence"]},
            )
            self.assertFalse(any("head256" in gate.lower() for gate in report["next_gates"]))
            self.assertFalse(any("head1024 multi-objective" in gate.lower() for gate in report["next_gates"]))
            self.assertFalse(any("region-specialized adapter" in gate.lower() for gate in report["next_gates"]))
            self.assertFalse(any("run k-mer leakage audit" in gate.lower() for gate in report["next_gates"]))
            self.assertFalse(any("utr local-search baseline" in gate.lower() for gate in report["next_gates"]))
            self.assertFalse(any("one-step oracle teacher" in gate.lower() for gate in report["next_gates"]))
            methods = {row["method"]: row for row in report["sota_references"]}
            self.assertIn("CodonFM", methods)
            self.assertTrue(methods["CodonFM"]["registered_external_model"])
            # Newly-integrated 2025-2026 generative mRNA SOTA landscape must be
            # present with a consistent schema (and honestly not yet registered
            # as executable external models).
            expected_keys = {
                "method",
                "venue_year",
                "scope",
                "reported_signal",
                "accuracy_f1",
                "speed_scale",
                "mef_gap",
                "citation_url",
                "registered_external_model",
            }
            for name in ("GEMORNA", "mRNA-GPT", "ProMORNA", "RNAGenScape"):
                self.assertIn(name, methods)
                self.assertEqual(set(methods[name]), expected_keys)
                self.assertFalse(methods[name]["registered_external_model"])
                self.assertTrue(methods[name]["citation_url"].startswith("http"))

            out_json = os.path.join(docs, "sota_gap_report.json")
            out_md = os.path.join(docs, "sota_gap_report.md")
            sota_gap_report.write_report_json(report, out_json)
            sota_gap_report.write_report_markdown(report, out_md)
            self.assertTrue(os.path.exists(out_json))
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Measured MEF Evidence", text)
            self.assertIn("CodonFM", text)
            self.assertIn("External SOTA Dry-Run Readiness", text)
            self.assertIn("Dataset Survey / Leakage Alignment", text)
            self.assertIn("protocol ready: `True`", text)
            self.assertIn("mRNA-GPT", text)
            self.assertIn("external SOTA reproduction", text)
            self.assertIn("Unified SOTA Readiness", text)
            self.assertIn("Positive SOTA claim ready: `False`", text)
            self.assertIn("Full de novo claim ready: `False`", text)
            self.assertIn("External SOTA metric claim ready: `False`", text)
            self.assertIn("Sections ready: `1` / `3`", text)
            self.assertIn("external_sota_protocol", text)
            self.assertIn("LinearDesign", text)
            self.assertIn("Dataset audit", text)
            self.assertIn("Hardware audit", text)
            self.assertIn("Candidate audit: LINEARDESIGN_BIN=env_unset", text)
            for name in ("GEMORNA", "mRNA-GPT", "ProMORNA", "RNAGenScape"):
                self.assertIn(name, text)


class TestMultiObjectiveDecodedPropertyAnalyzer(unittest.TestCase):
    def _write_mode(self, bench_root, mode, uaug_by_seed, agg_delta, slice_name="head256"):
        d = os.path.join(bench_root, f"multiseed_t5_public_{slice_name}_mo_{mode}_top64")
        os.makedirs(d, exist_ok=True)
        # aggregate summary exposes only oracle TE/MRL.
        with open(os.path.join(d, "multiseed_summary.json"), "w", encoding="utf-8") as fh:
            json.dump({"aggregate": {"delta_oracle_te_vs_source": {"mean": agg_delta}}}, fh)
        # per-seed eval_summary carries the decoded-output non-TE properties.
        for seed, uaug in enumerate(uaug_by_seed):
            sd = os.path.join(d, f"seed_{seed:03d}")
            os.makedirs(sd, exist_ok=True)
            with open(os.path.join(sd, "eval_summary.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "metrics": {
                            "kozak_uaug": {"mean_uaug_count": uaug, "uaug_fraction": 0.3, "mean_kozak_score": 0.5},
                            "structure": {"mean_start_accessibility": 0.6},
                            "distribution": {"candidate_mean_gc": 0.61},
                            "cai": {"mean_cai": 0.67},
                        }
                    },
                    fh,
                )

    def test_analyzer_means_per_seed_decoded_properties_and_reads_aggregate_te(self):
        with tempfile.TemporaryDirectory(prefix="mef_mo_decoded_") as tmp:
            bench = os.path.join(tmp, "benchmark")
            os.makedirs(bench, exist_ok=True)
            # te_only mean uAUG = 0.40; scalar mean uAUG = 0.50 (higher).
            self._write_mode(bench, "te_only", [0.30, 0.50], agg_delta=0.00348)
            self._write_mode(bench, "scalar", [0.45, 0.55], agg_delta=0.00300)

            te = analyze_mo_fusion_decoded_properties.mode_values(bench, "te_only")
            sc = analyze_mo_fusion_decoded_properties.mode_values(bench, "scalar")

            # Aggregate TE is read straight from multiseed_summary.
            self.assertAlmostEqual(te["delta_oracle_te_vs_source"], 0.00348, places=6)
            self.assertAlmostEqual(sc["delta_oracle_te_vs_source"], 0.00300, places=6)
            # Per-seed decoded uAUG is meaned across the 2 seeds.
            self.assertAlmostEqual(te["mean_uaug_count"], 0.40, places=6)
            self.assertAlmostEqual(sc["mean_uaug_count"], 0.50, places=6)
            # Other decoded properties are surfaced (not dropped like the aggregate does).
            self.assertAlmostEqual(te["mean_start_accessibility"], 0.60, places=6)
            self.assertAlmostEqual(te["candidate_mean_gc"], 0.61, places=6)
            self.assertAlmostEqual(te["mean_cai"], 0.67, places=6)
            self.assertEqual(int(te["n_seeds"]), 2)

    def test_missing_mode_yields_nan_without_crashing(self):
        with tempfile.TemporaryDirectory(prefix="mef_mo_decoded_missing_") as tmp:
            bench = os.path.join(tmp, "benchmark")
            os.makedirs(bench, exist_ok=True)
            vals = analyze_mo_fusion_decoded_properties.mode_values(bench, "grpo")
            self.assertEqual(int(vals["n_seeds"]), 0)
            self.assertTrue(math.isnan(vals["mean_uaug_count"]))

    def test_slice_name_selects_scaled_slice_and_isolates_from_head256(self):
        # The analyzer must be slice-parameterized so head1024 scale-up reuses the
        # same code path, and a non-default slice must NOT read head256 dirs.
        with tempfile.TemporaryDirectory(prefix="mef_mo_decoded_slice_") as tmp:
            bench = os.path.join(tmp, "benchmark")
            os.makedirs(bench, exist_ok=True)
            # Only a head1024 dir exists; head256 is intentionally absent.
            self._write_mode(bench, "scalar", [0.10, 0.20], agg_delta=0.02000, slice_name="head1024")

            scaled = analyze_mo_fusion_decoded_properties.mode_values(
                bench, "scalar", slice_name="head1024"
            )
            self.assertAlmostEqual(scaled["delta_oracle_te_vs_source"], 0.02000, places=6)
            self.assertAlmostEqual(scaled["mean_uaug_count"], 0.15, places=6)
            self.assertEqual(int(scaled["n_seeds"]), 2)

            # Default slice (head256) must find nothing -> NaN / 0 seeds.
            default_slice = analyze_mo_fusion_decoded_properties.mode_values(bench, "scalar")
            self.assertEqual(int(default_slice["n_seeds"]), 0)
            self.assertTrue(math.isnan(default_slice["mean_uaug_count"]))

    def test_seed_vectors_preserve_per_seed_order_and_feed_paired_pvalue(self):
        # Per-seed decoded properties must be recoverable as ordered vectors so the
        # non-TE tradeoffs carry the same 10-seed paired significance as oracle TE.
        with tempfile.TemporaryDirectory(prefix="mef_mo_decoded_paired_") as tmp:
            bench = os.path.join(tmp, "benchmark")
            os.makedirs(bench, exist_ok=True)
            # Control uAUG per seed = [0.30, 0.50]; scalar strictly higher = [0.45, 0.55].
            self._write_mode(bench, "te_only", [0.30, 0.50], agg_delta=0.00348)
            self._write_mode(bench, "scalar", [0.45, 0.55], agg_delta=0.01000)

            te_vec = analyze_mo_fusion_decoded_properties.mode_seed_vectors(bench, "te_only")
            sc_vec = analyze_mo_fusion_decoded_properties.mode_seed_vectors(bench, "scalar")
            # Vectors keep per-seed order (seed_000 then seed_001), not just the mean.
            self.assertEqual(te_vec["mean_uaug_count"], [0.30, 0.50])
            self.assertEqual(sc_vec["mean_uaug_count"], [0.45, 0.55])
            # A finite paired p-value is computable from the paired seed vectors.
            p = analyze_mo_fusion_decoded_properties.paired_permutation_pvalue(
                sc_vec["mean_uaug_count"], te_vec["mean_uaug_count"]
            )
            self.assertTrue(math.isfinite(p))
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)
            # Identical vectors (zero paired difference) must return p == 1.0.
            p_same = analyze_mo_fusion_decoded_properties.paired_permutation_pvalue(
                te_vec["mean_uaug_count"], te_vec["mean_uaug_count"]
            )
            self.assertEqual(p_same, 1.0)


class TestT1RuntimeReport(unittest.TestCase):
    def test_runtime_report_separates_measured_and_resumed_seed_events(self):
        with tempfile.TemporaryDirectory(prefix="mef_t1_runtime_") as tmp:
            run_dir = os.path.join(tmp, "benchmark", "unit_runtime")
            os.makedirs(run_dir, exist_ok=True)
            summary_path = os.path.join(run_dir, "multiseed_summary.json")
            progress_path = os.path.join(run_dir, "multiseed_progress.jsonl")
            with open(summary_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "aggregate": {
                            "legal_fraction": {"mean": 1.0, "n": 2},
                            "mean_oracle_te": {"mean": 0.79, "n": 2},
                            "delta_oracle_te_vs_source": {"mean": 0.01, "n": 2},
                            "mean_oracle_mrl": {"mean": 8.3, "n": 2},
                        },
                        "config": {
                            "n_records": 4,
                            "seeds": [0, 1],
                            "progress_jsonl": "/remote/mrna_editflow/benchmark/unit_runtime/multiseed_progress.jsonl",
                            "edit_budget": 3,
                            "effective_proposal_top_k": 64,
                            "decoder_family": "checkpoint_guided",
                        },
                        "per_seed": [{"seed": 0}, {"seed": 1}],
                    },
                    fh,
                    indent=2,
                    sort_keys=True,
                )
            events = [
                {"event": "seed_start", "seed": 0, "time": 100.0},
                {"event": "seed_candidates_written", "seed": 0, "time": 110.0, "n_candidates": 4},
                {"event": "seed_evaluated", "seed": 0, "time": 115.0},
                {"event": "seed_resumed", "seed": 0, "time": 119.0},
                {"event": "seed_resumed", "seed": 1, "time": 120.0},
                {"event": "benchmark_complete", "time": 121.0},
            ]
            with open(progress_path, "w", encoding="utf-8") as fh:
                for event in events:
                    fh.write(json.dumps(event, sort_keys=True) + "\n")

            report = summarize_t1_runtime.build_t1_runtime_report(
                project_root=tmp,
                run_specs=[
                    {
                        "label": "head256_mo_grpo",
                        "slice": "head256",
                        "decoder": "mo_grpo",
                        "role": "unit",
                        "summary": os.path.relpath(summary_path, tmp),
                    }
                ],
                expected_seeds=[0, 1],
            )
            row = report["rows"][0]
            self.assertEqual(row["status"], "complete")
            self.assertEqual(row["runtime"]["n_measured_seeds"], 1)
            self.assertEqual(row["runtime"]["n_resumed_seeds"], 1)
            self.assertEqual(row["runtime"]["measured_seeds"], [0])
            self.assertEqual(row["runtime"]["resumed_seeds"], [1])
            self.assertEqual(row["runtime"]["measured_with_resume_marker"], [0])
            self.assertAlmostEqual(row["runtime"]["generation_s"]["mean"], 10.0)
            self.assertAlmostEqual(row["runtime"]["evaluation_s"]["mean"], 5.0)
            self.assertAlmostEqual(row["runtime"]["seed_total_s"]["mean"], 15.0)
            self.assertAlmostEqual(row["runtime"]["measured_records_per_s_total"], 4 / 15)
            self.assertEqual(row["runtime"]["observed_elapsed_scope"], "mixed_resume_wall_clock")
            self.assertFalse(report["interpretation"]["strict_hardware_benchmark_ready"])

            out_json = os.path.join(tmp, "runtime.json")
            out_md = os.path.join(tmp, "runtime.md")
            summarize_t1_runtime.write_report_json(report, out_json)
            summarize_t1_runtime.write_report_markdown(report, out_md)
            with open(out_json, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["artifact_kind"], "t1_runtime_report")
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("T1 Runtime Audit", text)
            self.assertIn("1/1", text)
            self.assertIn("mixed_resume_wall_clock", text)


class TestT2T3DistributionNoveltyReport(unittest.TestCase):
    def _write_eval_summary(self, path, *, seed, exact_match_fraction):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "task_metrics": {
                "T2": {
                    "kmer_js": 0.00001 + seed * 0.000001,
                    "codon_usage_kl": 0.0,
                    "candidate_mean_gc": 0.60 + seed * 0.01,
                    "source_mean_gc": 0.61,
                    "candidate_mean_length": 101.0 + seed,
                    "source_mean_length": 100.0,
                    "gc_quantile_distance": 0.002,
                    "length_quantile_distance": 0.001,
                    "combined_gc_length_distance": 0.003,
                    "embedding_frechet_proxy": 0.0002,
                },
                "T3": {
                    "mean_novelty": 0.02 + seed * 0.001,
                    "exact_source_match_fraction": exact_match_fraction,
                    "unique_fraction": 1.0,
                    "pairwise_diversity": 0.50 + seed * 0.01,
                    "pairwise_diversity_exact": False,
                    "pairwise_pairs_total": 10,
                    "pairwise_pairs_evaluated": 4,
                    "novelty_exact": True,
                    "novelty_source_comparisons": 5 + seed,
                    "novelty_sources_total": 2,
                    "novelty_sources_evaluated_cap": 0,
                },
            }
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

    def test_t2_t3_report_resolves_seed_eval_summaries_and_writes_claim_flags(self):
        with tempfile.TemporaryDirectory(prefix="mef_t2_t3_") as tmp:
            run_dir = os.path.join(tmp, "benchmark", "unit_multiseed")
            os.makedirs(run_dir, exist_ok=True)
            for seed in (0, 1):
                self._write_eval_summary(
                    os.path.join(run_dir, f"seed_{seed:03d}", "eval_summary.json"),
                    seed=seed,
                    exact_match_fraction=0.01,
                )
            summary_path = os.path.join(run_dir, "multiseed_summary.json")
            aggregate = {
                "delta_oracle_te_vs_source": {"mean": 0.011, "n": 2},
                "mean_oracle_te": {"mean": 0.79, "n": 2},
                "legal_fraction": {"mean": 1.0, "n": 2},
                "mean_protein_identity": {"mean": 1.0, "n": 2},
                "within_budget_fraction": {"mean": 1.0, "n": 2},
                "reading_frame_intact_fraction": {"mean": 1.0, "n": 2},
                "kmer_js": {"mean": 0.000011, "n": 2},
                "codon_usage_kl": {"mean": 0.0, "n": 2},
                "mean_novelty": {"mean": 0.0205, "n": 2},
                "exact_source_match_fraction": {"mean": 0.01, "n": 2},
            }
            with open(summary_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "aggregate": aggregate,
                        "config": {
                            "n_records": 2,
                            "seeds": [0, 1],
                            "max_pairwise_pairs": 4,
                            "max_novelty_sources": 0,
                            "edit_budget": 3,
                            "effective_proposal_top_k": 64,
                        },
                        "per_seed": [
                            {
                                "seed": seed,
                                # Deliberately remote-looking; the report must
                                # fall back to seed_XXX/eval_summary.json.
                                "eval_json_path": f"/remote/mrna_editflow/benchmark/unit_multiseed/seed_{seed:03d}/eval_summary.json",
                            }
                            for seed in (0, 1)
                        ],
                    },
                    fh,
                    indent=2,
                    sort_keys=True,
                )

            report = summarize_t2_t3_distribution_novelty.build_t2_t3_report(
                project_root=tmp,
                run_specs=[
                    {
                        "label": "head256_mo_grpo",
                        "slice": "head256",
                        "decoder": "mo_grpo",
                        "role": "unit",
                        "summary": os.path.relpath(summary_path, tmp),
                    }
                ],
                expected_seeds=[0, 1],
            )
            row = report["rows"][0]
            self.assertEqual(row["status"], "complete")
            self.assertEqual(row["seed_eval_audit"]["n_found"], 2)
            self.assertEqual(row["seed_eval_audit"]["n_missing"], 0)
            self.assertAlmostEqual(row["T2_distribution"]["candidate_mean_gc"]["mean"], 0.605)
            self.assertAlmostEqual(row["T3_novelty_diversity"]["pairwise_diversity"]["mean"], 0.505)
            self.assertFalse(row["T3_novelty_diversity"]["pairwise_diversity_exact"]["all_true"])
            self.assertTrue(row["T3_novelty_diversity"]["novelty_exact"]["all_true"])
            self.assertFalse(report["interpretation"]["primary_head256_distribution_collapse_flag"])
            self.assertTrue(report["interpretation"]["primary_head256_de_novo_overclaim_flag"])

            out_json = os.path.join(tmp, "report.json")
            out_md = os.path.join(tmp, "report.md")
            summarize_t2_t3_distribution_novelty.write_report_json(report, out_json)
            summarize_t2_t3_distribution_novelty.write_report_markdown(report, out_md)
            with open(out_json, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["artifact_kind"], "t2_t3_distribution_novelty_report")
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("T2/T3 Distribution And Novelty Audit", text)
            self.assertIn("pairwise diversity can be sampled", text)
            self.assertIn("de-novo overclaim flag: `True`", text)


class TestT4ProteinIdentityCaiGcReport(unittest.TestCase):
    def test_slice_specific_default_paths_are_parameterized(self):
        paths = summarize_t4_protein_identity_cai_gc.default_paths("head1024")
        self.assertEqual(
            paths["protein_conditioned"],
            "benchmark/protein_conditioned_cds_head1024.summary.json",
        )
        self.assertEqual(
            paths["protein_conditioned_codon_metrics"],
            "benchmark/protein_conditioned_codon_metrics_head1024.json",
        )
        self.assertEqual(
            paths["gc_sweep_audit"],
            "benchmark/protein_conditioned_cds_gc_sweep_head1024.audit.json",
        )

    def test_t4_report_combines_dp_protein_conditioned_and_gc_sweep_audits(self):
        with tempfile.TemporaryDirectory(prefix="mef_t4_report_") as tmp:
            bench = os.path.join(tmp, "benchmark")
            os.makedirs(bench, exist_ok=True)
            primary_dir = os.path.join(bench, "multiseed_t5_public_head256_mo_grpo_top64")
            os.makedirs(primary_dir, exist_ok=True)
            with open(
                os.path.join(primary_dir, "multiseed_summary.json"),
                "w",
                encoding="utf-8",
            ) as fh:
                json.dump(
                    {
                        "aggregate": {
                            "mean_protein_identity": {"mean": 1.0},
                            "legal_fraction": {"mean": 1.0},
                            "within_budget_fraction": {"mean": 1.0},
                            "reading_frame_intact_fraction": {"mean": 1.0},
                        }
                    },
                    fh,
                )
            with open(os.path.join(bench, "codon_lattice_dp_head256.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "summary": {
                            "n": 2,
                            "protein_identity_fraction": 1.0,
                            "mean_source_cai": 0.60,
                            "mean_optimized_cai": 0.70,
                            "mean_delta_cai": 0.10,
                            "mean_source_gc": 0.50,
                            "mean_optimized_gc": 0.55,
                            "mean_delta_gc": 0.05,
                            "mean_codon_changes": 3.0,
                        }
                    },
                    fh,
                )
            with open(os.path.join(bench, "protein_conditioned_cds_head256.summary.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "summary": {
                            "n": 2,
                            "mean_protein_identity": 1.0,
                            "protein_identity_eq_1_fraction": 1.0,
                            "native_protein_identity_eq_1_fraction": 1.0,
                            "mean_seed_cai": 0.45,
                            "mean_native_cai": 0.65,
                            "mean_designed_cai": 0.90,
                            "mean_designed_vs_native_cai_delta": 0.25,
                            "mean_seed_gc": 0.42,
                            "mean_native_gc": 0.52,
                            "mean_designed_gc": 0.58,
                            "mean_designed_vs_native_gc_delta": 0.06,
                            "mean_codon_changes": 12.0,
                            "designed_cai_ge_native_fraction": 1.0,
                        }
                    },
                    fh,
                )
            with open(
                os.path.join(bench, "protein_conditioned_codon_metrics_head256.json"),
                "w",
                encoding="utf-8",
            ) as fh:
                json.dump(
                    {
                        "artifact_kind": "protein_conditioned_codon_metrics_audit",
                        "summary": {
                            "ready_for_codon_level_claim_audit": True,
                            "n_rows": 2,
                            "n_with_native_cds": 2,
                            "protein_identity_eq_1_fraction": 1.0,
                            "native_protein_identity_eq_1_fraction": 1.0,
                            "designed_valid_cds_fraction": 1.0,
                            "designed_start_ok_fraction": 1.0,
                            "designed_terminal_stop_ok_fraction": 1.0,
                            "mean_native_codon_recovery": 0.70,
                            "mean_seed_codon_recovery": 0.30,
                            "mean_native_codon_edit_fraction": 0.30,
                            "mean_native_synonymous_substitution_fraction": 0.30,
                            "mean_native_nonsynonymous_substitution_fraction": 0.0,
                            "mean_designed_gc3": 0.90,
                            "mean_native_gc3": 0.55,
                            "designed_vs_native_codon_usage_kl": 0.10,
                            "designed_vs_native_codon_pair_kl": 0.20,
                        },
                    },
                    fh,
                )
            points = [
                {
                    "gc_weight": 0.0,
                    "is_pareto_front": True,
                    "pareto_rank": 0,
                    "summary": {
                        "mean_designed_cai": 1.0,
                        "mean_designed_gc": 0.70,
                        "mean_abs_gc_error": 0.15,
                        "mean_designed_vs_native_cai_delta": 0.30,
                        "mean_designed_vs_native_gc_delta": 0.10,
                        "mean_codon_changes": 20.0,
                        "protein_identity_eq_1_fraction": 1.0,
                        "designed_cai_ge_native_fraction": 1.0,
                    },
                },
                {
                    "gc_weight": 8.0,
                    "is_pareto_front": True,
                    "pareto_rank": 0,
                    "summary": {
                        "mean_designed_cai": 0.86,
                        "mean_designed_gc": 0.60,
                        "mean_abs_gc_error": 0.05,
                        "mean_designed_vs_native_cai_delta": 0.20,
                        "mean_designed_vs_native_gc_delta": 0.00,
                        "mean_codon_changes": 18.0,
                        "protein_identity_eq_1_fraction": 1.0,
                        "designed_cai_ge_native_fraction": 0.99,
                    },
                },
            ]
            with open(
                os.path.join(bench, "protein_conditioned_cds_gc_sweep_head256.summary.json"),
                "w",
                encoding="utf-8",
            ) as fh:
                json.dump(
                    {
                        "sweep_kind": "protein_conditioned_cai_gc_pareto",
                        "n_targets": 2,
                        "points": points,
                    },
                    fh,
                )
            with open(
                os.path.join(bench, "protein_conditioned_cds_gc_sweep_head256.audit.json"),
                "w",
                encoding="utf-8",
            ) as fh:
                json.dump(
                    {
                        "summary": {
                            "ready_for_pareto_claim_audit": True,
                            "all_points_identity_exact_1": True,
                            "pareto_metadata_ok": True,
                            "n_pareto_front": 2,
                        },
                        "jsonl_audit": {
                            "n_rows": 4,
                            "all_row_identity_exact_1": True,
                        },
                    },
                    fh,
                )

            report = summarize_t4_protein_identity_cai_gc.build_t4_report(tmp)
            self.assertTrue(report["summary"]["ready"])
            self.assertTrue(report["summary"]["hard_constraints_exact_1"])
            self.assertTrue(report["summary"]["codon_level_metrics_ready"])
            self.assertFalse(report["summary"]["external_baselines_configured"])
            self.assertFalse(report["summary"]["true_mfe_structure_metric_available"])
            self.assertAlmostEqual(
                report["codon_lattice_dp"]["mean_delta_cai"], 0.10
            )
            self.assertAlmostEqual(
                report["protein_conditioned_codon_metrics"]["mean_native_codon_recovery"],
                0.70,
            )
            self.assertEqual(report["gc_sweep"]["n_points"], 2)
            self.assertEqual(report["gc_sweep"]["best_cai_point"]["gc_weight"], 0.0)
            self.assertEqual(report["gc_sweep"]["best_gc_point"]["gc_weight"], 8.0)

            out_json = os.path.join(tmp, "t4.json")
            out_md = os.path.join(tmp, "t4.md")
            summarize_t4_protein_identity_cai_gc.write_report_json(report, out_json)
            summarize_t4_protein_identity_cai_gc.write_report_markdown(report, out_md)
            with open(out_json, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["artifact_kind"], "t4_protein_identity_cai_gc_report")
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("T4 Protein Identity", text)
            self.assertIn("Codon-level metrics ready: `True`", text)
            self.assertIn("Native codon recovery", text)
            self.assertIn("External baselines configured: `False`", text)
            self.assertIn("True MFE/structure metric available: `False`", text)


class TestProteinConditionedCodonMetricsAudit(unittest.TestCase):
    def test_codon_level_metrics_capture_native_recovery_and_synonymous_edits(self):
        with tempfile.TemporaryDirectory(prefix="mef_codon_metrics_") as tmp:
            jsonl_path = os.path.join(tmp, "protein_conditioned_cds.jsonl")
            rows = [
                {
                    "protein": "MAK",
                    "native_cds": "AUGGCUAAAUAA",
                    "seed_cds": "AUGGCAAAAUAA",
                    "designed_cds": "AUGGCCAAGUAA",
                    "protein_identity": 1.0,
                },
                {
                    "protein": "MPG",
                    "native_cds": "AUGCCCGGGUAA",
                    "seed_cds": "AUGCCUGGAUAA",
                    "designed_cds": "AUGCCCGGGUAA",
                    "protein_identity": 1.0,
                },
            ]
            with open(jsonl_path, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")

            report = audit_protein_conditioned_codon_metrics.build_protein_conditioned_codon_metrics(
                jsonl_path=jsonl_path,
                project_root=tmp,
                top_n=5,
            )
            summary = report["summary"]
            self.assertEqual(report["artifact_kind"], "protein_conditioned_codon_metrics_audit")
            self.assertTrue(summary["ready_for_codon_level_claim_audit"])
            self.assertEqual(summary["protein_identity_eq_1_fraction"], 1.0)
            self.assertEqual(summary["native_protein_identity_eq_1_fraction"], 1.0)
            self.assertAlmostEqual(summary["mean_native_codon_recovery"], 0.75)
            self.assertAlmostEqual(summary["mean_native_nonsynonymous_substitution_fraction"], 0.0)
            self.assertGreater(summary["mean_native_synonymous_substitution_fraction"], 0.0)
            self.assertGreaterEqual(summary["designed_vs_native_codon_usage_kl"], 0.0)
            self.assertGreaterEqual(summary["designed_vs_native_codon_pair_kl"], 0.0)
            self.assertEqual(len(report["row_metrics"]), 2)
            self.assertGreater(len(report["top_codon_frequency_deltas"]), 0)

            out_json = os.path.join(tmp, "codon_metrics.json")
            out_md = os.path.join(tmp, "codon_metrics.md")
            audit_protein_conditioned_codon_metrics.write_report_json(report, out_json)
            audit_protein_conditioned_codon_metrics.write_report_markdown(report, out_md)
            with open(out_json, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["artifact_kind"], "protein_conditioned_codon_metrics_audit")
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Protein-Conditioned Codon Metrics Audit", text)
            self.assertIn("Mean native codon recovery", text)


class TestT6LengthCurveReport(unittest.TestCase):
    def _write_t6_summary(self, root, slice_name="head256", delta=-30, suffix=None, seeds=None):
        if seeds is None:
            seeds = list(range(10))
        name = f"neg{abs(delta)}" if delta < 0 else f"pos{delta}"
        if suffix:
            name = f"{name}_{suffix}"
        out_dir = os.path.join(
            root,
            "benchmark",
            f"multiseed_t6_public_{slice_name}_stagea10k_len_{name}_top64",
        )
        os.makedirs(out_dir, exist_ok=True)
        aggregate = {
            "mean_abs_length_error": {"mean": 0.25, "n": len(seeds)},
            "legal_fraction": {"mean": 1.0, "n": len(seeds)},
            "mean_protein_identity": {"mean": 1.0, "n": len(seeds)},
            "within_budget_fraction": {"mean": 1.0, "n": len(seeds)},
            "reading_frame_intact_fraction": {"mean": 1.0, "n": len(seeds)},
            "delta_oracle_te_vs_source": {"mean": 0.005, "n": len(seeds)},
            "mean_oracle_te": {"mean": 0.78, "n": len(seeds)},
            "mean_edit_distance": {"mean": 29.75, "n": len(seeds)},
        }
        path = os.path.join(out_dir, "multiseed_summary.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "aggregate": aggregate,
                    "config": {"seeds": list(seeds)},
                    "per_seed": [{"seed": int(seed), "metrics": {}} for seed in seeds],
                },
                fh,
                indent=2,
                sort_keys=True,
            )
        return path

    def test_t6_length_curve_report_reads_completed_rows_and_marks_running(self):
        with tempfile.TemporaryDirectory(prefix="mef_t6_curve_") as tmp:
            summary_path = self._write_t6_summary(tmp, "head256", -30)
            payload = summarize_t6_length_curve.build_t6_length_curve_report(
                project_root=tmp,
                slices=["head256", "head1024"],
                running_logs={"head1024": "logs/t6_head1024.log"},
                running_deltas={"head1024": [-30]},
            )

            head256 = payload["head256_stagea10k"]
            self.assertEqual(head256["status"], "partial")
            self.assertEqual(len(head256["rows"]), 1)
            self.assertEqual(head256["rows"][0]["target_length_delta"], -30)
            self.assertEqual(head256["rows"][0]["summary_path"], os.path.relpath(summary_path, tmp))
            self.assertEqual(len(head256["rows"][0]["summary_sha256"]), 64)
            self.assertAlmostEqual(head256["rows"][0]["mean_abs_length_error"], 0.25)
            self.assertEqual(head256["pending_target_length_deltas"], [-15, 0, 15, 30])

            head1024 = payload["head1024_stagea10k"]
            self.assertEqual(head1024["status"], "running")
            self.assertEqual(head1024["pending_target_length_deltas"], [-30, -15, 0, 15, 30])
            self.assertEqual(head1024["running_target_length_deltas"], [-30])

            out_json = os.path.join(tmp, "report.json")
            out_md = os.path.join(tmp, "report.md")
            summarize_t6_length_curve.write_report_json(payload, out_json)
            summarize_t6_length_curve.write_report_markdown(payload, out_md)
            with open(out_json, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            self.assertEqual(loaded["artifact_kind"], "t6_length_curve_report")
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Head256 Stage A 10k", text)
            self.assertIn("| -30 | 0.25000 | +0.00500 | 0.78000 | 29.75000 | legal/protein/budget/frame = 1.0 |", text)
            self.assertIn("| -30 | pending | pending | pending | pending | running |", text)

    def test_t6_length_curve_report_accepts_nonstandard_parallel_summary(self):
        with tempfile.TemporaryDirectory(prefix="mef_t6_curve_parallel_") as tmp:
            summary_path = self._write_t6_summary(
                tmp,
                slice_name="head1024",
                delta=-15,
                suffix="parallel_20260715",
            )
            payload = summarize_t6_length_curve.build_t6_length_curve_report(
                project_root=tmp,
                slices=["head1024"],
            )
            head1024 = payload["head1024_stagea10k"]
            self.assertEqual(head1024["status"], "partial")
            rows = {row["target_length_delta"]: row for row in head1024["rows"]}
            self.assertIn(-15, rows)
            self.assertEqual(rows[-15]["summary_path"], os.path.relpath(summary_path, tmp))
            self.assertEqual(head1024["pending_target_length_deltas"], [-30, 0, 15, 30])

    def test_t6_length_curve_report_ignores_incomplete_parallel_summary(self):
        with tempfile.TemporaryDirectory(prefix="mef_t6_curve_partial_") as tmp:
            self._write_t6_summary(
                tmp,
                slice_name="head1024",
                delta=15,
                suffix="shard_a",
                seeds=list(range(5)),
            )
            payload = summarize_t6_length_curve.build_t6_length_curve_report(
                project_root=tmp,
                slices=["head1024"],
            )
            head1024 = payload["head1024_stagea10k"]
            self.assertEqual(head1024["status"], "pending")
            self.assertEqual(head1024["rows"], [])
            self.assertEqual(head1024["pending_target_length_deltas"], [-30, -15, 0, 15, 30])


class TestMergeMultiseedShards(unittest.TestCase):
    def _write_seed_eval(self, root, seed, *, delta_te=0.01):
        seed_dir = os.path.join(root, f"seed_{seed:03d}")
        os.makedirs(seed_dir, exist_ok=True)
        with open(os.path.join(seed_dir, "candidates.jsonl"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"transcript_id": f"cand_{seed}"}) + "\n")
        summary = {
            "task_metrics": {
                "T1": {
                    "legal_fraction": 1.0,
                    "mean_oracle_te": 0.78 + seed * 0.001,
                    "mean_oracle_mrl": 8.0,
                },
                "T2": {"kmer_js": 0.001, "codon_usage_kl": 0.0},
                "T3": {"mean_novelty": 0.03, "exact_source_match_fraction": 0.0},
                "T4": {"mean_protein_identity": 1.0},
                "T5": {"within_budget_fraction": 1.0, "mean_edit_distance": 29.0},
                "T6": {"mean_abs_length_error": 0.25},
                "T7": {"reading_frame_intact_fraction": 1.0},
            },
            "per_record_metrics": {
                "oracle_ensemble_te": [0.50 + delta_te + seed * 0.001, 0.60 + delta_te],
                "source_oracle_ensemble_te": [0.50, 0.60],
            },
        }
        with open(os.path.join(seed_dir, "eval_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
        with open(os.path.join(seed_dir, "paper_table.md"), "w", encoding="utf-8") as fh:
            fh.write("| metric | value |\n")

    def test_merge_multiseed_shards_writes_complete_summary(self):
        with tempfile.TemporaryDirectory(prefix="mef_merge_shards_") as tmp:
            shard_a = os.path.join(tmp, "shard_a")
            shard_b = os.path.join(tmp, "shard_b")
            os.makedirs(shard_a)
            os.makedirs(shard_b)
            with open(os.path.join(shard_a, "sources.jsonl"), "w", encoding="utf-8") as fh:
                fh.write("{}\n{}\n")
            for seed in range(5):
                self._write_seed_eval(shard_a, seed)
            for seed in range(5, 10):
                self._write_seed_eval(shard_b, seed)
            out_dir = os.path.join(
                tmp,
                "benchmark",
                "multiseed_t6_public_head1024_stagea10k_len_pos15_merged_top64",
            )
            result = merge_multiseed_shards.merge_multiseed_shards(
                source_dirs=[shard_a, shard_b],
                out_dir=out_dir,
                expected_seeds=list(range(10)),
                checkpoint_path="ckpts/stage_a.pt",
                target_length_delta=15,
            )
            self.assertTrue(result["merge_audit"]["complete"])
            self.assertEqual(result["merge_audit"]["merged_seeds"], list(range(10)))
            summary_path = result["json_path"]
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, "r", encoding="utf-8") as fh:
                summary = json.load(fh)
            self.assertEqual(summary["config"]["seeds"], list(range(10)))
            self.assertEqual(summary["aggregate"]["mean_abs_length_error"]["n"], 10)
            self.assertTrue(summarize_t6_length_curve._is_complete_summary_path(summary_path))


class TestT4ExternalCdsComparison(unittest.TestCase):
    def _write(self, root, rel, payload):
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    def test_complete_lineardesign_row_enables_descriptive_not_superiority_table(self):
        with tempfile.TemporaryDirectory(prefix="mef_t4_external_cds_") as tmp:
            self._write(
                tmp,
                "benchmark/protein_conditioned_cds_head1024.summary.json",
                {
                    "summary": {
                        "n": 1024,
                        "mean_native_cai": 0.69,
                        "mean_native_gc": 0.56,
                        "mean_designed_cai": 1.0,
                        "mean_designed_vs_native_cai_delta": 0.31,
                        "mean_designed_gc": 0.69,
                        "mean_designed_vs_native_gc_delta": 0.13,
                        "protein_identity_eq_1_fraction": 1.0,
                    }
                },
            )
            self._write(
                tmp,
                "benchmark/protein_conditioned_codon_metrics_head1024.json",
                {
                    "summary": {
                        "mean_native_gc3": 0.67,
                        "mean_designed_gc3": 1.0,
                        "designed_vs_native_codon_usage_kl": 0.70,
                        "designed_vs_native_codon_pair_kl": 1.40,
                    }
                },
            )
            self._write(
                tmp,
                "benchmark/codon_lattice_dp_head1024.json",
                {
                    "summary": {
                        "n": 1024,
                        "mean_optimized_cai": 0.72,
                        "mean_delta_cai": 0.03,
                        "mean_optimized_gc": 0.57,
                        "mean_delta_gc": 0.01,
                        "protein_identity_fraction": 1.0,
                    }
                },
            )
            self._write(
                tmp,
                "benchmark/external_sota/real_runs_t5_head1024/LinearDesign/summary.json",
                {
                    "n_inputs": 1024,
                    "n_outputs": 1024,
                    "n_failures": 0,
                    "mean_cai": 0.80,
                    "mean_gc": 0.62,
                    "mean_gc3": 0.78,
                    "mean_codon_usage_kl_vs_native": 0.45,
                    "mean_codon_pair_kl_vs_native": 0.90,
                    "protein_identity_exact_1_fraction": 1.0,
                    "valid_cds_fraction": 1.0,
                    "mean_mfe_without_stop": -500.0,
                    "mean_wall_clock_s": 60.0,
                },
            )
            self._write(
                tmp,
                "benchmark/external_sota/real_runs_t5_head1024/EnsembleDesign/summary.json",
                {
                    "n_inputs": 1024,
                    "n_outputs": 1024,
                    "n_failures": 0,
                    "mean_cai": 0.76,
                    "mean_gc": 0.60,
                    "mean_gc3": 0.75,
                    "mean_codon_usage_kl_vs_native": 0.50,
                    "mean_codon_pair_kl_vs_native": 1.0,
                    "protein_identity_exact_1_fraction": 1.0,
                    "valid_cds_fraction": 1.0,
                    "mean_ensemble_free_energy": -450.0,
                    "mean_wall_clock_s": 200.0,
                },
            )
            self._write(
                tmp,
                "benchmark/external_sota/real_runs_t5_head1024/codonGPT/summary.json",
                {
                    "n_inputs": 1024,
                    "n_outputs": 1024,
                    "n_failures": 0,
                    "mean_cai": 0.74,
                    "mean_gc": 0.58,
                    "mean_gc3": 0.72,
                    "mean_codon_accuracy_vs_native": 0.42,
                    "mean_codon_usage_kl_vs_native": 0.55,
                    "mean_codon_pair_kl_vs_native": 1.10,
                    "protein_identity_exact_1_fraction": 1.0,
                    "valid_cds_fraction": 1.0,
                    "mean_wall_clock_s": 0.2,
                },
            )
            self._write(
                tmp,
                "benchmark/external_sota/"
                "codongpt_multiseed_head1024/summary.json",
                {
                    "summary": {
                        "complete_10seed_head1024": True,
                        "hard_constraints_exact_1": True,
                        "n_complete_seeds": 10,
                        "delta_cai_vs_native_paired_signflip_p": 0.0045,
                    },
                    "aggregate": {
                        "mean_cai": {"mean": 0.75},
                        "mean_gc": {"mean": 0.59},
                        "mean_gc3": {"mean": 0.73},
                        "mean_codon_accuracy_vs_native": {"mean": 0.43},
                        "mean_codon_usage_kl_vs_native": {"mean": 0.54},
                        "mean_codon_pair_kl_vs_native": {"mean": 1.09},
                        "mean_wall_clock_s": {"mean": 0.21},
                        "delta_cai_vs_native": {
                            "mean": 0.06,
                            "low": 0.05,
                            "high": 0.07,
                        },
                    },
                },
            )
            self._write(
                tmp,
                "docs/external_sota_real_run_audit.json",
                {
                    "rows": [
                        {
                            "model_name": "LinearDesign",
                            "status": "measured",
                            "real_metric_ready": True,
                        },
                        {
                            "model_name": "EnsembleDesign",
                            "status": "measured",
                            "real_metric_ready": True,
                        },
                        {
                            "model_name": "codonGPT",
                            "status": "measured",
                            "real_metric_ready": True,
                        },
                    ]
                },
            )
            report = build_t4_external_cds_comparison.build_t4_external_cds_comparison(tmp)
            self.assertTrue(report["summary"]["ready_for_t4_external_cds_descriptive_table"])
            self.assertFalse(report["summary"]["ready_for_mef_superiority_claim"])
            linear = next(row for row in report["rows"] if row["method"] == "LinearDesign_official")
            self.assertEqual(linear["status"], "measured_external")
            self.assertEqual(linear["n"], 1024)
            self.assertAlmostEqual(linear["delta_cai_vs_native"], 0.11)
            ensemble = next(
                row
                for row in report["rows"]
                if row["method"] == "EnsembleDesign_official"
            )
            self.assertEqual(ensemble["status"], "measured_external_budgeted")
            self.assertEqual(ensemble["mean_ensemble_free_energy"], -450.0)
            self.assertTrue(
                report["summary"][
                    "both_external_optimizers_complete_head1024"
                ]
            )
            codongpt = next(
                row
                for row in report["rows"]
                if row["method"] == "codonGPT_official_HF_pretrained"
            )
            self.assertEqual(
                codongpt["status"],
                "measured_external_pretrained_checkpoint_10seed",
            )
            self.assertEqual(codongpt["codon_accuracy_vs_native"], 0.43)
            self.assertEqual(codongpt["n_seeds"], 10)
            self.assertEqual(
                codongpt["delta_cai_seed_paired_p_vs_native"],
                0.0045,
            )
            self.assertTrue(
                report["summary"]["codongpt_complete_head1024"]
            )
            self.assertTrue(
                report["summary"]["codongpt_10seed_complete_head1024"]
            )
            self.assertTrue(
                report["summary"]["codongpt_seed_level_inference_ready"]
            )
            self.assertFalse(
                report["summary"]["codongpt_rl_policy_reproduced"]
            )

    def test_codongpt_multiseed_summary_requires_ten_exact_seed_runs(self):
        with tempfile.TemporaryDirectory(prefix="mef_codongpt_multiseed_") as tmp:
            pack_dir = os.path.join(
                tmp,
                "benchmark/external_sota/input_pack_t5_head1024",
            )
            os.makedirs(pack_dir, exist_ok=True)
            cds_path = os.path.join(pack_dir, "cds_protein_inputs.jsonl")
            with open(cds_path, "w", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "transcript_id": "tx",
                            "native_cds": "AUGGCUUAA",
                        }
                    )
                    + "\n"
                )
            self._write(
                tmp,
                "benchmark/external_sota/input_pack_t5_head1024/summary.json",
                {
                    "outputs": {
                        "cds_protein_jsonl": cds_path,
                    }
                },
            )
            for seed in range(10):
                run_rel = (
                    "benchmark/external_sota/real_runs_t5_head1024/codonGPT"
                    if seed == 0
                    else (
                        "benchmark/external_sota/"
                        f"codongpt_multiseed_head1024/seed_{seed:03d}"
                    )
                )
                self._write(
                    tmp,
                    f"{run_rel}/summary.json",
                    {
                        "runtime": {"seed": seed},
                        "n_inputs": 1,
                        "n_outputs": 1,
                        "n_failures": 0,
                        "valid_cds_fraction": 1.0,
                        "protein_identity_exact_1_fraction": 1.0,
                        "mean_cai": 1.0,
                        "mean_gc": 4.0 / 9.0,
                        "mean_gc3": 1.0 / 3.0,
                        "mean_codon_accuracy_vs_native": 1.0,
                        "mean_codon_usage_kl_vs_native": 0.0,
                        "mean_codon_pair_kl_vs_native": 0.0,
                        "mean_wall_clock_s": 0.1,
                    },
                )
                path = os.path.join(tmp, run_rel, "cds_outputs.jsonl")
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "transcript_id": "tx",
                                "designed_cds": "AUGGCUUAA",
                            }
                        )
                        + "\n"
                    )
            report = (
                build_codongpt_multiseed_summary
                .build_codongpt_multiseed_summary(tmp)
            )
            self.assertTrue(
                report["summary"]["complete_10seed_head1024"]
            )
            self.assertTrue(
                report["summary"]["hard_constraints_exact_1"]
            )
            self.assertEqual(len(report["per_seed"]), 10)
            self.assertEqual(
                report["aggregate"]["mean_codon_accuracy_vs_native"]["mean"],
                1.0,
            )
            self.assertFalse(report["summary"]["ready_for_paper_rl_claim"])


class TestT5ExternalUtrComparison(unittest.TestCase):
    def _write_json(self, root, rel, payload):
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)

    def _write_jsonl(self, root, rel, rows):
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    def test_report_json_canonicalizes_cross_platform_float_noise(self):
        with tempfile.TemporaryDirectory(prefix="mef_t5_json_") as tmp:
            path_a = os.path.join(tmp, "a.json")
            path_b = os.path.join(tmp, "b.json")
            build_t5_external_utr_comparison.write_report_json(
                {"value": 0.006464236355026222},
                path_a,
            )
            build_t5_external_utr_comparison.write_report_json(
                {"value": 0.006464236355026225},
                path_b,
            )
            with open(path_a, "rb") as fh:
                payload_a = fh.read()
            with open(path_b, "rb") as fh:
                payload_b = fh.read()
            self.assertEqual(payload_a, payload_b)
            self.assertEqual(json.loads(payload_a)["value"], 0.0064642363550262)

    def test_scores_nonempty_utailor_eligible_model_subset(self):
        with tempfile.TemporaryDirectory(prefix="mef_t5_subset_") as tmp:
            candidate_path = os.path.join(
                tmp, "seed_000", "candidates.jsonl"
            )
            self._write_jsonl(
                tmp,
                "seed_000/candidates.jsonl",
                [
                    {
                        "transcript_id": "tx_t5_model",
                        "five_utr": "G" + "A" * 29,
                        "cds": "AUGGCCUAA",
                        "three_utr": "AAUAAA",
                    }
                ],
            )
            rows, failures = (
                build_t5_external_utr_comparison._score_model_subset(
                    candidate_paths=[candidate_path],
                    source_rows=[
                        {
                            "transcript_id": "tx",
                            "five_utr": "A" * 30,
                            "cds": "AUGGCCUAA",
                            "three_utr": "AAUAAA",
                        }
                    ],
                    eligible_ids={"tx"},
                )
            )
            self.assertEqual(failures, [])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["n"], 1)
            self.assertEqual(
                rows[0]["mean_utr_edit_distance_vs_native"],
                1.0,
            )

    def test_accepts_complete_ten_seed_protocol_subset_with_explicit_size(self):
        with tempfile.TemporaryDirectory(prefix="mef_t5_protocol_subset_") as tmp:
            sources = [
                {
                    "transcript_id": f"tx{idx}",
                    "five_utr": "A" * 30,
                    "cds": "AUGGCCUAA",
                    "three_utr": "AAUAAA",
                }
                for idx in range(2)
            ]
            candidate_paths = []
            seed_payloads = []
            for seed in range(10):
                rel = f"seed_{seed:03d}/candidates.jsonl"
                self._write_jsonl(
                    tmp,
                    rel,
                    [
                        {
                            **row,
                            "transcript_id": (
                                f"{row['transcript_id']}_t5_model"
                            ),
                            "five_utr": "G" + "A" * 29,
                        }
                        for row in sources
                    ],
                )
                candidate_paths.append(os.path.join(tmp, rel))
                seed_payloads.append(
                    {
                        "metrics": {
                            "kozak_uaug": {
                                "mean_uaug_count": 0.0,
                                "mean_kozak_score": 0.7,
                            },
                            "structure": {
                                "mean_start_accessibility": 0.6,
                            },
                            "diversity_novelty": {
                                "unique_fraction": 1.0,
                            },
                        }
                    }
                )
            summary = {
                "config": {
                    "editable_regions": ["utr5"],
                    "edit_budget": 5,
                },
                "aggregate": {
                    "mean_oracle_te": {"mean": 0.8, "n": 10},
                    "delta_oracle_te_vs_source": {
                        "mean": 0.01,
                        "low": 0.009,
                        "high": 0.011,
                        "n": 10,
                    },
                    "mean_edit_distance": {"mean": 1.0, "n": 10},
                    "exact_source_match_fraction": {
                        "mean": 0.0,
                        "n": 10,
                    },
                    "mean_protein_identity": {"mean": 1.0, "n": 10},
                    "within_budget_fraction": {"mean": 1.0, "n": 10},
                    "legal_fraction": {"mean": 1.0, "n": 10},
                },
                "per_seed": [
                    {
                        "seed": seed,
                        "metrics": {
                            "delta_oracle_te_vs_source": 0.01,
                        },
                    }
                    for seed in range(10)
                ],
            }
            row, failures = (
                build_t5_external_utr_comparison._mef_utr5_model_row(
                    summary,
                    seed_payloads,
                    sources,
                    candidate_paths,
                    expected_n=2,
                    measured_status="measured_protocol_subset",
                )
            )
            self.assertEqual(failures, [])
            self.assertEqual(row["status"], "measured_protocol_subset")
            self.assertEqual(row["n"], 2)
            self.assertEqual(row["edit_budget"], 5)
            self.assertEqual(row["cds_unchanged_fraction"], 1.0)
            self.assertEqual(row["three_utr_unchanged_fraction"], 1.0)

    def test_complete_fixed_region_rows_enable_descriptive_not_model_claim(self):
        with tempfile.TemporaryDirectory(prefix="mef_t5_external_utr_") as tmp:
            input_rows = [
                {
                    "transcript_id": f"tx{idx}",
                    "native_five_utr": "AAAAAA",
                    "fixed_cds_context": "AUGGCCUAA",
                    "fixed_three_utr_context": "AAUAAA",
                }
                for idx in range(1024)
            ]
            self._write_json(
                tmp,
                "benchmark/external_sota/input_pack_t5_head1024/summary.json",
                {
                    "artifact_kind": "external_sota_input_pack",
                    "ready_for_external_real_run": True,
                },
            )
            self._write_jsonl(
                tmp,
                "benchmark/external_sota/input_pack_t5_head1024/utr5_inputs.jsonl",
                input_rows,
            )
            self._write_json(
                tmp,
                "docs/external_sota_real_run_audit.json",
                {
                    "rows": [
                        {
                            "model_name": "UTRGAN",
                            "status": "measured",
                            "real_metric_ready": True,
                        }
                    ]
                },
            )
            local_rows = [
                {
                    "transcript_id": row["transcript_id"],
                    "source_five_utr": "AAAAAA",
                    "optimized_five_utr": "GAAAAA",
                    "delta_te": 0.01,
                    "utr_edit_distance": 1,
                }
                for row in input_rows
            ]
            self._write_json(
                tmp,
                "benchmark/utr_local_search_head1024.json",
                {
                    "config": {"edit_budget": 3},
                    "runtime": {"mean_wall_clock_s": 0.1},
                    "summary": {
                        "cds_unchanged_fraction": 1.0,
                        "three_utr_unchanged_fraction": 1.0,
                    },
                    "per_record": local_rows,
                },
            )
            external_rows = [
                {
                    "transcript_id": row["transcript_id"],
                    "te_proxy": 0.8,
                    "te_proxy_delta_vs_native": 0.02,
                    "uaug_count": 0.0,
                    "kozak_score": 0.5,
                    "start_accessibility_proxy": 0.6,
                    "utr_edit_distance_vs_native": 5,
                    "normalized_utr_edit_distance_vs_native": 0.5,
                    "designed_utr_length": 10,
                    "utr_length_delta": 4,
                    "exact_native_utr_match": False,
                }
                for row in input_rows
            ]
            self._write_jsonl(
                tmp,
                "benchmark/external_sota/real_runs_t5_head1024/UTRGAN/utr5_outputs.jsonl",
                external_rows,
            )
            self._write_json(
                tmp,
                "benchmark/external_sota/real_runs_t5_head1024/UTRGAN/summary.json",
                {
                    "n_inputs": 1024,
                    "n_outputs": 1024,
                    "n_failures": 0,
                    "mean_te_proxy": 0.8,
                    "mean_te_proxy_delta_vs_native": 0.02,
                    "mean_uaug_count": 0.0,
                    "mean_kozak_score": 0.5,
                    "mean_start_accessibility_proxy": 0.6,
                    "mean_utr_edit_distance_vs_native": 5.0,
                    "mean_normalized_utr_edit_distance_vs_native": 0.5,
                    "mean_designed_utr_length": 10.0,
                    "mean_utr_length_delta": 4.0,
                    "exact_native_utr_match_fraction": 0.0,
                    "unique_designed_utr_fraction": 1.0,
                    "cds_unchanged_fraction": 1.0,
                    "three_utr_unchanged_fraction": 1.0,
                    "protein_identity_exact_1_fraction": 1.0,
                    "mean_wall_clock_s": 0.02,
                    "protocol_fidelity": "budgeted",
                    "protocol_fidelity_sufficient_for_sota_reproduction": False,
                },
            )
            input_pack_summary_path = os.path.join(
                tmp,
                "benchmark/external_sota/input_pack_t5_head1024/summary.json",
            )
            with open(input_pack_summary_path, "rb") as fh:
                input_pack_summary_sha = hashlib.sha256(fh.read()).hexdigest()
            paper_summary = {
                "n_inputs": 1024,
                "n_outputs": 1024,
                "n_failures": 0,
                "mean_te_proxy": 0.81,
                "mean_te_proxy_delta_vs_native": 0.03,
                "mean_uaug_count": 0.0,
                "mean_kozak_score": 0.5,
                "mean_start_accessibility_proxy": 0.6,
                "mean_utr_edit_distance_vs_native": 5.0,
                "mean_normalized_utr_edit_distance_vs_native": 0.5,
                "mean_designed_utr_length": 10.0,
                "mean_utr_length_delta": 4.0,
                "exact_native_utr_match_fraction": 0.0,
                "unique_designed_utr_fraction": 1.0,
                "cds_unchanged_fraction": 1.0,
                "three_utr_unchanged_fraction": 1.0,
                "protein_identity_exact_1_fraction": 1.0,
                "mean_wall_clock_s": 0.2,
                "protocol_fidelity": (
                    "official_code_paper_default_10000_steps"
                ),
                "protocol_fidelity_sufficient_for_sota_reproduction": True,
                "input_pack": {
                    "summary_sha256": input_pack_summary_sha
                },
            }
            paper_external_rows = [
                {
                    **row,
                    "te_proxy": 0.81,
                    "te_proxy_delta_vs_native": 0.03,
                }
                for row in external_rows
            ]
            self._write_jsonl(
                tmp,
                "benchmark/external_sota/real_runs_t5_head1024/"
                "UTRGAN_paper10000/utr5_outputs.jsonl",
                paper_external_rows,
            )
            self._write_json(
                tmp,
                "benchmark/external_sota/real_runs_t5_head1024/"
                "UTRGAN_paper10000/summary.json",
                paper_summary,
            )
            self._write_json(
                tmp,
                "benchmark/multiseed_t5_public_head1024_mo_pareto_top64/multiseed_summary.json",
                {
                    "aggregate": {
                        "mean_oracle_te": {"mean": 0.79, "n": 10},
                        "delta_oracle_te_vs_source": {"mean": 0.01, "n": 10},
                        "mean_edit_distance": {"mean": 3.0, "n": 10},
                        "mean_protein_identity": {"mean": 1.0, "n": 10},
                        "within_budget_fraction": {"mean": 1.0, "n": 10},
                        "legal_fraction": {"mean": 1.0, "n": 10},
                    }
                },
            )
            self._write_json(
                tmp,
                "benchmark/multiseed_t5_public_head1024_mo_pareto_top64/seed_000/eval_summary.json",
                {
                    "n_candidates": 1024,
                    "metrics": {
                        "kozak_uaug": {
                            "mean_uaug_count": 0.2,
                            "mean_kozak_score": 0.6,
                        },
                        "structure": {"mean_start_accessibility": 0.7},
                    },
                },
            )
            utr5_run = (
                "benchmark/"
                "multiseed_t5_public_head1024_region_adapter_utr5only_top64"
            )
            model_sources = [
                {
                    "transcript_id": row["transcript_id"],
                    "five_utr": "AAAAAA",
                    "cds": "AUGGCCUAA",
                    "three_utr": "AAUAAA",
                }
                for row in input_rows
            ]
            self._write_jsonl(
                tmp,
                f"{utr5_run}/sources.jsonl",
                model_sources,
            )
            self._write_json(
                tmp,
                f"{utr5_run}/multiseed_summary.json",
                {
                    "config": {
                        "editable_regions": ["utr5"],
                        "edit_budget": 3,
                    },
                    "aggregate": {
                        "mean_oracle_te": {"mean": 0.81, "n": 10},
                        "delta_oracle_te_vs_source": {
                            "mean": 0.015,
                            "n": 10,
                        },
                        "mean_edit_distance": {"mean": 1.0, "n": 10},
                        "exact_source_match_fraction": {
                            "mean": 0.0,
                            "n": 10,
                        },
                        "mean_protein_identity": {"mean": 1.0, "n": 10},
                        "within_budget_fraction": {"mean": 1.0, "n": 10},
                        "legal_fraction": {"mean": 1.0, "n": 10},
                    },
                    "per_seed": [
                        {
                            "seed": seed,
                            "metrics": {
                                "delta_oracle_te_vs_source": 0.015
                            },
                        }
                        for seed in range(10)
                    ],
                },
            )
            for seed in range(10):
                candidates = [
                    {
                        "transcript_id": f"{row['transcript_id']}_t5_model",
                        "five_utr": "GAAAAA",
                        "cds": "AUGGCCUAA",
                        "three_utr": "AAUAAA",
                    }
                    for row in input_rows
                ]
                self._write_jsonl(
                    tmp,
                    f"{utr5_run}/seed_{seed:03d}/candidates.jsonl",
                    candidates,
                )
                self._write_json(
                    tmp,
                    f"{utr5_run}/seed_{seed:03d}/eval_summary.json",
                    {
                        "n_candidates": 1024,
                        "metrics": {
                            "kozak_uaug": {
                                "mean_uaug_count": 0.1,
                                "mean_kozak_score": 0.7,
                            },
                            "structure": {
                                "mean_start_accessibility": 0.65
                            },
                            "diversity_novelty": {
                                "unique_fraction": 0.5
                            },
                        },
                    },
                )

            report = (
                build_t5_external_utr_comparison.build_t5_external_utr_comparison(
                    tmp
                )
            )
            self.assertTrue(
                report["summary"]["ready_for_t5_utrgan_descriptive_table"]
            )
            self.assertFalse(
                report["summary"]["ready_for_t5_utr_descriptive_table"]
            )
            self.assertFalse(
                report["summary"]["ready_for_model_only_head_to_head"]
            )
            self.assertTrue(
                report["summary"][
                    "ready_for_model_only_descriptive_head_to_head"
                ]
            )
            self.assertTrue(
                report["summary"]["mef_model_utr5_only_run_available"]
            )
            utr5_model = next(
                row
                for row in report["rows"]
                if row["method"] == "MEF_region_adapter_utr5only_top64"
            )
            self.assertEqual(
                utr5_model["te_proxy_delta_seed_signal"],
                "significant_positive",
            )
            self.assertLess(
                utr5_model["te_proxy_delta_seed_paired_p_vs_source"],
                0.05,
            )
            self.assertFalse(
                report["summary"]["ready_for_paired_per_record_inference"]
            )
            self.assertFalse(report["summary"]["ready_for_mef_superiority_claim"])
            self.assertTrue(
                report["summary"][
                    "utrgan_paper10000_measured_complete_head1024"
                ]
            )
            paper_utrgan = next(
                row
                for row in report["rows"]
                if row["method"]
                == "UTRGAN_official_paper_default_10000_steps"
            )
            self.assertEqual(
                paper_utrgan["status"],
                "measured_external_paper_default",
            )
            self.assertTrue(
                paper_utrgan[
                    "protocol_fidelity_sufficient_for_sota_reproduction"
                ]
            )
            protocol_comparison = report[
                "utrgan_paper10000_vs_budgeted10"
            ]
            te_delta = next(
                row
                for row in protocol_comparison["rows"]
                if row["metric"] == "te_proxy_delta_vs_native"
            )
            self.assertAlmostEqual(
                te_delta["paper_default_minus_budgeted"],
                0.01,
            )
            self.assertIsNone(te_delta["paired_p"])
            local = next(
                row
                for row in report["rows"]
                if row["method"] == "MEF_utr5_constrained_local_search_budget3"
            )
            self.assertEqual(local["n"], 1024)
            self.assertEqual(local["within_edit_budget_fraction"], 1.0)
            self.assertTrue(
                all(
                    row["paired_p"] is None
                    for row in report["distributional_comparison"]["rows"]
                )
            )


class TestMultiScaleSequenceSpectrumAudit(unittest.TestCase):
    def test_audit_writes_base_region_distributions_and_svg_figures(self):
        with tempfile.TemporaryDirectory(prefix="mef_spectrum_audit_") as tmp:
            cand_path = os.path.join(tmp, "candidates.jsonl")
            src_path = os.path.join(tmp, "sources.jsonl")
            candidates = [
                MRNARecord("c1", "AACCGG", "AUGGCCGCUUAA", "UUUAAA"),
                MRNARecord("c2", "GGGGAA", "AUGCCCGGGUAA", "CCCCUU"),
            ]
            sources = [
                MRNARecord("s1", "AAAAAA", "AUGGCUAAAUAA", "UUUUUU"),
                MRNARecord("s2", "CCCCAA", "AUGCCUGGAUAA", "AAAACC"),
            ]
            write_records_jsonl(candidates, cand_path)
            write_records_jsonl(sources, src_path)
            fig_dir = os.path.join(tmp, "figures")
            report = multi_scale_sequence_spectrum_audit.build_multi_scale_sequence_spectrum_audit(
                candidate_paths=[cand_path],
                source_paths=[src_path],
                out_fig_dir=fig_dir,
                kmer_k=2,
                top_n=5,
            )
            self.assertEqual(report["artifact_kind"], "multi_scale_sequence_spectrum_audit")
            self.assertEqual(report["summary"]["n_candidates"], 2)
            self.assertEqual(report["summary"]["n_sources"], 2)
            self.assertIn("full", report["base_composition"])
            self.assertIn("five_utr", report["base_composition"]["regions"])
            for base in "ACGU":
                self.assertIn(base, report["base_composition"]["full"]["candidate"])
                self.assertIn(base, report["base_composition"]["regions"]["cds"]["source"])
            self.assertGreaterEqual(len(report["length_distribution"]["histogram"]["candidate"]), 2)
            self.assertEqual(report["kmer_spectrum"]["k"], 2)
            self.assertGreater(len(report["kmer_spectrum"]["top_abs_delta"]), 0)
            self.assertIn("figures", report)
            for fig_path in report["figures"].values():
                self.assertTrue(os.path.exists(fig_path))
                with open(fig_path, "r", encoding="utf-8") as fh:
                    self.assertIn("<svg", fh.read())

            out_json = os.path.join(tmp, "spectrum.json")
            out_md = os.path.join(tmp, "spectrum.md")
            multi_scale_sequence_spectrum_audit.write_report_json(report, out_json)
            multi_scale_sequence_spectrum_audit.write_report_markdown(report, out_md)
            with open(out_md, "r", encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn("Base Composition", text)
            self.assertIn("Top k-mer Differences", text)


if __name__ == "__main__":
    unittest.main()
