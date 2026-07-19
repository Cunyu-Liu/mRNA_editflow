"""End-to-end offline tests for the mRNA-EditFlow data pipeline.

Uses stdlib :mod:`unittest` only (no pytest). Everything runs on CPU with no
internet by driving the pipeline from :func:`synthesize_corpus`. Run from the
repo root::

    /Users/bytedance/Documents/research/editflow/.venv/bin/python \
        -m unittest mrna_editflow.tests.test_data_pipeline -v
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import tempfile
import unittest
from typing import Dict

from mrna_editflow.core.config import DataConfig
from mrna_editflow.core.constants import (
    NUM_REGIONS,
    PAD_TOKEN,
    is_valid_cds,
)
from mrna_editflow.core.schema import MRNARecord
from mrna_editflow.data.clean_mrna import clean_corpus, clean_record
from mrna_editflow.data.dedup_split import (
    family_disjoint_split,
    read_idx,
)
from mrna_editflow.data.download_mrna import synthesize_corpus
from mrna_editflow.data.download_mrna import write_records_jsonl
from mrna_editflow.data.leakage_audit import audit_leakage, write_leakage_report
from mrna_editflow.data.split_contract import (
    SplitHashError,
    SplitIndexError,
    SplitOverlapError,
    SplitRoleError,
    SplitSchemaError,
    build_split_manifest,
    load_and_verify_split_manifest,
    select_role_records,
    sha256_file,
    transcript_id_digest,
)
from mrna_editflow.data.mrna_dataset import (
    LengthBucketBatchSampler,
    MRNADataset,
    PHASE_PAD,
    REGION_PAD,
    collate_fn,
)
from mrna_editflow.data.precompute_features import (
    load_features,
    precompute_corpus,
    verify_manifest,
)

N = 200
SEED = 20260714


class SplitContractTest(unittest.TestCase):
    def _fixture(self, tmp: str):
        records = synthesize_corpus(12, seed=41)
        records_path = os.path.join(tmp, "records.jsonl")
        write_records_jsonl(records, records_path)
        role_indices = {
            "train": list(range(0, 6)),
            "val": list(range(6, 9)),
            "test": list(range(9, 12)),
        }
        role_paths = {}
        for role, indices in role_indices.items():
            role_path = os.path.join(tmp, f"{role}.idx")
            with open(role_path, "w", encoding="utf-8") as fh:
                fh.write("".join(f"{idx}\n" for idx in indices))
            role_paths[role] = role_path
        leakage_path = os.path.join(tmp, "leakage.json")
        cluster_path = os.path.join(tmp, "cluster_assignments.json")
        assignments = list(range(len(records)))
        with open(cluster_path, "w", encoding="utf-8") as fh:
            json.dump(assignments, fh, separators=(",", ":"))
        with open(leakage_path, "w", encoding="utf-8") as fh:
            json.dump({
                "split": {"cluster_disjoint": True},
                "summary": {
                    "exact_match_count": 0,
                    "leakage_flagged_fraction": 0.0,
                    "near_neighbor_threshold_passed": True,
                },
            }, fh)
        manifest = build_split_manifest(
            dataset_id="synthetic_contract_fixture",
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
        manifest_path = os.path.join(tmp, "split.manifest.json")
        self._write_manifest(manifest_path, manifest)
        return records, records_path, role_paths, manifest, manifest_path

    @staticmethod
    def _write_manifest(path: str, manifest: dict) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)

    def test_valid_manifest_passes_and_selects_deterministically(self):
        with tempfile.TemporaryDirectory(prefix="split_contract_") as tmp:
            records, _records_path, _role_paths, _manifest, manifest_path = self._fixture(tmp)
            contract = load_and_verify_split_manifest(manifest_path)
            selected = select_role_records(records, contract, "train")
            self.assertEqual([row.transcript_id for row in selected], [row.transcript_id for row in records[:6]])
            self.assertTrue(contract.paper_eligible)
            with self.assertRaises(SplitRoleError):
                select_role_records(records, contract, "headline")

    def test_records_and_index_sha_mismatch_fail(self):
        with tempfile.TemporaryDirectory(prefix="split_contract_sha_") as tmp:
            _records, records_path, role_paths, _manifest, manifest_path = self._fixture(tmp)
            with open(records_path, "a", encoding="utf-8") as fh:
                fh.write("\n")
            with self.assertRaises(SplitHashError):
                load_and_verify_split_manifest(manifest_path)
        with tempfile.TemporaryDirectory(prefix="split_contract_idx_sha_") as tmp:
            _records, _records_path, role_paths, _manifest, manifest_path = self._fixture(tmp)
            with open(role_paths["train"], "a", encoding="utf-8") as fh:
                fh.write("\n")
            with self.assertRaises(SplitHashError):
                load_and_verify_split_manifest(manifest_path)

    def test_duplicate_overlap_and_out_of_range_indices_fail(self):
        cases = {
            "duplicate": ("train", [0, 1, 1, 2, 3, 4], SplitIndexError),
            "overlap": ("val", [5, 6, 7], SplitOverlapError),
            "out_of_range": ("test", [9, 10, 99], SplitIndexError),
        }
        for label, (role, indices, error_type) in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory(prefix=f"split_{label}_") as tmp:
                records, _records_path, role_paths, manifest, manifest_path = self._fixture(tmp)
                with open(role_paths[role], "w", encoding="utf-8") as fh:
                    fh.write("".join(f"{idx}\n" for idx in indices))
                role_meta = manifest["split"]["roles"][role]
                role_meta["sha256"] = sha256_file(role_paths[role])
                role_meta["count"] = len(indices)
                valid_ids = [records[idx].transcript_id for idx in indices if idx < len(records)]
                role_meta["selected_id_digest"] = transcript_id_digest(valid_ids)
                self._write_manifest(manifest_path, manifest)
                with self.assertRaises(error_type):
                    load_and_verify_split_manifest(manifest_path)

    def test_changed_record_order_or_identifier_fails(self):
        for mutation in ("order", "identifier"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(prefix="split_record_change_") as tmp:
                records, records_path, _role_paths, manifest, manifest_path = self._fixture(tmp)
                changed = list(records)
                if mutation == "order":
                    changed[0], changed[1] = changed[1], changed[0]
                else:
                    changed[0] = MRNARecord(
                        changed[0].transcript_id + "_changed",
                        changed[0].five_utr,
                        changed[0].cds,
                        changed[0].three_utr,
                        changed[0].species,
                    )
                write_records_jsonl(changed, records_path)
                manifest["records"]["sha256"] = sha256_file(records_path)
                self._write_manifest(manifest_path, manifest)
                with self.assertRaises(SplitHashError):
                    load_and_verify_split_manifest(manifest_path)

    def test_in_memory_record_content_change_fails_selection(self):
        with tempfile.TemporaryDirectory(prefix="split_record_content_") as tmp:
            records, _records_path, _role_paths, _manifest, manifest_path = self._fixture(tmp)
            contract = load_and_verify_split_manifest(manifest_path)
            changed = list(records)
            original = changed[0]
            replacement = "A" if original.five_utr[:1] != "A" else "C"
            changed[0] = MRNARecord(
                original.transcript_id,
                replacement + original.five_utr[1:],
                original.cds,
                original.three_utr,
                original.species,
            )
            with self.assertRaises(SplitHashError):
                select_role_records(changed, contract, "train")

    def test_relative_manifest_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="split_path_escape_") as tmp:
            _records, _records_path, _role_paths, manifest, _manifest_path = self._fixture(tmp)
            nested = os.path.join(tmp, "manifest_dir")
            os.makedirs(nested)
            manifest["records"]["path"] = "../records.jsonl"
            manifest_path = os.path.join(nested, "split.json")
            self._write_manifest(manifest_path, manifest)
            with self.assertRaises(SplitSchemaError):
                load_and_verify_split_manifest(manifest_path)

    def test_cross_role_cluster_overlap_fails(self):
        with tempfile.TemporaryDirectory(prefix="split_cluster_overlap_") as tmp:
            _records, _records_path, _role_paths, manifest, manifest_path = self._fixture(tmp)
            cluster_meta = manifest["split"]["cluster_assignments"]
            cluster_path = cluster_meta["path"]
            assignments = list(range(12))
            assignments[6] = assignments[0]
            with open(cluster_path, "w", encoding="utf-8") as fh:
                json.dump(assignments, fh, separators=(",", ":"))
            cluster_meta["sha256"] = sha256_file(cluster_path)
            cluster_meta["assignment_digest"] = hashlib.sha256(
                json.dumps(assignments, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            self._write_manifest(manifest_path, manifest)
            with self.assertRaises(SplitOverlapError):
                load_and_verify_split_manifest(manifest_path)


class DataPipelineTest(unittest.TestCase):
    """Full corpus -> clean -> split -> features -> dataset flow."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = DataConfig(seed=SEED)
        cls.raw = synthesize_corpus(N, seed=SEED)
        cls.survivors, cls.drop_stats = clean_corpus(cls.raw, cls.cfg)
        cls.tmp = tempfile.mkdtemp(prefix="mef_data_test_")

    # ---- 1. synthesis ----
    def test_synthesis_offline_valid_and_reproducible(self) -> None:
        self.assertEqual(len(self.raw), N)
        self.assertTrue(all(is_valid_cds(r.cds) for r in self.raw))
        again = synthesize_corpus(N, seed=SEED)
        self.assertEqual([r.seq for r in self.raw], [r.seq for r in again])
        # Redundancy families exist (fewer families than records).
        fams = {r.transcript_id.rsplit("_", 1)[0] for r in self.raw}
        self.assertLess(len(fams), N)

    # ---- 2. cleaning ----
    def test_clean_survivors_valid_and_within_caps(self) -> None:
        self.assertGreater(len(self.survivors), 0)
        for r in self.survivors:
            self.assertTrue(is_valid_cds(r.cds))
            self.assertLessEqual(len(r.five_utr), self.cfg.max_5utr)
            self.assertLessEqual(len(r.cds), self.cfg.max_cds)
            self.assertLessEqual(len(r.three_utr), self.cfg.max_3utr)
            self.assertTrue(set(r.seq) <= set("ACGU"))

    def test_clean_drop_stats_accounting(self) -> None:
        s = self.drop_stats
        self.assertEqual(s["total"], N)
        self.assertEqual(s["kept"], len(self.survivors))
        drops = sum(s[k] for k in s if k not in ("total", "kept", "truncated_5utr", "truncated_3utr"))
        self.assertEqual(s["kept"] + drops, N)

    def test_clean_rejects_malformed(self) -> None:
        from mrna_editflow.core.schema import MRNARecord

        # Illegal char, bad start, internal stop, frame error, over-long CDS.
        self.assertIsNone(clean_record(MRNARecord("x", "ACGN", "AUGAAAUAA", "")))
        self.assertIsNone(clean_record(MRNARecord("x", "", "CCCAAAUAA", "")))
        self.assertIsNone(clean_record(MRNARecord("x", "", "AUGUAAAAAUAA", "")))
        self.assertIsNone(clean_record(MRNARecord("x", "", "AUGAAUA", "")))
        toolong = "AUG" + "AAA" * (self.cfg.max_cds) + "UAA"
        self.assertIsNone(clean_record(MRNARecord("x", "", toolong, "")))
        # DNA T is normalised to U and accepted.
        ok = clean_record(MRNARecord("x", "ACGT", "ATGAAATAA", "ACGT"))
        self.assertIsNotNone(ok)
        self.assertTrue(is_valid_cds(ok.cds))
        self.assertNotIn("T", ok.seq)

    def test_clean_truncation_policy(self) -> None:
        from mrna_editflow.core.schema import MRNARecord

        rec = MRNARecord("t", "A" * (self.cfg.max_5utr + 40), "AUGAAAUAA",
                         "C" * (self.cfg.max_3utr + 40))
        cleaned = clean_record(rec, self.cfg)
        self.assertIsNotNone(cleaned)
        self.assertEqual(len(cleaned.five_utr), self.cfg.max_5utr)  # 5' truncated
        self.assertEqual(len(cleaned.three_utr), self.cfg.max_3utr)  # 3' truncated

    # ---- 3. dedup / split ----
    def test_split_family_disjoint_and_fractions(self) -> None:
        split_dir = os.path.join(self.tmp, "splits")
        res = family_disjoint_split(
            self.survivors, self.cfg, out_dir=split_dir,
            use_mmseqs="never", write=True,
        )
        n = len(self.survivors)
        s_train, s_val, s_test = set(res.train), set(res.val), set(res.test)
        # Hard requirement: train/test cluster-disjoint.
        self.assertEqual(len(s_train & s_test), 0)
        self.assertEqual(len(s_train & s_val), 0)
        self.assertEqual(len(s_val & s_test), 0)
        self.assertEqual(len(s_train) + len(s_val) + len(s_test), n)
        # No cluster (family) straddles splits.
        for cl in res.clusters:
            cset = set(cl)
            placed = (bool(cset & s_train) + bool(cset & s_val) + bool(cset & s_test))
            self.assertLessEqual(placed, 1)
        # Fractions approximately match the config (loose tolerance; whole
        # clusters are indivisible so exactness is impossible in general).
        self.assertAlmostEqual(len(s_train) / n, self.cfg.train_frac, delta=0.12)
        self.assertAlmostEqual(len(s_val) / n, self.cfg.val_frac, delta=0.12)
        self.assertAlmostEqual(len(s_test) / n, self.cfg.test_frac, delta=0.12)
        # .idx files written and readable.
        for name in ("train", "val", "test"):
            path = os.path.join(split_dir, f"{name}.idx")
            self.assertTrue(os.path.exists(path))
        self.assertEqual(sorted(read_idx(os.path.join(split_dir, "train.idx"))), sorted(res.train))

    def test_split_reproducible(self) -> None:
        a = family_disjoint_split(self.survivors, self.cfg, use_mmseqs="never", write=False)
        b = family_disjoint_split(self.survivors, self.cfg, use_mmseqs="never", write=False)
        self.assertEqual(a.train, b.train)
        self.assertEqual(a.val, b.val)
        self.assertEqual(a.test, b.test)

    def test_kmer_leakage_audit_flags_exact_and_contained_queries(self) -> None:
        refs = [
            self.survivors[0],
            self.survivors[1],
            self.survivors[2],
        ]
        query_exact = refs[0]
        query_contained = MRNARecord(
            "contained",
            refs[1].five_utr[:20],
            refs[1].cds,
            refs[1].three_utr[:20],
        )
        query_clean = MRNARecord("clean", "A" * 20, "AUGAAAUAA", "U" * 20)
        result = audit_leakage(
            [query_exact, query_contained, query_clean],
            refs,
            k=9,
            top_k=2,
            jaccard_threshold=0.80,
            containment_threshold=0.80,
        )
        self.assertEqual(result["summary"]["n_query"], 3)
        self.assertGreaterEqual(result["summary"]["flagged_count"], 2)
        rows = {row["query_id"]: row for row in result["per_query"]}
        self.assertTrue(rows[query_exact.transcript_id]["exact_sequence_match"])
        self.assertTrue(rows["contained"]["flagged"])
        self.assertFalse(rows["clean"]["flagged"])

        out_json = os.path.join(self.tmp, "leakage.json")
        out_md = os.path.join(self.tmp, "leakage.md")
        write_leakage_report(result, out_json, out_md)
        self.assertTrue(os.path.exists(out_json))
        self.assertTrue(os.path.exists(out_md))
        with open(out_md, "r", encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn("mRNA-EditFlow Leakage Audit", text)
        self.assertIn("flagged_fraction", text)

    # ---- 4. precomputed features ----
    def test_precompute_features_and_manifest(self) -> None:
        feat_dir = os.path.join(self.tmp, "features")
        manifest = precompute_corpus(
            self.survivors, feat_dir, self.cfg, drop_stats=self.drop_stats,
            shard_size=64,
        )
        self.assertTrue(os.path.exists(manifest))
        self.assertTrue(verify_manifest(feat_dir))

        with open(manifest, "r", encoding="utf-8") as fh:
            lines = [json.loads(x) for x in fh if x.strip()]
        summary = lines[0]
        shards = lines[1:]
        self.assertEqual(summary["type"], "summary")
        self.assertEqual(summary["n_total"], len(self.survivors))
        self.assertEqual(summary["n_shards"], len(shards))
        self.assertIn("region_fractions", summary)
        self.assertIsNotNone(summary["cleaning_drop_counts"])
        # Each shard records a 64-hex SHA256, positive n_seqs, and dists.
        for sh in shards:
            self.assertEqual(sh["type"], "shard")
            self.assertEqual(len(sh["sha256"]), 64)
            self.assertGreater(sh["n_seqs"], 0)
            self.assertIn("mfe", sh)
            self.assertIn("start_accessibility", sh)
            self.assertIn("length_distribution", sh)

        feats = load_features(feat_dir)
        self.assertEqual(len(feats), len(self.survivors))
        by_id: Dict[str, object] = {r.transcript_id: r for r in self.survivors}
        for tid, f in feats.items():
            self.assertTrue(math.isfinite(f.mfe))
            self.assertTrue(math.isfinite(f.start_accessibility))
            self.assertGreaterEqual(f.start_accessibility, 0.0)
            self.assertLessEqual(f.start_accessibility, 1.0)
            # pairing_prob aligned to transcript length.
            self.assertEqual(len(f.pairing_prob), len(by_id[tid].seq))

    # ---- 5. dataset + collate ----
    def test_dataset_collate_shapes_and_alignment(self) -> None:
        feat_dir = os.path.join(self.tmp, "features_ds")
        precompute_corpus(self.survivors, feat_dir, self.cfg, shard_size=64)
        feats = load_features(feat_dir)
        ds = MRNADataset(self.survivors, feats)
        self.assertEqual(len(ds), len(self.survivors))

        # Per-item region/phase alignment length == token length.
        item = ds[0]
        L = item["token_ids"].shape[0]
        self.assertEqual(item["region_ids"].shape[0], L)
        self.assertEqual(item["phase_ids"].shape[0], L)
        self.assertEqual(item["pairing_prob"].shape[0], L)
        self.assertEqual(item["length"], L)

        sampler = LengthBucketBatchSampler(ds, batch_size=16, cfg=self.cfg, shuffle=True)
        seen = 0
        for batch_indices in sampler:
            batch = collate_fn([ds[i] for i in batch_indices])
            B, Lmax = batch["token_ids"].shape
            seen += B
            # All per-position tensors share (B, Lmax).
            self.assertEqual(batch["region_ids"].shape, (B, Lmax))
            self.assertEqual(batch["phase_ids"].shape, (B, Lmax))
            self.assertEqual(batch["pairing_prob"].shape, (B, Lmax))
            self.assertEqual(batch["padding_mask"].shape, (B, Lmax))
            self.assertEqual(batch["mfe"].shape, (B,))
            self.assertEqual(batch["start_accessibility"].shape, (B,))
            for r in range(B):
                Lr = int(batch["lengths"][r])
                # Real region: mask False; token != PAD.
                self.assertTrue(bool((~batch["padding_mask"][r, :Lr]).all()))
                # Region/phase ids valid in the real region.
                self.assertTrue(bool((batch["region_ids"][r, :Lr] < NUM_REGIONS).all()))
                # Pad region: mask True and PAD sentinels.
                if Lr < Lmax:
                    self.assertTrue(bool(batch["padding_mask"][r, Lr:].all()))
                    self.assertEqual(int(batch["token_ids"][r, Lr]), PAD_TOKEN)
                    self.assertEqual(int(batch["region_ids"][r, Lr]), REGION_PAD)
                    self.assertEqual(int(batch["phase_ids"][r, Lr]), PHASE_PAD)
        self.assertEqual(seen, len(ds))  # every record batched exactly once


if __name__ == "__main__":
    unittest.main(verbosity=2)
