"""Tests for deterministic Stage 6 development evaluation asset selection."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_PARENT = _ROOT.parent
for _path in (str(_PARENT), str(_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from mrna_editflow.eval.build_rl_stage6_assets import select_extreme_indices, select_ood_family_indices


class Stage6AssetSelectionTests(unittest.TestCase):
    def test_ood_selection_keeps_entire_family(self) -> None:
        selected = select_ood_family_indices([0, 1, 2, 3, 4], [10, 10, 11, 12, 12], max_records=3, seed=7)
        self.assertTrue(selected)
        for family in ({0, 1}, {2}, {3, 4}):
            self.assertFalse(bool(family & set(selected)) and bool(family - set(selected)))

    def test_extreme_selection_is_deterministic(self) -> None:
        values = {0: 0.1, 1: 0.8, 2: 0.8, 3: 0.2}
        self.assertEqual(select_extreme_indices(values, largest=True, max_records=2), [2, 1])
        self.assertEqual(select_extreme_indices(values, largest=False, max_records=2), [0, 3])

    def test_ood_without_cluster_vector_is_a_declared_hash_slice(self) -> None:
        selected = select_ood_family_indices([0, 1, 2, 3], None, max_records=2, seed=7)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected, select_ood_family_indices([0, 1, 2, 3], None, max_records=2, seed=7))


if __name__ == "__main__":
    unittest.main()
