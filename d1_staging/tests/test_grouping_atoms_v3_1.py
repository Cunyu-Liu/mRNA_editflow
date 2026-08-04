from pathlib import Path
import sys

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "scripts"))

from grouping_atoms_v3_1 import (  # noqa: E402
    ASSIGNMENT_ALGORITHM_ID,
    derive_grouping_atoms,
    group_id_for,
)


def test_projection_is_provenance_bound_and_does_not_invent_missing_atoms():
    rec = {
        "accession": "GSE114002",
        "metadata": {"source_file": "designed.csv.gz", "library": "L1"},
    }
    atoms = derive_grouping_atoms(
        rec, "r__src", "r__cand", "AAAA", "AAAT", "PAIR", "pair-1",
    )
    assert "STUDY" in atoms
    assert "LIBRARY_LINEAGE" in atoms
    assert "PAIR" in atoms
    assert "SOURCE" in atoms
    assert "SEQUENCE_CLUSTER" in atoms
    assert "GENE" not in atoms
    assert "TRANSCRIPT" not in atoms
    assert "TILE_FAMILY" not in atoms
    assert "BIOLOGICAL_PARENT" not in atoms
    assert "SEQUENCE" not in atoms


def test_explicit_parent_and_coordinate_tile_are_materialized():
    rec = {
        "accession": "GSE149487",
        "metadata": {
            "gene": "GENE1",
            "source_file": "moesm8.xlsx",
            "wt_id": "WT1",
            "chrom": "chr1",
            "pos_start": 10,
            "pos_end": 30,
            "enst": "ENST1",
        },
    }
    atoms = derive_grouping_atoms(
        rec, "r__src", "r__cand", "AAAA", "AAAT", "PAIR", "pair-1",
    )
    assert atoms["GENE"] == ["gene:GENE1"]
    assert atoms["TRANSCRIPT"] == ["enst:ENST1"]
    assert atoms["BIOLOGICAL_PARENT"] == ["wt_id:WT1"]
    assert len(atoms["TILE_FAMILY"]) == 1


def test_observation_scope_does_not_emit_pair_or_source_atoms():
    rec = {
        "accession": "GSE145046",
        "metadata": {"source_file": "random.txt.gz", "gene_symbol": "G"},
    }
    atoms = derive_grouping_atoms(
        rec, "r__src", "r__cand", "AAAA", "AAAT", "OBSERVATION", "obs-1",
        context_id="ctx_gse145046",
    )
    assert "PAIR" not in atoms
    assert "SOURCE" not in atoms
    assert "BIOLOGICAL_PARENT" not in atoms
    assert atoms["CONTEXT"] == ["ctx_gse145046"]
    assert atoms["GENE"] == ["gene_symbol:G"]


def test_group_id_algorithm_is_stable():
    assert group_id_for("STUDY", "accession:gse114002") == group_id_for(
        "STUDY", "accession:gse114002"
    )
    assert ASSIGNMENT_ALGORITHM_ID == "D1_PROVENANCE_ATOM_PROJECTION_V1"
