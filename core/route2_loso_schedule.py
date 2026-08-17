"""Shared six-GPU assignment for matched Route 2 LOSO model/baseline pairs."""

from __future__ import annotations


HOLDOUT_STUDIES = (
    "GSE200304",
    "GSE114002",
    "GSE149487",
    "GSE217518",
    "ENCSR854RUF",
    "GSE186455",
    "GSE269595",
)
FINAL_SEEDS = (20260822, 20260823, 20260824)
PHYSICAL_GPU_INDICES = (0, 1, 2, 3, 4, 5)


def assigned_gpu(study: str, seed: int) -> int:
    study_index = HOLDOUT_STUDIES.index(study)
    seed_index = FINAL_SEEDS.index(seed)
    task_index = study_index * len(FINAL_SEEDS) + seed_index
    return PHYSICAL_GPU_INDICES[task_index % len(PHYSICAL_GPU_INDICES)]


def loso_assignments() -> tuple[tuple[str, int, int], ...]:
    return tuple(
        (study, seed, assigned_gpu(study, seed))
        for study in HOLDOUT_STUDIES
        for seed in FINAL_SEEDS
    )
