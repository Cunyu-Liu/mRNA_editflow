"""Shared, frozen numerical settings for MK0 CPU acceptance tests."""

from __future__ import annotations

import itertools
import random

import numpy as np
import pytest


SEED = 20_260_802
FLOAT64_ATOL = 1.0e-10
FLOAT64_RTOL = 1.0e-8


@pytest.fixture(autouse=True)
def _freeze_randomness() -> None:
    """Make every sampled property test independently replayable."""

    random.seed(SEED)
    np.random.seed(SEED)


def rna_sequences(min_length: int = 1, max_length: int = 3):
    """Yield the preregistered exhaustive tiny RNA state domain."""

    alphabet = "ACGU"
    for length in range(min_length, max_length + 1):
        for tokens in itertools.product(alphabet, repeat=length):
            yield "".join(tokens)
