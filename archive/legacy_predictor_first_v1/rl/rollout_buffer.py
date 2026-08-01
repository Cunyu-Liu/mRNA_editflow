"""Versioned replay-buffer helpers for offline DAgger ranker distillation."""
from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional, Sequence


REPLAY_BUCKETS: tuple[str, ...] = ("original", "rollout", "hard_negative", "stop")


@dataclass(frozen=True)
class ReplayMixConfig:
    """Exact row-level sampling proportions for the four required replay pools."""

    original: float = 0.40
    rollout: float = 0.40
    hard_negative: float = 0.10
    stop: float = 0.10

    def normalized(self) -> dict[str, float]:
        values = {key: max(0.0, float(getattr(self, key))) for key in REPLAY_BUCKETS}
        total = sum(values.values())
        if total <= 0.0:
            raise ValueError("at least one replay mixing proportion must be positive")
        return {key: value / total for key, value in values.items()}

    def quotas(self, total: int) -> dict[str, int]:
        """Largest-remainder quotas that sum exactly to ``total``."""
        n = max(0, int(total))
        weights = self.normalized()
        raw = {key: n * value for key, value in weights.items()}
        quotas = {key: int(value) for key, value in raw.items()}
        remaining = n - sum(quotas.values())
        for key in sorted(REPLAY_BUCKETS, key=lambda item: (-(raw[item] - quotas[item]), item))[:remaining]:
            quotas[key] += 1
        return quotas


@dataclass
class ReplayBuffer:
    """In-memory, serializable buffer; rows retain their source bucket."""

    policy_checkpoint_sha256: str
    iteration: int
    rows: list[dict[str, object]] = field(default_factory=list)

    def add(self, rows: Sequence[Mapping[str, object]], *, bucket: str) -> None:
        if bucket not in REPLAY_BUCKETS:
            raise ValueError(f"unknown replay bucket {bucket!r}")
        for raw in rows:
            row = dict(raw)
            row["replay_bucket"] = bucket
            row.setdefault("policy_checkpoint_sha256", self.policy_checkpoint_sha256)
            row.setdefault("dagger_iteration", int(self.iteration))
            self.rows.append(row)

    def bucket_counts(self) -> dict[str, int]:
        return {
            key: sum(1 for row in self.rows if row.get("replay_bucket") == key)
            for key in REPLAY_BUCKETS
        }

    def sample_mixed(
        self,
        total: int,
        *,
        config: Optional[ReplayMixConfig] = None,
        seed: int = 0,
    ) -> list[dict[str, object]]:
        """Sample exact configured bucket quotas without replacement.

        Failing when a required pool is too small is intentional: silently
        changing the mixture would invalidate the iteration manifest.
        """
        mix = config or ReplayMixConfig()
        quotas = mix.quotas(total)
        rng = random.Random(int(seed))
        selected: list[dict[str, object]] = []
        for bucket in REPLAY_BUCKETS:
            pool = [row for row in self.rows if row.get("replay_bucket") == bucket]
            quota = quotas[bucket]
            if len(pool) < quota:
                raise ValueError(
                    f"replay bucket {bucket!r} has {len(pool)} rows but requires {quota}; "
                    "do not silently alter configured proportions"
                )
            indices = list(range(len(pool)))
            rng.shuffle(indices)
            selected.extend(dict(pool[index]) for index in indices[:quota])
        rng.shuffle(selected)
        return selected

    def manifest(self) -> dict[str, object]:
        state_sequences = {
            str(row["state_sequence"])
            for row in self.rows if isinstance(row.get("state_sequence"), str)
        }
        return {
            "dagger_iteration": int(self.iteration),
            "policy_checkpoint_sha256": self.policy_checkpoint_sha256,
            "buffer_size": len(self.rows),
            "bucket_counts": self.bucket_counts(),
            "state_diversity": len(state_sequences),
        }

    def write_jsonl(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for row in self.rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

    def write_manifest(self, path: str, *, extra: Optional[Mapping[str, object]] = None) -> dict[str, object]:
        payload = self.manifest()
        if extra:
            payload.update(dict(extra))
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        return payload


def iteration_directory(root: str, iteration: int, *, create: bool = True) -> str:
    """Create one non-overwritable iteration directory."""
    path = os.path.join(root, f"iteration_{int(iteration):03d}")
    if os.path.exists(path):
        raise FileExistsError(f"DAgger iteration artifact already exists: {path}")
    if create:
        os.makedirs(path, exist_ok=False)
    return path


__all__ = ["REPLAY_BUCKETS", "ReplayMixConfig", "ReplayBuffer", "iteration_directory"]
