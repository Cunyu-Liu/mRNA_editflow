"""Central configuration dataclasses for mRNA-EditFlow (MEF).

A single, explicit config contract shared by the data pipeline, model, training
and evaluation code. Every ablation switch mentioned in the spec is a field
here so experiments are fully described by one serialisable object.

No heavy dependencies: pure dataclasses + optional JSON (de)serialisation.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class DataConfig:
    """Corpus construction + length-bucketing parameters."""
    data_dir: str = "./data"
    max_5utr: int = 128
    max_cds: int = 1536
    max_3utr: int = 256
    length_buckets: tuple = (256, 512, 1024, 2048)
    mmseqs_min_seq_id: float = 0.8
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1
    seed: int = 20260714


@dataclass
class BackboneConfig:
    """Frozen mRNA-native foundation-model encoder settings.

    ``name`` selects the encoder. First-line mRNA-native backbones are the
    default; ncRNA backbones are negative controls; ``none`` falls back to a
    from-scratch light embedding (also used for offline smoke tests).
    """
    name: str = "none"  # {mrnabert, helix_mrna, orthrus, orthrus_mlm, lamar,
                        #  rna_fm, rinalmo, none}
    hidden_dim: int = 256          # backbone output width (projected to model dim)
    freeze: bool = True            # requires_grad=False on all backbone params
    cache_embeddings: bool = True  # offline per-token embedding cache for Stage A
    weights_path: Optional[str] = None
    # Token granularity of the backbone; used to align to nt positions.
    # {nt, dual, codon}
    granularity: str = "nt"


@dataclass
class ModelConfig:
    """MEF generation-head architecture + ablation switches."""
    model_dim: int = 384
    num_layers: int = 6
    num_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.1
    max_seq_len: int = 2048
    # --- ablation switches (each independently toggleable) ---
    use_region_film: bool = True       # region-aware FiLM modulation
    use_codon_constraint: bool = True  # codon-lattice constrained CDS operators
    codon_indel: bool = False          # allow frame-safe whole-codon indels in CDS
    use_rope: bool = True              # rotary vs absolute positions
    # Dormant until an explicit, shape-checked target artifact with provenance
    # is supplied.  An implicit all-zero tensor is never a biological target.
    use_aux_struct: bool = False       # experimental MFE/accessibility head
    aux_loss_weight: float = 0.0


@dataclass
class CouplingConfig:
    """Three-way hybrid coupling mixture weights (must sum > 0)."""
    empty_prob: float = 0.5        # empty-growth coupling
    corruption_prob: float = 0.3   # corruption-refinement coupling
    ortholog_prob: float = 0.2     # evolution-aware ortholog coupling
    # corruption operation rates (refinement mode)
    sub_prob: float = 0.2
    ins_prob: float = 0.1
    del_prob: float = 0.1
    # bounds for random x0 length in growth mode
    min_x0_len: int = 8
    max_x0_len: int = 64


@dataclass
class TrainConfig:
    """Optimisation + numerical-stability guards."""
    epochs: int = 100
    batch_size: int = 32
    grad_accum: int = 1
    lr: float = 1e-4
    weight_decay: float = 0.0
    amp: bool = True               # mixed precision
    grad_clip: float = 1.0
    oom_batch_ladder: tuple = (32, 16, 8, 4, 2, 1)  # step-down on OOM
    nan_retry: int = 3
    save_dir: str = "./ckpts"
    profile_path: str = "./profile.jsonl"
    log_every: int = 20


@dataclass
class MEFConfig:
    """Top-level config aggregating all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    coupling: CouplingConfig = field(default_factory=CouplingConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "MEFConfig":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(
            data=DataConfig(**data.get("data", {})),
            backbone=BackboneConfig(**data.get("backbone", {})),
            model=ModelConfig(**data.get("model", {})),
            coupling=CouplingConfig(**data.get("coupling", {})),
            train=TrainConfig(**data.get("train", {})),
        )


__all__ = [
    "DataConfig", "BackboneConfig", "ModelConfig",
    "CouplingConfig", "TrainConfig", "MEFConfig",
]
