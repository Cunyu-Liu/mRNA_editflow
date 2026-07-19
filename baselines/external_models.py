"""Offline external-model reference records for mRNA-EditFlow baselines.

The project needs to compare against published systems such as UTRGAN,
Optimus+GA, mRNABERT, Helix-mRNA, Orthrus+MLM, LAMAR, mRNAutilus and Prot2RNA.
Those integrations require external weights, private preprocessing scripts or
online downloads that are deliberately unavailable in the local smoke
environment. This module therefore provides a strict offline contract:

* no network calls,
* no optional third-party imports,
* explicit protocol-difference notes,
* structured ``ExternalResult`` records suitable for an ablation ledger.

Complexity: registry lookup is ``O(1)`` by normalized model name; listing all
references is ``O(N)`` for ``N`` registered models.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional


@dataclass(frozen=True)
class ExternalModelSpec:
    """Static metadata for an external model reference.

    Complexity: construction and access are ``O(1)``.
    """

    name: str
    citation: str
    family: str
    expected_inputs: str
    expected_outputs: str
    protocol_difference: str
    offline_note: str


@dataclass(frozen=True)
class ExternalResult:
    """Structured result emitted when an external baseline is unavailable.

    ``status`` is intentionally explicit; offline smoke tests should see
    ``"offline_placeholder"`` rather than a fake metric. ``protocol_difference``
    records why the result is not a like-for-like local score.

    Complexity: conversion to dict is ``O(number_of_fields)``.
    """

    model_name: str
    status: str
    offline: bool
    citation: str
    family: str
    expected_inputs: str
    expected_outputs: str
    protocol_difference: str
    notes: str
    metrics: Mapping[str, float]

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable mapping. Complexity: ``O(fields)``."""
        return dict(asdict(self))


def _key(name: str) -> str:
    """Normalize external-model names for lookup. Complexity: ``O(len(name))``."""
    return (
        name.lower()
        .replace("+", "plus")
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


EXTERNAL_MODEL_REGISTRY: Dict[str, ExternalModelSpec] = {}


def _register(spec: ExternalModelSpec) -> None:
    EXTERNAL_MODEL_REGISTRY[_key(spec.name)] = spec


_register(
    ExternalModelSpec(
        name="UTRGAN",
        citation="UTRGAN: GAN-based 5'UTR generation baseline for expression-control protocols.",
        family="generative 5'UTR GAN",
        expected_inputs="5'UTR sequence corpus and expression labels, often task-specific.",
        expected_outputs="synthetic 5'UTR sequences or expression-optimized UTR candidates.",
        protocol_difference=(
            "UTRGAN optimizes/generates UTRs rather than full 5'UTR-CDS-3'UTR transcripts; "
            "it has no codon-frame edit operator and is not evaluated on variable-length edit tasks."
        ),
        offline_note="No public checkpoint is bundled; local run records only the comparison protocol.",
    )
)
_register(
    ExternalModelSpec(
        name="Optimus+GA",
        citation="Sample et al., Human 5'UTR design and Optimus predictor/GA-style optimization, Nature Biotechnology 2019.",
        family="predictor-guided evolutionary optimization",
        expected_inputs="candidate 5'UTR sequences and a trained expression predictor.",
        expected_outputs="optimized fixed-region UTR candidates selected by a genetic algorithm.",
        protocol_difference=(
            "Optimus+GA is predictor-guided search over UTR strings, not a likelihood model over "
            "full mRNA transcripts; edits are GA mutations without mRNA region/codon constraints."
        ),
        offline_note="External predictor weights are absent; no online model fetch is attempted.",
    )
)
_register(
    ExternalModelSpec(
        name="mRNABERT",
        citation="mRNABERT-style mRNA-native masked-language encoder baseline for representation probing.",
        family="mRNA foundation encoder",
        expected_inputs="tokenized mRNA sequences at the model's native granularity.",
        expected_outputs="per-token or pooled embeddings for downstream probes.",
        protocol_difference=(
            "mRNABERT is primarily an encoder/probe baseline; generation requires an added head "
            "or decoding protocol, so local comparisons should separate representation quality from generation."
        ),
        offline_note="Weights and tokenizer are not bundled; use FrozenBackbone fallback for offline shape tests.",
    )
)
_register(
    ExternalModelSpec(
        name="Helix-mRNA",
        citation="Wood et al., Helix-mRNA: A Hybrid Foundation Model For Full Sequence mRNA Therapeutics, arXiv:2502.13785, 2025.",
        family="mRNA foundation encoder",
        expected_inputs="mRNA sequences with model-specific tokenizer and checkpoint.",
        expected_outputs="contextual mRNA embeddings or task predictions.",
        protocol_difference=(
            "Helix-mRNA supplies representations rather than Edit-Flow-style variable-length edit rates; "
            "a fair run needs a frozen-backbone probe or adapter protocol."
        ),
        offline_note="No external dependency is imported in smoke mode.",
    )
)
_register(
    ExternalModelSpec(
        name="Orthrus+MLM",
        citation="Orthrus/Orthrus-MLM pretrained RNA/mRNA masked-language-model reference.",
        family="masked-language encoder",
        expected_inputs="model-tokenized RNA/mRNA sequences.",
        expected_outputs="MLM logits and contextual embeddings.",
        protocol_difference=(
            "Orthrus+MLM is a denoising/representation model; it lacks explicit insert/delete dynamics "
            "and should be compared either as a masked-diffusion-style baseline or frozen encoder."
        ),
        offline_note="Checkpoint loading is intentionally deferred to an online/full experiment environment.",
    )
)
_register(
    ExternalModelSpec(
        name="LAMAR",
        citation="LAMAR large mRNA/RNA language-model reference for scale-control comparisons.",
        family="large sequence model",
        expected_inputs="model-specific mRNA/RNA tokenization and checkpoint files.",
        expected_outputs="embeddings, logits or downstream task predictions depending on release.",
        protocol_difference=(
            "LAMAR scale and tokenizer may differ from the small offline MEF smoke setting; "
            "report parameter count, tokenization and train/freeze protocol before metric comparison."
        ),
        offline_note="Only protocol metadata is available locally.",
    )
)
_register(
    ExternalModelSpec(
        name="mRNAutilus",
        citation="mRNAutilus mRNA design and optimization toolkit reference.",
        family="mRNA optimization toolkit",
        expected_inputs="design objective, coding sequence or transcript constraints.",
        expected_outputs="optimized mRNA designs under toolkit-specific objective functions.",
        protocol_difference=(
            "mRNAutilus is a design pipeline/toolkit rather than a single trainable generator; "
            "its objective and constraints must be aligned to each MEF task before reporting metrics."
        ),
        offline_note="No toolkit execution is attempted in local tests.",
    )
)
_register(
    ExternalModelSpec(
        name="Prot2RNA",
        citation="Martinovic et al., Prot2RNA: A Diffusion Language Model for Protein-Conditioned mRNA Coding Sequence Generation, OpenReview ICLR 2026 submission.",
        family="conditional codon/mRNA generator",
        expected_inputs="protein sequence or amino-acid target, often with codon-usage constraints.",
        expected_outputs="coding RNA sequence candidates.",
        protocol_difference=(
            "Prot2RNA is protein/CDS-conditioned and does not directly model UTR editing or full-transcript "
            "variable-length tasks; comparisons should isolate CDS design from full mRNA generation."
        ),
        offline_note="Weights and preprocessing are external to this repository.",
    )
)
_register(
    ExternalModelSpec(
        name="mRNA-LM",
        citation="Li et al., mRNA-LM: full-length integrated SLM for mRNA analysis, Nucleic Acids Research 53(3):gkaf044, 2025.",
        family="full-length integrated mRNA language model",
        expected_inputs="full mRNA split into 5'UTR/CDS/3'UTR model-specific segment inputs.",
        expected_outputs="full-transcript embeddings or downstream task predictions.",
        protocol_difference=(
            "mRNA-LM is an analysis/selection model, not a CTMC edit generator; comparisons should "
            "report predictor accuracy separately from variable-length generation quality."
        ),
        offline_note="No model artifact is bundled; use as an external SOTA target in full experiments.",
    )
)
_register(
    ExternalModelSpec(
        name="codonGPT",
        citation="Rajbanshi and Guruacharya, codonGPT: reinforcement learning on a generative language model enables scalable mRNA design, Nucleic Acids Research 53(22):gkaf1345, 2025.",
        family="codon-level autoregressive generator plus RL optimizer",
        expected_inputs="protein/CDS target and biological constraints such as expression, stability and GC content.",
        expected_outputs="optimized synonymous coding mRNA sequences.",
        protocol_difference=(
            "codonGPT focuses on CDS-constrained synonymous generation; it does not jointly edit 5'UTR, "
            "CDS and 3'UTR under one region-aware variable-length process."
        ),
        offline_note=(
            "The official naniltx/codonGPT Hugging Face pretrained checkpoint "
            "is integrated; task-specific RL policy and reward-training "
            "artifacts remain unavailable."
        ),
    )
)
_register(
    ExternalModelSpec(
        name="CodonFM",
        citation=(
            "NVIDIA Digital Bio, CodonFM open RNA foundation model announcement and "
            "model release, 2025."
        ),
        family="codon-level RNA foundation encoder",
        expected_inputs="protein-coding RNA represented as codon tokens with model-specific context windows.",
        expected_outputs="codon-aware embeddings and downstream property predictions such as stability or protein yield.",
        protocol_difference=(
            "CodonFM is a bidirectional foundation encoder rather than a constrained edit generator; "
            "a fair MEF comparison needs frozen embeddings, leakage audits and a shared downstream head."
        ),
        offline_note="Checkpoint and tokenizer downloads are external; no network fetch is attempted in local tests.",
    )
)
_register(
    ExternalModelSpec(
        name="LinearDesign",
        citation="Zhang et al., Algorithm for optimized mRNA design improves stability and immunogenicity, Nature 621:396-403, 2023.",
        family="dynamic-programming/lattice parsing mRNA optimizer",
        expected_inputs="target protein sequence and codon-usage/structure objective weights.",
        expected_outputs="CDS design co-optimized for folding stability and codon usage.",
        protocol_difference=(
            "LinearDesign is a deterministic CDS optimizer with strong biological validation; it is "
            "not trained as a stochastic full-transcript edit model and does not optimize UTR edits."
        ),
        offline_note="Use released software in the server benchmark; no local binary is bundled.",
    )
)
_register(
    ExternalModelSpec(
        name="EnsembleDesign",
        citation="Dai et al., EnsembleDesign: messenger RNA design minimizing ensemble free energy via probabilistic lattice parsing, Bioinformatics 41(Suppl.1):i391-i400, 2025.",
        family="probabilistic-lattice structure optimizer",
        expected_inputs="target protein/codon lattice and ensemble free-energy objective.",
        expected_outputs="CDS designs minimizing ensemble free energy under codon constraints.",
        protocol_difference=(
            "EnsembleDesign targets structure/codon optimization in the protein-conditioned CDS lattice; "
            "MEF should compare against it on CDS structure objectives, not full UTR editing tasks."
        ),
        offline_note="External implementation is not bundled in the smoke environment.",
    )
)
_register(
    ExternalModelSpec(
        name="UTailoR",
        citation="Liu et al., Enhancing mRNA translation efficiency with discriminative and generative AI by optimizing 5' UTR sequences, iScience 28:113544, 2025.",
        family="5'UTR discriminative-plus-generative optimizer",
        expected_inputs="5'UTR sequence and target mRNA context.",
        expected_outputs="optimized 5'UTR candidates predicted and experimentally tested for translation efficiency.",
        protocol_difference=(
            "UTailoR is focused on 5'UTR optimization; it does not preserve or edit CDS/3'UTR with an "
            "explicit CTMC grammar."
        ),
        offline_note="Online service/model artifacts are external; local code records protocol only.",
    )
)
_register(
    ExternalModelSpec(
        name="mRNA2vec",
        citation="Zhang et al., mRNA2vec: mRNA Embedding with Language Model in the 5'UTR-CDS for mRNA Design, arXiv:2408.09048, 2024.",
        family="self-supervised mRNA representation model",
        expected_inputs="5'UTR plus CDS sequence with mRNA-specific masking and auxiliary structure labels.",
        expected_outputs="embeddings for TE, expression, stability and protein-production prediction.",
        protocol_difference=(
            "mRNA2vec is a representation/prediction baseline; generation requires an added optimizer "
            "or decoder protocol."
        ),
        offline_note="Use as a predictor/probe baseline when checkpoints are installed.",
    )
)
_register(
    ExternalModelSpec(
        name="StructmRNA",
        citation="Nahali et al., StructmRNA: a BERT based model with dual level and conditional masking for mRNA representation, Scientific Reports 14:26043, 2024.",
        family="sequence-structure BERT representation model",
        expected_inputs="mRNA sequences with optional structural context.",
        expected_outputs="embeddings and degradation/structure-related predictions.",
        protocol_difference=(
            "StructmRNA evaluates representation quality and RNA degradation tasks rather than "
            "full-transcript constrained generation."
        ),
        offline_note="No checkpoint is bundled; include in external representation comparisons.",
    )
)


def available_external_models() -> List[str]:
    """Return registered external-model display names. Complexity: ``O(N)``."""
    return [spec.name for spec in EXTERNAL_MODEL_REGISTRY.values()]


def get_external_result(
    model_name: str,
    task_id: str = "T1",
    offline: bool = True,
    extra_note: Optional[str] = None,
) -> ExternalResult:
    """Return an offline placeholder/protocol record for one external model.

    The function never performs network access. ``offline=False`` is accepted so
    callers can keep a stable API, but this scaffold still reports
    ``offline_placeholder`` until a real adapter is implemented.

    Complexity: ``O(1)`` registry lookup plus ``O(len(extra_note))`` string work.
    """
    normalized = _key(model_name)
    if normalized not in EXTERNAL_MODEL_REGISTRY:
        raise ValueError(
            f"unknown external model {model_name!r}; available: {available_external_models()}"
        )
    spec = EXTERNAL_MODEL_REGISTRY[normalized]
    notes = f"{spec.offline_note} Task context: {task_id}."
    if extra_note:
        notes = f"{notes} {extra_note}"
    return ExternalResult(
        model_name=spec.name,
        status="offline_placeholder",
        offline=offline,
        citation=spec.citation,
        family=spec.family,
        expected_inputs=spec.expected_inputs,
        expected_outputs=spec.expected_outputs,
        protocol_difference=f"Protocol difference: {spec.protocol_difference}",
        notes=notes,
        metrics={},
    )


def list_external_results(task_id: str = "T1", offline: bool = True) -> List[ExternalResult]:
    """Return protocol records for all registered references. Complexity: ``O(N)``."""
    return [
        get_external_result(spec.name, task_id=task_id, offline=offline)
        for spec in EXTERNAL_MODEL_REGISTRY.values()
    ]


__all__ = [
    "ExternalModelSpec",
    "ExternalResult",
    "EXTERNAL_MODEL_REGISTRY",
    "available_external_models",
    "get_external_result",
    "list_external_results",
]
