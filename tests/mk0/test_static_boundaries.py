"""Static fail-closed audits for leakage, critic roles and scientific claims."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import runpy


ROOT = Path(__file__).resolve().parents[2]
MK0_CORE = ROOT / "core" / "mk0"

RUNTIME_RATE_FILES = (
    "rate_kernel.py",
    "foundation_fusion.py",
    "samplers.py",
)

TRAINING_ONLY_IDENTIFIERS = {
    "z_aux",
    "z_src",
    "z_tar",
    "z_target",
    "target_alignment",
    "target_sequence",
    "remaining_target_edits",
    "remaining_target_switches",
}

FORBIDDEN_BASE_SAMPLER_IDENTIFIERS = {
    "critic",
    "critic_score",
    "guidance",
    "guidance_score",
    "evaluator",
    "final_evaluator",
    "reward_model",
}


def _names(tree: ast.AST) -> set[str]:
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            values.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            values.add(node.attr.lower())
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            values.add(node.name.lower())
        elif isinstance(node, ast.arg):
            values.add(node.arg.lower())
    return values


def test_runtime_rate_and_sampler_interfaces_cannot_access_training_alignment() -> None:
    for name in RUNTIME_RATE_FILES:
        path = MK0_CORE / name
        assert path.is_file(), f"missing frozen MK0 runtime module: {path}"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        leaked = _names(tree) & TRAINING_ONLY_IDENTIFIERS
        assert (
            not leaked
        ), f"training-only target/alignment identifiers in {name}: {leaked}"


def test_base_samplers_have_no_critic_guidance_or_final_evaluator_input() -> None:
    path = MK0_CORE / "samplers.py"
    assert path.is_file()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in {
            "paper_first_order_parallel",
            "constrained_single_event_first_order",
        }:
            continue
        parameter_names = {
            arg.arg.lower()
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        }
        assert not parameter_names & FORBIDDEN_BASE_SAMPLER_IDENTIFIERS


def test_mk0_runtime_does_not_import_final_evaluation_or_reward_modules() -> None:
    for name in RUNTIME_RATE_FILES:
        tree = ast.parse((MK0_CORE / name).read_text(encoding="utf-8"))
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.lower() for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.lower())
        assert not any(
            token in module
            for module in imports
            for token in ("final_evaluator", "reward_model", ".eval", ".critic")
        ), f"forbidden guidance/evaluation dependency in {name}: {imports}"


def test_no_unsupported_affirmative_exact_sampling_claim_in_mk0_scope() -> None:
    paths = list(MK0_CORE.glob("*.py"))
    paths += list((ROOT / "docs" / "math").glob("mk0*.md"))
    paths += list((ROOT / "configs" / "math").glob("*"))
    paths += list((ROOT / "schemas").glob("*_v1.schema.json"))
    assert paths

    # Structural boolean identifiers are audited separately for their
    # mandatory false value.  Here we inspect
    # human-language claims, where spaces or hyphens are used.
    claim = re.compile(r"\b(?:exact[- ]gillespie|exact[- ]ctmc[- ]sampling)\b", re.I)
    negation = re.compile(
        r"(?:\bfalse\b|\bnot\b|\bnone\b|\bneither\b|\bnever\b|\bnon[-_ ]|"
        r"\bunsupported\b|\bprohibit(?:ed)?\b|\bforbid(?:den)?\b|"
        r"不得|不是|禁止|不称为|不支持)",
        re.I,
    )
    violations: list[str] = []
    for path in paths:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not claim.search(line):
                continue
            # A prohibited-phrases list commonly puts the negating heading or
            # predicate immediately before the quoted phrase.
            window = " ".join(lines[max(0, line_number - 4) : line_number])
            if not negation.search(window):
                violations.append(
                    f"{path.relative_to(ROOT)}:{line_number}:{line.strip()}"
                )
    assert (
        not violations
    ), "unsupported affirmative exact-sampling claims:\n" + "\n".join(violations)


def test_failed_numerical_is_not_a_generator_state() -> None:
    for name in ("types.py", "rate_kernel.py", "samplers.py"):
        text = (MK0_CORE / name).read_text(encoding="utf-8")
        if name == "types.py":
            # The execution-status enum is expected to name the failure.
            continue
        tree = ast.parse(text)
        generator_nodes = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and "generator" in node.name.lower()
        ]
        for node in generator_nodes:
            assert "FAILED_NUMERICAL" not in ast.unparse(node)


def test_formal_exact_claim_auditor_passes_repo_and_detects_injected_claims(
    tmp_path: Path,
) -> None:
    audit = runpy.run_path(str(ROOT / "scripts" / "mk0" / "audit_exact_claims.py"))[
        "audit"
    ]
    current = audit(ROOT)
    assert current["pass"] is True
    assert current["unsupported_affirmative_claim_count"] == 0
    assert current["structural_false_binding_failures"] == []

    core = tmp_path / "core" / "mk0"
    core.mkdir(parents=True)
    claim_file = core / "claim.py"
    # fmt: off
    claim_file.write_text('"""This implementation is exact Gillespie."""\n', encoding="utf-8")  # mk0-claim-audit-negative-fixture
    # fmt: on
    affirmative = audit(tmp_path)
    assert affirmative["pass"] is False
    assert affirmative["unsupported_affirmative_claim_count"] == 1

    claim_file.write_text(
        '"""This implementation is not exact Gillespie."""\n', encoding="utf-8"
    )
    assert audit(tmp_path)["pass"] is True

    claim_file.write_text(
        '"""This part is not exact Gillespie. The next sampler is exact Gillespie."""\n',  # mk0-claim-audit-negative-fixture
        encoding="utf-8",
    )
    adjacent = audit(tmp_path)
    assert adjacent["pass"] is False
    assert adjacent["unsupported_affirmative_claim_count"] == 1

    claim_file.write_text(
        '"""This is a genuine Continuous-Time Markov Chain sampler."""\n',  # mk0-claim-audit-negative-fixture
        encoding="utf-8",
    )
    expanded = audit(tmp_path)
    assert expanded["pass"] is False
    assert expanded["unsupported_affirmative_claim_count"] == 1

    config_dir = tmp_path / "configs" / "math"
    config_dir.mkdir(parents=True)
    config = config_dir / "bad.yaml"
    # fmt: off
    config.write_text("sampler:\n  exact_gillespie: true\n", encoding="utf-8")  # mk0-claim-audit-negative-fixture
    # fmt: on
    structural = audit(tmp_path)
    assert structural["pass"] is False
    assert structural["structural_false_binding_failures"]
