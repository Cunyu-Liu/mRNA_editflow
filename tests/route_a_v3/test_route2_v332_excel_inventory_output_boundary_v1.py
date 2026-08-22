from pathlib import Path

from scripts.data.import_excel_inventory import (
    DEFAULT_AUDIT_PATH,
    DEFAULT_PARQUET_PATH,
    REPO_ROOT,
    ROUTE2_STORAGE_ROOT,
    build_parser,
)


def test_default_generated_parquet_is_under_route2_mnt_not_git() -> None:
    args = build_parser().parse_args([])

    assert Path(args.parquet) == DEFAULT_PARQUET_PATH
    assert DEFAULT_PARQUET_PATH == (
        ROUTE2_STORAGE_ROOT / "data_registry/excel_inventory.parquet"
    )
    assert ROUTE2_STORAGE_ROOT == Path(
        "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2"
    )
    assert REPO_ROOT not in DEFAULT_PARQUET_PATH.parents


def test_small_audit_summary_default_remains_in_git() -> None:
    args = build_parser().parse_args([])

    assert Path(args.audit_md) == DEFAULT_AUDIT_PATH
    assert DEFAULT_AUDIT_PATH == REPO_ROOT / "docs/data/excel_inventory_audit.md"


def test_historical_audit_distinguishes_legacy_output_from_future_default() -> None:
    text = (REPO_ROOT / "docs/data/excel_inventory_audit.md").read_text(
        encoding="utf-8"
    )

    assert "Historical audit retained as a small Git summary" in text
    assert str(DEFAULT_PARQUET_PATH) in text
    assert "the output line below records the legacy run" in " ".join(text.split())
