"""Unit tests for scripts/data/import_excel_inventory.py (D0-01)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.data.import_excel_inventory import (
    DEFAULT_AUDIT_PATH,
    DEFAULT_PARQUET_PATH,
    EXPECTED_MODEL_ROWS,
    EXPECTED_RESOURCE_ROWS,
    INVENTORY_KINDS,
    acceptance_checks,
    build_parser,
    build_inventory,
    classify_resource,
)


def test_default_generated_parquet_is_outside_git_and_audit_remains_in_git():
    args = build_parser().parse_args([])
    assert Path(args.parquet) == DEFAULT_PARQUET_PATH
    assert DEFAULT_PARQUET_PATH == Path(
        "/mnt/cunyuliu/mrna_xeditflow_routea_v3/route2/data_registry/"
        "excel_inventory.parquet"
    )
    assert REPO_ROOT not in DEFAULT_PARQUET_PATH.parents
    assert Path(args.audit_md) == DEFAULT_AUDIT_PATH
    assert DEFAULT_AUDIT_PATH == REPO_ROOT / "docs/data/excel_inventory_audit.md"


def _write_catalog(path: Path, n_models: int = 0, n_resources: int = 0) -> None:
    model_rows = [
        {
            "适配排名": str(i + 1),
            "适配等级": "S",
            "适配分": "98",
            "CodonFlow 使用定位": "backbone",
            "排序理由": "r",
            "建议动作": "a",
            "原始行号": str(i + 2),
            "类别": "RNA 序列 LM",
            "模型名称": f"Model-{i}",
            "论文标题(英文原文)": f"Paper {i}",
            "作者(简写)": "Doe",
            "发表 venue / 年份": "arXiv 2025",
            "可核验 ID": f"arXiv:2500.{i:05d}",
            "训练数据集": "RefSeq",
            "数据集访问链接": "https://example.org/data",
            "数据量": "n/a",
            "数据清洗/预处理": "n/a",
            "代码或模型链接": "https://example.org/code",
            "许可协议": "MIT",
            "tokenizer 类型": "codon",
            "备注/可信度说明": "n/a",
        }
        for i in range(n_models)
    ]
    resource_rows = [
        {
            "使用定位": f"role-{i}",
            "内容": "c",
            "类别": "RNA 序列 LM",
            "代表来源模型": "m",
            "数据集 ": name,  # trailing space is intentional (real catalog)
            "数据集访问链接": "https://example.org",
            "数据量": "n/a",
            "数据清洗/预处理": "n/a",
            "备注/可信度说明": "n/a",
        }
        for i, name in enumerate(
            ["RefSeq / Ensembl transcripts", "custom 5'UTR release"][j % 2]
            for j in range(n_resources)
        )
    ]
    source_rows = [
        {"模型名称": f"Model-{i}", "核验来源 URL": "https://arxiv.org/abs/0000.00001"}
        for i in range(n_models)
    ]
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(model_rows).to_excel(writer, sheet_name="模型适配排序", index=False)
        pd.DataFrame(resource_rows).to_excel(writer, sheet_name="数据集资源排序", index=False)
        pd.DataFrame(source_rows).to_excel(writer, sheet_name="Sources", index=False)


def test_classify_resource_database_tokens():
    assert classify_resource("RefSeq / Ensembl transcript-CDS-protein + GFF/GTF") == "DATABASE"
    assert classify_resource("GTEx expression / RNA-seq") == "DATABASE"
    assert classify_resource("Rfam families / MSA") == "DATABASE"
    assert classify_resource("CELLxGENE / GEO / HCA / hECA scRNA-seq") == "DATABASE"


def test_classify_resource_bounded_dataset():
    assert classify_resource("mRNABERT downstream datasets (Zenodo)") == "DATASET"
    assert classify_resource("5'UTR / 3'UTR 序列数据") == "DATASET"
    assert classify_resource("bpRNA / RNAStralign / ArchiveII 二级结构数据") == "DATASET"


def test_every_row_classified_with_rationale(tmp_path):
    xlsx = tmp_path / "catalog.xlsx"
    _write_catalog(xlsx, n_models=3, n_resources=2)
    df = build_inventory(xlsx)
    # 3 models + 2 resources + 3 sources rows
    assert len(df) == 8
    assert df["inventory_kind"].isin(INVENTORY_KINDS).all()
    assert (df["rationale"] != "").all()
    assert (df["name"] != "").all()
    kinds = df.groupby("sheet")["inventory_kind"].first().to_dict()
    assert kinds["模型适配排序"] == "MODEL"
    assert kinds["Sources"] == "PAPER"


def test_sources_urls_joined_onto_models(tmp_path):
    xlsx = tmp_path / "catalog.xlsx"
    _write_catalog(xlsx, n_models=2, n_resources=0)
    df = build_inventory(xlsx)
    models = df[df["sheet"] == "模型适配排序"]
    assert (models["source_urls"] == "https://arxiv.org/abs/0000.00001").all()


def test_acceptance_checks_catch_wrong_counts(tmp_path):
    xlsx = tmp_path / "catalog.xlsx"
    _write_catalog(xlsx, n_models=2, n_resources=1)
    df = build_inventory(xlsx)
    checks = dict((name, passed) for name, passed, _ in acceptance_checks(df))
    assert checks["78 model entries mapped"] is False
    assert checks["14 resource classes mapped"] is False
    assert checks["no unexplained entries"] is True


def test_acceptance_checks_pass_at_expected_counts(tmp_path):
    xlsx = tmp_path / "catalog.xlsx"
    _write_catalog(xlsx, n_models=EXPECTED_MODEL_ROWS, n_resources=EXPECTED_RESOURCE_ROWS)
    df = build_inventory(xlsx)
    checks = acceptance_checks(df)
    assert all(passed for _, passed, _ in checks), checks


def test_cli_end_to_end(tmp_path):
    from scripts.data.import_excel_inventory import main

    xlsx = tmp_path / "catalog.xlsx"
    _write_catalog(xlsx, n_models=EXPECTED_MODEL_ROWS, n_resources=EXPECTED_RESOURCE_ROWS)
    parquet = tmp_path / "out.parquet"
    audit = tmp_path / "audit.md"
    rc = main(["--excel", str(xlsx), "--parquet", str(parquet), "--audit-md", str(audit)])
    assert rc == 0
    out = pd.read_parquet(parquet)
    assert len(out) == EXPECTED_MODEL_ROWS + EXPECTED_RESOURCE_ROWS + EXPECTED_MODEL_ROWS
    text = audit.read_text(encoding="utf-8")
    assert "PASS" in text and "FAIL" not in text
    assert f"- output: `{parquet}`" in text
    assert "- output: `data_registry/excel_inventory.parquet`" not in text
