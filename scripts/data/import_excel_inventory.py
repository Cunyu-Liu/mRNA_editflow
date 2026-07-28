#!/usr/bin/env python3
"""D0-01: import the CodonFlow integrated Excel catalog into the data registry.

Input:
    data/raw/codonflow_integrated_dataset_catalog_ranked.xlsx

Every Excel row is mapped to exactly one ``inventory_kind``:

    DATASET | MODEL | DATABASE | PAPER | AUXILIARY_RESOURCE | NOT_RELEVANT

Mapping rules (deterministic, frozen for D0-01):

* Sheet ``模型适配排序`` (model adaptation ranking) -> ``MODEL``.
  Each row is a model entry (模型名称 + 论文标题 + 可核验 ID). The 适配等级
  column (S/A/B/C/D) is preserved as ``adapt_level``; it does not change the
  inventory kind.
* Sheet ``数据集资源排序`` (dataset resource ranking) -> ``DATABASE`` when the
  resource name contains a known stable-database token (RefSeq, Ensembl,
  GENCODE, NCBI, GTEx, ENCODE, FANTOM5, 4DN, RNAcentral, Rfam, PDB, RNASolo,
  CELLxGENE, GEO, HCA, hECA); otherwise -> ``DATASET`` (bounded, versionable
  data releases rather than living database portals).
* Sheet ``Sources`` -> ``PAPER``. Each row is a per-model bibliographic /
  code verification source (paper or official-repo URL evidence).

Outputs:
    data_registry/excel_inventory.parquet
    docs/data/excel_inventory_audit.md

Usage:
    python scripts/data/import_excel_inventory.py \
        --excel data/raw/codonflow_integrated_dataset_catalog_ranked.xlsx
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

MODEL_SHEET = "模型适配排序"
RESOURCE_SHEET = "数据集资源排序"
SOURCES_SHEET = "Sources"

INVENTORY_KINDS = [
    "DATASET",
    "MODEL",
    "DATABASE",
    "PAPER",
    "AUXILIARY_RESOURCE",
    "NOT_RELEVANT",
]

# Tokens identifying living database portals (as opposed to bounded dataset
# releases). Matched case-insensitively against the resource name.
KNOWN_DATABASE_TOKENS = [
    "refseq",
    "ensembl",
    "gencode",
    "ncbi",
    "gtex",
    "encode",
    "fantom5",
    "4dn",
    "rnacentral",
    "rfam",
    "pdb",
    "rnasolo",
    "cellxgene",
    "geo",
    "hca",
    "heca",
]

EXPECTED_MODEL_ROWS = 78
EXPECTED_RESOURCE_ROWS = 14


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names (the catalog has e.g. '数据集 ')."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _cell(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def classify_resource(name: str) -> str:
    """DATABASE when the resource is a living database portal, else DATASET."""
    lowered = name.lower()
    for token in KNOWN_DATABASE_TOKENS:
        if token in lowered:
            return "DATABASE"
    return "DATASET"


def build_inventory(excel_path: Path) -> pd.DataFrame:
    """Read the workbook and return one classified row per Excel data row."""
    xl = pd.ExcelFile(excel_path)
    records: list[dict] = []

    models = _clean_columns(xl.parse(MODEL_SHEET))
    for offset, row in models.iterrows():
        excel_row = offset + 2  # header occupies Excel row 1
        records.append(
            {
                "row_uid": f"{MODEL_SHEET}:{excel_row}",
                "sheet": MODEL_SHEET,
                "excel_row": excel_row,
                "inventory_kind": "MODEL",
                "rationale": "model adaptation ranking entry (模型名称+论文标题+可核验 ID)",
                "name": _cell(row.get("模型名称")),
                "category": _cell(row.get("类别")),
                "adapt_level": _cell(row.get("适配等级")),
                "adapt_rank": _cell(row.get("适配排名")),
                "adapt_score": _cell(row.get("适配分")),
                "usage_role": _cell(row.get("CodonFlow 使用定位")),
                "paper_title": _cell(row.get("论文标题(英文原文)")),
                "verifiable_id": _cell(row.get("可核验 ID")),
                "training_data": _cell(row.get("训练数据集")),
                "data_url": _cell(row.get("数据集访问链接")),
                "code_url": _cell(row.get("代码或模型链接")),
                "license": _cell(row.get("许可协议")),
                "notes": _cell(row.get("备注/可信度说明")),
                "source_urls": "",
            }
        )

    resources = _clean_columns(xl.parse(RESOURCE_SHEET))
    for offset, row in resources.iterrows():
        excel_row = offset + 2
        name = _cell(row.get("数据集"))
        kind = classify_resource(name)
        rationale = (
            f"resource name matches database token -> {kind}"
            if kind == "DATABASE"
            else "bounded/versionable data release -> DATASET"
        )
        records.append(
            {
                "row_uid": f"{RESOURCE_SHEET}:{excel_row}",
                "sheet": RESOURCE_SHEET,
                "excel_row": excel_row,
                "inventory_kind": kind,
                "rationale": rationale,
                "name": name,
                "category": _cell(row.get("类别")),
                "adapt_level": "",
                "adapt_rank": "",
                "adapt_score": "",
                "usage_role": _cell(row.get("使用定位")),
                "paper_title": "",
                "verifiable_id": "",
                "training_data": "",
                "data_url": _cell(row.get("数据集访问链接")),
                "code_url": "",
                "license": "",
                "notes": _cell(row.get("备注/可信度说明")),
                "source_urls": "",
            }
        )

    sources = _clean_columns(xl.parse(SOURCES_SHEET))
    url_by_model: dict[str, str] = {}
    for offset, row in sources.iterrows():
        excel_row = offset + 2
        model_name = _cell(row.get("模型名称"))
        urls = _cell(row.get("核验来源 URL"))
        if model_name:
            url_by_model[model_name] = urls
        records.append(
            {
                "row_uid": f"{SOURCES_SHEET}:{excel_row}",
                "sheet": SOURCES_SHEET,
                "excel_row": excel_row,
                "inventory_kind": "PAPER",
                "rationale": "per-model bibliographic/code verification source",
                "name": model_name,
                "category": "",
                "adapt_level": "",
                "adapt_rank": "",
                "adapt_score": "",
                "usage_role": "verification source",
                "paper_title": "",
                "verifiable_id": "",
                "training_data": "",
                "data_url": "",
                "code_url": "",
                "license": "",
                "notes": "",
                "source_urls": urls,
            }
        )

    df = pd.DataFrame.from_records(records)

    # Join Sources evidence URLs onto the model rows for traceability.
    df.loc[df["sheet"] == MODEL_SHEET, "source_urls"] = df.loc[
        df["sheet"] == MODEL_SHEET, "name"
    ].map(url_by_model).fillna("")
    return df


def acceptance_checks(df: pd.DataFrame) -> list[tuple[str, bool, str]]:
    """Return (check_name, passed, detail) triples for the D0-01 acceptance."""
    models = df[df["sheet"] == MODEL_SHEET]
    resources = df[df["sheet"] == RESOURCE_SHEET]
    unexplained = df[(df["inventory_kind"] == "") | (df["rationale"] == "")]
    bad_kind = df[~df["inventory_kind"].isin(INVENTORY_KINDS)]
    empty_name = df[df["name"] == ""]
    checks = [
        (
            "78 model entries mapped",
            len(models) == EXPECTED_MODEL_ROWS
            and (models["inventory_kind"] == "MODEL").all(),
            f"model rows={len(models)} (expected {EXPECTED_MODEL_ROWS}), "
            f"all kind=MODEL: {(models['inventory_kind'] == 'MODEL').all()}",
        ),
        (
            "14 resource classes mapped",
            len(resources) == EXPECTED_RESOURCE_ROWS
            and resources["inventory_kind"].isin(["DATASET", "DATABASE"]).all(),
            f"resource rows={len(resources)} (expected {EXPECTED_RESOURCE_ROWS}), "
            f"kinds={sorted(resources['inventory_kind'].unique())}",
        ),
        (
            "no unexplained entries",
            len(unexplained) == 0 and len(bad_kind) == 0 and len(empty_name) == 0,
            f"unexplained={len(unexplained)}, invalid_kind={len(bad_kind)}, "
            f"empty_name={len(empty_name)}, total_rows={len(df)}",
        ),
    ]
    return checks


def _display_path(path: Path) -> str:
    """Repo-relative path when possible, else the absolute path."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def render_audit_md(df: pd.DataFrame, excel_path: Path, checks) -> str:
    kind_counts = df["inventory_kind"].value_counts().to_dict()
    models = df[df["sheet"] == MODEL_SHEET]
    resources = df[df["sheet"] == RESOURCE_SHEET]
    n_with_sources = int((models["source_urls"] != "").sum())

    lines = [
        "# Excel Inventory Audit (D0-01)",
        "",
        f"- input: `{_display_path(excel_path)}`",
        f"- input sha256: `{sha256_of(excel_path)}`",
        f"- output: `data_registry/excel_inventory.parquet`",
        f"- total Excel data rows classified: {len(df)}",
        "",
        "## Sheet inventory",
        "",
        "| sheet | data rows | role |",
        "|---|---|---|",
        f"| {MODEL_SHEET} | {len(models)} | model adaptation ranking |",
        f"| {RESOURCE_SHEET} | {len(resources)} | dataset resource ranking |",
        f"| {SOURCES_SHEET} | {len(df[df['sheet'] == SOURCES_SHEET])} | per-model verification sources |",
        "",
        "## Mapping rules (frozen)",
        "",
        f"- `{MODEL_SHEET}` -> `MODEL` (model entry: name + paper title + verifiable ID).",
        f"- `{RESOURCE_SHEET}` -> `DATABASE` if the resource name contains a known",
        "  database token (RefSeq/Ensembl/GENCODE/NCBI/GTEx/ENCODE/FANTOM5/4DN/",
        "  RNAcentral/Rfam/PDB/RNASolo/CELLxGENE/GEO/HCA/hECA), else `DATASET`.",
        f"- `{SOURCES_SHEET}` -> `PAPER` (bibliographic/code verification source).",
        "",
        "## Inventory kind counts",
        "",
        "| inventory_kind | rows |",
        "|---|---|",
    ]
    for kind in INVENTORY_KINDS:
        lines.append(f"| {kind} | {kind_counts.get(kind, 0)} |")
    lines += [
        "",
        "## Model sheet coverage",
        "",
        f"- model rows: {len(models)}",
        f"- rows with Sources verification URL: {n_with_sources}",
        "- adapt_level distribution: "
        + ", ".join(
            f"{k}={v}" for k, v in models["adapt_level"].value_counts().sort_index().items()
        ),
        "- category distribution: "
        + ", ".join(
            f"{k}={v}" for k, v in models["category"].value_counts().items()
        ),
        "",
        "## Resource sheet mapping (all 14 rows)",
        "",
        "| excel_row | usage_role | resource | inventory_kind |",
        "|---|---|---|---|",
    ]
    for _, r in resources.iterrows():
        lines.append(
            f"| {r['excel_row']} | {r['usage_role']} | {r['name']} | {r['inventory_kind']} |"
        )
    lines += [
        "",
        "## Acceptance",
        "",
        "| check | status | detail |",
        "|---|---|---|",
    ]
    for name, passed, detail in checks:
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} | {detail} |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--excel",
        default=str(REPO_ROOT / "data/raw/codonflow_integrated_dataset_catalog_ranked.xlsx"),
    )
    parser.add_argument(
        "--parquet",
        default=str(REPO_ROOT / "data_registry/excel_inventory.parquet"),
    )
    parser.add_argument(
        "--audit-md",
        default=str(REPO_ROOT / "docs/data/excel_inventory_audit.md"),
    )
    args = parser.parse_args(argv)

    excel_path = Path(args.excel)
    if not excel_path.is_file():
        print(f"excel not found: {excel_path}")
        return 2

    df = build_inventory(excel_path)
    parquet_path = Path(args.parquet)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)

    checks = acceptance_checks(df)
    audit_path = Path(args.audit_md)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(render_audit_md(df, excel_path, checks), encoding="utf-8")

    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'}: {name} ({detail})")
    n_fail = sum(1 for _, p, _ in checks if not p)
    if n_fail:
        print(f"excel inventory INVALID: {n_fail} acceptance check(s) failed")
        return 1
    print(
        f"excel inventory VALID: {len(df)} rows "
        f"-> {_display_path(parquet_path)}, {_display_path(audit_path)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
