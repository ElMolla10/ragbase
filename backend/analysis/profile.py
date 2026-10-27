"""Classify authorized CSV rows without guessing metric definitions."""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal

from backend.analysis.limits import MAX_COLUMNS, MAX_SOURCE_ROWS

DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
SUBTOTALS = frozenset(
    {"total", "subtotal", "grand total", "المجموع", "الإجمالي", "مجموع"}
)
UNIT_NAMES = frozenset({"unit", "units", "currency", "currency_code", "عملة", "وحدة"})
PERIOD_NAMES = frozenset(
    {"period", "periods", "quarter", "month", "الفترة", "الربع", "الشهر"}
)
MARGIN_NAMES = frozenset(
    {
        "margin",
        "gross margin",
        "net margin",
        "profit margin",
        "هامش",
        "هامش الربح",
        "هامش إجمالي",
        "هامش صافي",
    }
)


class DatasetError(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class ClarifyError(Exception):
    def __init__(self, code: str, **detail):
        super().__init__(code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class Column:
    ref: str
    name: str
    type: str
    scale: int = 0


@dataclass
class CsvDataset:
    version_id: str
    document_id: str
    filename: str
    sha256: str
    extractor: str | None
    warnings: list
    header: bool
    columns: list[Column]
    rows: list[dict]
    duplicate_identifiers: list[str] = field(default_factory=list)

    def column(self, name: str) -> Column:
        matches = [column for column in self.columns if same_name(column.name, name)]
        if not matches:
            raise DatasetError("unknown_column")
        if len(matches) > 1:
            raise DatasetError("ambiguous_column")
        return matches[0]

    def unit_column(self) -> Column | None:
        matches = [column for column in self.columns if fold(column.name) in UNIT_NAMES]
        return matches[0] if len(matches) == 1 else None

    def period_column(self) -> Column | None:
        matches = [
            column for column in self.columns if fold(column.name) in PERIOD_NAMES
        ]
        return matches[0] if len(matches) == 1 else None

    def margin_columns(self) -> list[Column]:
        return [column for column in self.columns if is_margin_name(column.name)]


def fold(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().casefold()


def same_name(left: str, right: str) -> bool:
    return fold(left) == fold(right)


def is_margin_name(name: str) -> bool:
    folded = fold(name)
    return folded in MARGIN_NAMES or folded.endswith(" margin") or "هامش" in folded


def mentions_margin(text: str | None) -> bool:
    folded = fold(text or "")
    return "margin" in folded or "هامش" in folded


def classify_cell(raw: str) -> tuple[str, Decimal | str | None]:
    stripped = raw.strip()
    if stripped == "":
        return "empty", None
    normalized = stripped.translate(DIGIT_MAP).replace("٫", ".")
    if (
        len(normalized) > 1
        and normalized[0] == "0"
        and normalized[1].isdigit()
        and "." not in normalized
    ):
        return "identifier", normalized
    if _is_decimal(normalized):
        return "decimal", Decimal(normalized)
    return "text", stripped


def _is_decimal(value: str) -> bool:
    if not value or value in {"-", ".", "-."}:
        return False
    body = value[1:] if value[0] == "-" else value
    whole, dot, fraction = body.partition(".")
    if dot and (not fraction or not fraction.isdigit() or "." in fraction):
        return False
    if not whole.isdigit():
        return False
    return whole == "0" or not whole.startswith("0")


def snapshot_value(column: Column, raw: str) -> str:
    kind, value = classify_cell(raw)
    if kind == "empty":
        return ""
    if column.type == "decimal" and kind == "decimal":
        return format(value, "f")
    if column.type == "identifier" and kind == "identifier":
        return str(value)
    return raw


def build_dataset(
    *,
    version_id,
    document_id,
    filename,
    sha256,
    extractor,
    warnings,
    pages,
) -> CsvDataset:
    parsed = _pages(pages)
    labels = _header(parsed[0]["cells"])
    names = labels or [
        f"column_{index}" for index in range(1, len(parsed[0]["cells"]) + 1)
    ]
    typed_rows = parsed[1:] if labels else parsed
    columns = _columns(names, typed_rows)
    rows = []
    for index, row in enumerate(parsed):
        if labels and index == 0:
            role = "header"
        elif _subtotal(row["cells"]):
            role = "subtotal"
        else:
            role = "data"
        rows.append({**row, "role": role})
    return CsvDataset(
        version_id=str(version_id),
        document_id=str(document_id),
        filename=filename,
        sha256=sha256,
        extractor=extractor,
        warnings=list(warnings or []),
        header=bool(labels),
        columns=columns,
        rows=rows,
        duplicate_identifiers=_duplicates(columns, rows),
    )


def public_profile(dataset: CsvDataset) -> dict:
    """Column types only. Cell values stay out of the planner prompt."""
    unit = dataset.unit_column()
    return {
        "kind": "csv_profile",
        "filename": dataset.filename,
        "version_id": dataset.version_id,
        "sha256": dataset.sha256,
        "header": dataset.header,
        "columns": [
            {"name": column.name, "type": column.type} for column in dataset.columns
        ],
        "data_row_count": sum(row["role"] == "data" for row in dataset.rows),
        "unit_column": None if unit is None else unit.name,
        "margin_columns": [column.name for column in dataset.margin_columns()],
        "duplicate_identifier_columns": list(dataset.duplicate_identifiers),
    }


def _pages(pages) -> list[dict]:
    if not isinstance(pages, list) or not pages or len(pages) > MAX_SOURCE_ROWS:
        raise DatasetError("source_limit")
    parsed = []
    width = None
    for page in pages:
        location = page.get("location")
        if not isinstance(location, dict) or location.get("kind") != "csv_row":
            raise DatasetError("not_csv")
        try:
            cells = json.loads(page["content"])
        except (json.JSONDecodeError, TypeError, ValueError):
            raise DatasetError("not_csv") from None
        if (
            not isinstance(cells, list)
            or not cells
            or len(cells) > MAX_COLUMNS
            or not all(isinstance(cell, str) for cell in cells)
        ):
            raise DatasetError("not_csv")
        width = width or len(cells)
        if len(cells) != width:
            raise DatasetError("not_csv")
        ordinal = location.get("row", page.get("ordinal"))
        if not isinstance(ordinal, int) or not 1 <= ordinal <= MAX_SOURCE_ROWS:
            raise DatasetError("not_csv")
        parsed.append(
            {
                "ordinal": ordinal,
                "line_start": location.get("line_start"),
                "line_end": location.get("line_end"),
                "cells": cells,
                "warnings": list(page.get("warnings") or []),
                "chunk_id": page.get("chunk_id"),
            }
        )
    return parsed


def _header(cells: list[str]) -> list[str] | None:
    labels = [cell.strip() for cell in cells]
    if any(not label for label in labels):
        return None
    if len({fold(label) for label in labels}) != len(labels):
        return None
    if any(classify_cell(label)[0] != "text" for label in labels):
        return None
    return labels


def _subtotal(cells: list[str]) -> bool:
    return any(fold(cell) in SUBTOTALS for cell in cells)


def _columns(names: list[str], rows: list[dict]) -> list[Column]:
    columns = []
    for index, name in enumerate(names):
        kinds = []
        scales = []
        for row in rows:
            kind, value = classify_cell(row["cells"][index])
            if kind == "empty":
                continue
            kinds.append(kind)
            if kind == "decimal" and isinstance(value, Decimal):
                rendered = format(value, "f")
                scales.append(len(rendered.partition(".")[2]))
        if kinds and all(kind == "identifier" for kind in kinds):
            column_type, scale = "identifier", 0
        elif kinds and all(kind == "decimal" for kind in kinds):
            column_type, scale = "decimal", max(scales or [0])
        else:
            column_type, scale = "text", 0
        columns.append(Column(f"c{index}", name, column_type, scale))
    return columns


def _duplicates(columns: list[Column], rows: list[dict]) -> list[str]:
    found = []
    data = [row for row in rows if row["role"] == "data"]
    for column in columns:
        if column.type != "identifier":
            continue
        index = int(column.ref[1:])
        seen = []
        for row in data:
            kind, value = classify_cell(row["cells"][index])
            if kind == "identifier":
                seen.append(value)
        if len(seen) != len(set(seen)):
            found.append(column.name)
    return found
