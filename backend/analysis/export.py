"""Spreadsheet exports. Formula-looking text is escaped; originals are never overwritten."""

from __future__ import annotations

import csv
import io
import re

_NUMBER = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?$")
_FORMULA = re.compile(r"^[\t\r\n ]*[=+\-@]")


def escape_cell(value) -> str:
    text = "" if value is None else str(value)
    if _NUMBER.fullmatch(text):
        return text
    if text[:1] in "=+-@\t\r\n" or _FORMULA.match(text):
        return "'" + text
    return text


def result_csv(columns: list[str], rows: list[dict]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([escape_cell(column) for column in columns])
    for row in rows:
        writer.writerow([escape_cell(row.get(column, "")) for column in columns])
    return "\ufeff" + buffer.getvalue()
