"""Run one validated plan on a job-local CSV snapshot inside locked DuckDB.

The connection never receives a PostgreSQL URL, extension, or model-written SQL.
External access and extension autoload are disabled before the analytical query.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
import threading
import time
import uuid
from decimal import ROUND_HALF_EVEN, Decimal

import duckdb

from backend.analysis.export import result_csv
from backend.analysis.limits import (
    MAX_OUTPUT_BYTES,
    MAX_OUTPUT_ROWS,
    MAX_TEMP_DIRECTORY,
    MEMORY_LIMIT,
    TIMEOUT_SECONDS,
)
from backend.analysis.plan import OperationPlan
from backend.analysis.profile import (
    ClarifyError,
    Column,
    CsvDataset,
    DatasetError,
    classify_cell,
    snapshot_value,
)

_PATH = re.compile(r"^/tmp/ragbase-calc-[0-9a-f]{32}$")
_NUMBER = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d+)?$")
_IDENT = re.compile(r"^(?:row_number|row_role|c\d{1,2}|measure_\d{1,2}(?:_count)?)$")
_NUMERIC = {"sum", "avg", "min", "max"}


def run(
    dataset: CsvDataset,
    plan: OperationPlan,
    *,
    max_output_rows: int = MAX_OUTPUT_ROWS,
    timeout_seconds: float = TIMEOUT_SECONDS,
) -> dict:
    started = time.monotonic()
    job = _job_dir()
    path = job + "/snapshot.csv"
    try:
        payload = _write_snapshot(dataset, path)
        digest = hashlib.sha256(payload).hexdigest()
        connection = duckdb.connect(":memory:")
        try:
            _prepare(connection, job, path, len(dataset.rows), len(dataset.columns))
            sandbox = _lock(connection)
            _reject_mixed_currency(connection, dataset, plan)
            included = _included_rows(connection, dataset, plan)
            exclusions = _exclusions(dataset, included)
            table = _calculate(
                connection, dataset, plan, included, max_output_rows, timeout_seconds
            )
        finally:
            connection.close()
        periods = _periods(dataset, included)
        record = _record(
            dataset,
            plan,
            digest,
            sandbox,
            exclusions,
            included,
            table,
            periods,
            int((time.monotonic() - started) * 1000),
        )
        export = result_csv(table["columns"], table["rows"])
        if len(export) > MAX_OUTPUT_BYTES:
            raise DatasetError("output_limit")
        return {
            "status": "completed",
            "dataset_sha256": digest,
            "export_csv": export,
            "record": record,
            "citations": _citations(dataset, included),
        }
    finally:
        shutil.rmtree(job, ignore_errors=True)


def _job_dir() -> str:
    path = f"/tmp/ragbase-calc-{uuid.uuid4().hex}"
    if not _PATH.fullmatch(path):
        raise DatasetError("sandbox_rejected")
    os.mkdir(path, 0o700)
    return path


def _write_snapshot(dataset: CsvDataset, path: str) -> bytes:
    header = ["row_number", "row_role"] + [column.ref for column in dataset.columns]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        os.chmod(path, 0o600)
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row in dataset.rows:
            writer.writerow(
                [row["ordinal"], row["role"]]
                + [
                    snapshot_value(column, row["cells"][index])
                    for index, column in enumerate(dataset.columns)
                ]
            )
    with open(path, "rb") as handle:
        return handle.read()


def _prepare(connection, job: str, path: str, rows: int, columns: int) -> None:
    _set(connection, "home_directory", job)
    _set(connection, "temp_directory", job)
    _set(connection, "extension_directory", job)
    connection.execute("SET threads=1")
    connection.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    connection.execute(f"SET max_temp_directory_size='{MAX_TEMP_DIRECTORY}'")
    connection.execute("SET autoinstall_known_extensions=false")
    connection.execute("SET autoload_known_extensions=false")
    connection.execute("SET enable_progress_bar=false")
    names = ", ".join(
        ["TRY_CAST(row_number AS INTEGER) AS row_number", "row_role"]
        + [f"NULLIF(trim(c{index}), '') AS c{index}" for index in range(columns)]
    )
    connection.execute(
        f"CREATE TABLE snapshot AS SELECT {names} FROM read_csv('{path}', header=true, all_varchar=true)"
    )
    loaded = connection.execute("SELECT COUNT(*) FROM snapshot").fetchone()[0]
    if loaded != rows:
        raise DatasetError("sandbox_rejected")


def _set(connection, name: str, value: str) -> None:
    if name not in {"home_directory", "temp_directory", "extension_directory"}:
        raise DatasetError("sandbox_rejected")
    if not _PATH.fullmatch(value):
        raise DatasetError("sandbox_rejected")
    connection.execute(f"SET {name}='{value}'")


def _lock(connection) -> dict:
    connection.execute("SET enable_external_access=false")
    connection.execute("SET lock_configuration=true")
    flags = {
        key: connection.execute(f"SELECT current_setting('{key}')").fetchone()[0]
        for key in (
            "enable_external_access",
            "autoinstall_known_extensions",
            "autoload_known_extensions",
            "lock_configuration",
            "threads",
            "memory_limit",
        )
    }
    if flags["enable_external_access"] not in (False, "false"):
        raise DatasetError("sandbox_rejected")
    if flags["lock_configuration"] not in (True, "true"):
        raise DatasetError("sandbox_rejected")
    if flags["autoload_known_extensions"] not in (False, "false"):
        raise DatasetError("sandbox_rejected")
    if flags["autoinstall_known_extensions"] not in (False, "false"):
        raise DatasetError("sandbox_rejected")
    if str(flags["threads"]) != "1":
        raise DatasetError("sandbox_rejected")
    for statement in (
        "SET threads=4",
        "SET enable_external_access=true",
        "SELECT * FROM read_csv_auto('/etc/passwd')",
        "LOAD httpfs",
    ):
        try:
            connection.execute(statement)
        except DatasetError:
            raise
        except Exception:
            continue
        raise DatasetError("sandbox_rejected")
    return {
        "engine": "duckdb",
        "duckdb_version": duckdb.__version__,
        "external_access": False,
        "extension_autoload": False,
        "extension_autoinstall": False,
        "settings_locked": True,
        "threads": 1,
        "memory_limit": str(flags["memory_limit"]),
        "job_local_snapshot": True,
        "postgresql_accessed": False,
        "os_cgroup_enforced": False,
        "production_qualified": False,
    }


def _reject_mixed_currency(
    connection, dataset: CsvDataset, plan: OperationPlan
) -> None:
    unit = dataset.unit_column()
    if unit is None or not any(
        item.function in {"sum", "avg"} for item in plan.aggregates
    ):
        return
    where, params = _where(dataset, plan)
    rows = _query(
        connection,
        f"SELECT DISTINCT {unit.ref} FROM snapshot WHERE {where} AND {unit.ref} IS NOT NULL",
        params,
        TIMEOUT_SECONDS,
    )
    values = sorted({row[0] for row in rows if row[0]})
    if len(values) > 1:
        raise ClarifyError("mixed_currency", values=values)


def _included_rows(connection, dataset: CsvDataset, plan: OperationPlan) -> list[int]:
    where, params = _where(dataset, plan)
    rows = _query(
        connection,
        f"SELECT row_number FROM snapshot WHERE {where} ORDER BY row_number",
        params,
        TIMEOUT_SECONDS,
    )
    return [int(row[0]) for row in rows]


def _calculate(connection, dataset, plan, included, max_output_rows, timeout) -> dict:
    if not included and plan.aggregates:
        raise DatasetError("no_numeric_values")
    select, params, meta = _select(dataset, plan)
    where, where_params = _where(dataset, plan)
    group = _group(dataset, plan)
    order = _order(dataset, plan)
    limit = int(plan.limit or (max_output_rows + 1))
    sql = f"SELECT {select} FROM snapshot WHERE {where}{group}{order} LIMIT {limit}"
    fetched = _query(connection, sql, [*params, *where_params], timeout)
    if plan.limit is None and len(fetched) > max_output_rows:
        raise DatasetError("output_limit")
    columns = [item["name"] for item in meta]
    rows = _sort_rows(
        [_render_row(meta, row) for row in fetched],
        plan,
    )
    if plan.aggregates and rows:
        measures = [item.output for item in plan.aggregates]
        if all(not row.get(name) for row in rows for name in measures):
            raise DatasetError("no_numeric_values")
    return {"columns": columns, "rows": rows}


def _select(dataset: CsvDataset, plan: OperationPlan):
    meta = []
    choose = []
    params = []
    if plan.aggregates:
        for name in plan.group_by:
            column = dataset.column(name)
            _ident(column.ref)
            choose.append(column.ref)
            meta.append({"name": column.name, "kind": "group", "ref": column.ref})
        for index, item in enumerate(plan.aggregates):
            alias = f"measure_{index}"
            _ident(alias)
            expression, count_alias = _aggregate(dataset, item, alias)
            choose.append(expression)
            meta.append(
                {
                    "name": item.output,
                    "kind": item.function,
                    "column": item.column,
                    "scale": _scale(dataset, item),
                    "count_alias": count_alias,
                }
            )
            if count_alias:
                choose.append(
                    f"COUNT({dataset.column(item.column).ref}) AS {count_alias}"
                )
    else:
        names = plan.project or plan.group_by
        for name in names:
            column = dataset.column(name)
            _ident(column.ref)
            choose.append(column.ref)
            meta.append({"name": column.name, "kind": "project", "ref": column.ref})
    if not choose:
        raise DatasetError("invalid_plan")
    return ", ".join(choose), params, meta


def _aggregate(dataset: CsvDataset, item, alias: str) -> tuple[str, str | None]:
    if item.function == "count" and item.column is None:
        return f"COUNT(*) AS {alias}", None
    column = dataset.column(item.column)
    _ident(column.ref)
    if item.function == "count":
        return f"COUNT({column.ref}) AS {alias}", None
    if item.function == "count_distinct":
        return f"COUNT(DISTINCT {column.ref}) AS {alias}", None
    if item.function not in _NUMERIC:
        raise DatasetError("invalid_plan")
    cast = f"TRY_CAST({column.ref} AS DECIMAL(38,9))"
    if item.function == "avg":
        return f"SUM({cast}) AS {alias}", f"{alias}_count"
    return f"{item.function.upper()}({cast}) AS {alias}", None


def _scale(dataset: CsvDataset, item) -> int:
    if item.column is None or item.function in {"count", "count_distinct"}:
        return 0
    column = dataset.column(item.column)
    if item.function == "avg":
        return max(column.scale, 6)
    return column.scale


def _where(dataset: CsvDataset, plan: OperationPlan) -> tuple[str, list]:
    parts = ["row_role = 'data'"]
    params = []
    for item in plan.filters:
        column = dataset.column(item.column)
        _ident(column.ref)
        clause, values = _predicate(column, item)
        parts.append(clause)
        params.extend(values)
    return " AND ".join(parts), params


def _predicate(column: Column, item) -> tuple[str, list]:
    ref = column.ref
    if item.operator in {"gt", "gte", "lt", "lte"}:
        symbol = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}[item.operator]
        return (
            f"TRY_CAST({ref} AS DECIMAL(38,9)) {symbol} TRY_CAST(? AS DECIMAL(38,9))",
            [_decimal_text(item.value)],
        )
    if item.operator in {"eq", "neq"}:
        symbol = "=" if item.operator == "eq" else "<>"
        if column.type == "decimal":
            return (
                f"TRY_CAST({ref} AS DECIMAL(38,9)) {symbol} TRY_CAST(? AS DECIMAL(38,9))",
                [_decimal_text(item.value)],
            )
        return f"{ref} {symbol} ?", [item.value.strip()]
    if item.operator == "contains":
        return f"contains({ref}, ?)", [item.value]
    if item.operator == "in":
        if column.type == "decimal":
            values = [_decimal_text(value) for value in item.values]
            marks = ", ".join("TRY_CAST(? AS DECIMAL(38,9))" for _ in values)
            return f"TRY_CAST({ref} AS DECIMAL(38,9)) IN ({marks})", values
        values = [value.strip() for value in item.values]
        marks = ", ".join("?" for _ in values)
        return f"{ref} IN ({marks})", values
    if item.operator == "is_null":
        return f"{ref} IS NULL", []
    if item.operator == "is_not_null":
        return f"{ref} IS NOT NULL", []
    raise DatasetError("invalid_plan")


def _decimal_text(value: str) -> str:
    kind, parsed = classify_cell(value or "")
    if kind != "decimal":
        raise DatasetError("invalid_plan")
    return format(parsed, "f")


def _group(dataset: CsvDataset, plan: OperationPlan) -> str:
    if not plan.group_by:
        return ""
    refs = []
    for name in plan.group_by:
        column = dataset.column(name)
        _ident(column.ref)
        refs.append(column.ref)
    return " GROUP BY " + ", ".join(refs)


def _order(dataset: CsvDataset, plan: OperationPlan) -> str:
    if plan.sort:
        return ""
    if plan.group_by:
        refs = []
        for name in plan.group_by:
            column = dataset.column(name)
            _ident(column.ref)
            refs.append(column.ref)
        return " ORDER BY " + ", ".join(refs)
    if plan.aggregates:
        return ""
    return " ORDER BY row_number"


def _sort_rows(rows: list[dict], plan: OperationPlan) -> list[dict]:
    ordered = rows
    for item in reversed(plan.sort):

        def key(row, column=item.column):
            text = row.get(column, "")
            if _NUMBER.fullmatch(text or ""):
                return (0, Decimal(text))
            return (1, text)

        ordered = sorted(ordered, key=key, reverse=item.direction == "desc")
    return ordered


def _render_row(meta, row) -> dict:
    rendered = {}
    cursor = 0
    for item in meta:
        value = row[cursor]
        cursor += 1
        if item["kind"] == "avg":
            count = row[cursor]
            cursor += 1
            rendered[item["name"]] = (
                "" if not count else _average(value, count, item["scale"])
            )
        elif item["kind"] in _NUMERIC:
            rendered[item["name"]] = (
                "" if value is None else _money(value, item["scale"])
            )
        elif item["kind"] in {"count", "count_distinct"}:
            rendered[item["name"]] = str(int(value))
        else:
            rendered[item["name"]] = "" if value is None else str(value)
    return rendered


def _money(value, scale: int) -> str:
    decimal = value if isinstance(value, Decimal) else Decimal(str(value))
    quantizer = Decimal("1") if scale <= 0 else Decimal("1").scaleb(-scale)
    return format(decimal.quantize(quantizer, rounding=ROUND_HALF_EVEN), "f")


def _average(total, count, scale: int) -> str:
    if not count:
        return ""
    decimal = total if isinstance(total, Decimal) else Decimal(str(total))
    return _money(decimal / Decimal(int(count)), scale)


def _query(connection, sql: str, params: list, timeout: float):
    if _unsafe_sql(sql):
        raise DatasetError("sandbox_rejected")
    box = {}

    def target():
        try:
            box["rows"] = connection.execute(sql, params).fetchall()
        except Exception as exc:
            box["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        connection.interrupt()
        thread.join(1)
        raise DatasetError("runtime_limit")
    if "error" in box:
        if isinstance(box["error"], DatasetError):
            raise box["error"]
        raise DatasetError("sandbox_rejected") from box["error"]
    return box["rows"]


def _unsafe_sql(sql: str) -> bool:
    folded = sql.casefold()
    return any(
        token in folded
        for token in (
            "read_csv",
            "read_parquet",
            "postgres",
            "http",
            "copy ",
            "attach",
            "load ",
            "install",
            "pragma",
            ";",
        )
    )


def _ident(name: str) -> None:
    if not _IDENT.fullmatch(name):
        raise DatasetError("sandbox_rejected")


def _exclusions(dataset: CsvDataset, included: list[int]) -> dict:
    roles = {"header": 0, "subtotal": 0, "data": 0}
    for row in dataset.rows:
        roles[row["role"]] += 1
    included_set = set(included)
    filter_excluded = sum(
        row["role"] == "data" and row["ordinal"] not in included_set
        for row in dataset.rows
    )
    return {
        "header": roles["header"],
        "subtotal": roles["subtotal"],
        "filter": filter_excluded,
        "included": len(included),
        "excluded": roles["header"] + roles["subtotal"] + filter_excluded,
        "physical_rows": len(dataset.rows),
    }


def _periods(dataset: CsvDataset, included: list[int]) -> list[str]:
    column = dataset.period_column()
    if column is None:
        return []
    index = int(column.ref[1:])
    chosen = set(included)
    values = []
    for row in dataset.rows:
        if row["ordinal"] in chosen and row["cells"][index].strip():
            values.append(row["cells"][index].strip())
    return list(dict.fromkeys(values))[:12]


def _citations(dataset: CsvDataset, included: list[int]) -> list[dict]:
    chosen = set(included)
    citations = []
    for row in dataset.rows:
        if row["ordinal"] not in chosen:
            continue
        citations.append(
            {
                "version_id": dataset.version_id,
                "document_id": dataset.document_id,
                "sha256": dataset.sha256,
                "filename": dataset.filename,
                "row": row["ordinal"],
                "line_start": row["line_start"],
                "line_end": row["line_end"],
                "quote": _quote(row["cells"]),
                "location": {
                    "kind": "csv_row",
                    "row": row["ordinal"],
                    "line_start": row["line_start"],
                    "line_end": row["line_end"],
                },
                "chunk_id": None if row["chunk_id"] is None else str(row["chunk_id"]),
                "warnings": row["warnings"],
            }
        )
    return citations


def _quote(cells: list[str]) -> str:
    import json

    return json.dumps(cells, ensure_ascii=False)[:500]


def _record(
    dataset, plan, digest, sandbox, exclusions, included, table, periods, elapsed
):
    nulls = _null_measures(dataset, plan, included)
    return {
        "production_qualified": False,
        "dataset": {
            "filename": dataset.filename,
            "version_id": dataset.version_id,
            "document_id": dataset.document_id,
            "sha256": dataset.sha256,
            "snapshot_sha256": digest,
            "extractor": dataset.extractor,
            "warnings": list(dataset.warnings),
            "header_row": 1 if dataset.header else None,
            "columns": [
                {
                    "name": column.name,
                    "type": column.type,
                    "scale": column.scale,
                }
                for column in dataset.columns
            ],
        },
        "filters": [
            {
                "column": item.column,
                "operator": item.operator,
                "value": item.value,
                "values": item.values,
            }
            for item in plan.filters
        ],
        "operation": {
            "group_by": list(plan.group_by),
            "aggregates": [
                {
                    "function": item.function,
                    "column": item.column,
                    "output": item.output,
                }
                for item in plan.aggregates
            ],
            "sort": [
                {"column": item.column, "direction": item.direction}
                for item in plan.sort
            ],
            "limit": plan.limit,
            "project": list(plan.project),
        },
        "units": _units(dataset, included),
        "included_row_count": exclusions["included"],
        "excluded_row_count": exclusions["excluded"],
        "exclusions": {
            "header": exclusions["header"],
            "subtotal": exclusions["subtotal"],
            "filter": exclusions["filter"],
            "null_measure_rows": nulls,
        },
        "result": table,
        "source_rows": included,
        "duplicate_identifier_columns": list(dataset.duplicate_identifiers),
        "periods_included": periods,
        "execution_ms": elapsed,
        "sandbox": sandbox,
        "formula_note": "Formula-like CSV text is stored literally and is not executed.",
        "limitation": (
            "Development calculation over one authorized CSV version. "
            "Not a forecast, not a causal claim, and not production qualification."
        ),
    }


def _units(dataset: CsvDataset, included: list[int]) -> str:
    column = dataset.unit_column()
    if column is None:
        return "unspecified"
    index = int(column.ref[1:])
    chosen = set(included)
    values = {
        row["cells"][index].strip()
        for row in dataset.rows
        if row["ordinal"] in chosen and row["cells"][index].strip()
    }
    if len(values) == 1:
        return next(iter(values))[:32]
    return "unspecified"


def _null_measures(
    dataset: CsvDataset, plan: OperationPlan, included: list[int]
) -> int:
    chosen = set(included)
    counted = 0
    for item in plan.aggregates:
        if item.function not in _NUMERIC or not item.column:
            continue
        index = int(dataset.column(item.column).ref[1:])
        counted = max(
            counted,
            sum(
                row["ordinal"] in chosen
                and classify_cell(row["cells"][index])[0] != "decimal"
                for row in dataset.rows
            ),
        )
    return counted
