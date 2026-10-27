"""User-facing calculation text. Numbers come from the trusted record only."""

from __future__ import annotations

import re

_ARABIC = re.compile(r"[\u0600-\u06FF]")

CLARIFICATIONS = {
    "ambiguous_margin": {
        False: (
            "Margin is not defined for this dataset, so nothing was calculated. "
            "Name the stored column to aggregate, or state the definition to use, "
            "such as revenue minus cost or (revenue minus cost) divided by revenue."
        ),
        True: (
            "لم يُحسب الهامش لأن تعريفه غير محدد. حدّد العمود المخزّن، أو اكتب التعريف المطلوب "
            "مثل الإيراد ناقص التكلفة أو (الإيراد ناقص التكلفة) مقسومًا على الإيراد. لم يُحسب أي رقم."
        ),
    },
    "mixed_currency": {
        False: (
            "The included rows use more than one currency or unit. No conversion was applied "
            "and no total was calculated. Filter to one currency or provide a conversion basis."
        ),
        True: (
            "الصفوف المشمولة تستخدم أكثر من عملة أو وحدة. لم يُطبَّق تحويل ولم يُحسب مجموع. "
            "صفِّ إلى عملة واحدة أو أعطِ أساس التحويل."
        ),
    },
    "interpretation_limit": {
        False: (
            "Forecasts and causal interpretations are not calculated. This tool returns "
            "descriptive aggregates only. A correlation or a total does not establish a cause."
        ),
        True: (
            "لا يحسب هذا المسار توقعات ولا تفسيرات سببية. يعيد تجميعات وصفية فقط. "
            "الارتباط أو المجموع لا يثبت سببًا."
        ),
    },
    "not_a_calculation": {
        False: (
            "This request was not run as a calculation. Name one column and an aggregate "
            "(sum, count, count distinct, minimum, maximum, or average), plus any filters."
        ),
        True: (
            "لم يُنفَّذ هذا الطلب كحساب. سمِّ عمودًا ودالة (مجموع أو عدد أو أدنى أو أعلى أو متوسط) وأي مرشحات."
        ),
    },
    "planner_clarification": {
        False: (
            "The calculation was not run because the request is incomplete. Name the column, "
            "the aggregate, and any filter. Margin, currency conversion, and forecasts are not inferred."
        ),
        True: (
            "لم يُنفَّذ الحساب لأن الطلب غير مكتمل. سمِّ العمود ودالة التجميع والمرشح. "
            "لا يُستنتج الهامش ولا تحويل العملة ولا التوقع."
        ),
    },
}


def arabic(*texts: str | None) -> bool:
    return any(_ARABIC.search(text or "") for text in texts)


def clarification(code: str, question: str | None, columns: list[str]) -> str:
    language = arabic(question, *columns)
    text = CLARIFICATIONS[code][language]
    if columns:
        listed = "، ".join(columns) if language else ", ".join(columns)
        label = "الأعمدة المتاحة" if language else "Available columns"
        return f"{text} {label}: {listed}."
    return text


def describe(record: dict, question: str | None) -> str:
    dataset = record["dataset"]
    language = arabic(
        question,
        dataset["filename"],
        *(column["name"] for column in dataset["columns"]),
    )
    filters = record["filters"] or []
    filter_text = (
        "none"
        if not filters
        else "; ".join(
            f"{item['column']} {item['operator']} {item['value'] if item['value'] is not None else item['values'] or ''}"
            for item in filters
        )
    )
    operation = _operation(record["operation"])
    columns = ", ".join(column["name"] for column in dataset["columns"])
    result = _result_text(record["result"])
    rows = ", ".join(str(row) for row in record["source_rows"][:15])
    if len(record["source_rows"]) > 15:
        rows += f" (+{len(record['source_rows']) - 15})"
    exclusions = record["exclusions"]
    if language:
        return (
            f"النتيجة {result} من {operation} على {dataset['filename']} "
            f"إصدار {dataset['version_id']} (SHA-256 {dataset['sha256']}). "
            f"الأعمدة: {columns}. المرشحات: {filter_text}. الوحدات: {record['units']}. "
            f"الصفوف المشمولة: {record['included_row_count']}. "
            f"الصفوف المستبعدة: {record['excluded_row_count']} "
            f"(ترويسة {exclusions['header']}، مجاميع فرعية {exclusions['subtotal']}، "
            f"مرشح {exclusions['filter']}). صفوف المصدر: {rows or 'لا يوجد'}."
        )
    return (
        f"Result {result} from {operation} on {dataset['filename']} "
        f"version {dataset['version_id']} (SHA-256 {dataset['sha256']}). "
        f"Columns: {columns}. Filters: {filter_text}. Units: {record['units']}. "
        f"Included rows: {record['included_row_count']}. "
        f"Excluded rows: {record['excluded_row_count']} "
        f"(header {exclusions['header']}, subtotal {exclusions['subtotal']}, "
        f"filter {exclusions['filter']}). Source rows: {rows or 'none'}."
    )


def _operation(operation: dict) -> str:
    parts = []
    if operation["aggregates"]:
        parts.append(
            ", ".join(
                f"{item['function']}({item['column'] or '*'})"
                for item in operation["aggregates"]
            )
        )
    if operation["group_by"]:
        parts.append("group by " + ", ".join(operation["group_by"]))
    if operation["project"] and not operation["aggregates"]:
        parts.append("project " + ", ".join(operation["project"]))
    if operation["limit"]:
        parts.append(f"limit {operation['limit']}")
    return "; ".join(parts) or "calculate"


def _result_text(result: dict) -> str:
    rows = result["rows"]
    if not rows:
        return "empty"
    if len(rows) == 1 and len(rows[0]) == 1:
        return next(iter(rows[0].values())) or "empty"
    shown = rows[:10]
    rendered = "; ".join(
        ", ".join(f"{key}={value}" for key, value in row.items()) for row in shown
    )
    if len(rows) > 10:
        rendered += f"; +{len(rows) - 10} more in the result table"
    return rendered
