"""CSV calculation contracts. No live provider, database, or model weights."""

import csv
import io
import json
from decimal import Decimal

import pytest

from backend.analysis.engine import _query, _where, run
from backend.analysis.export import escape_cell, result_csv
from backend.analysis.messages import clarification
from backend.analysis.plan import (
    OperationPlan,
    bind_plan,
    interpretation_limited,
    margin_ambiguous,
)
from backend.analysis.planner import fixture_plan, planner_request
from backend.analysis.profile import ClarifyError, DatasetError, build_dataset
from backend.gateway.openrouter import OpenRouterGateway
from backend.openrouter_client import MODELS, PROVIDERS
from backend.policy import assemble


def pages(text):
    rows = list(csv.reader(io.StringIO(text)))
    return [
        {
            "ordinal": index,
            "content": json.dumps(row, ensure_ascii=False),
            "location": {
                "kind": "csv_row",
                "row": index,
                "line_start": index,
                "line_end": index,
            },
            "warnings": [],
        }
        for index, row in enumerate(rows, 1)
    ]


def dataset(text, filename="sales.csv"):
    return build_dataset(
        version_id="version",
        document_id="document",
        filename=filename,
        sha256="ab" * 32,
        extractor="csv-utf8-strings-v1",
        warnings=["csv_strings_only_no_header_type_or_formula_inference"],
        pages=pages(text),
    )


def execute(text, payload, filename="sales.csv"):
    source = dataset(text, filename)
    chosen = bind_plan(OperationPlan.model_validate(payload), source)
    return run(source, chosen)


def test_money_sum_average_min_max_and_count_are_exact():
    text = "id,amount\n001,10.10\n002,20.20\n003,0.30\n"
    summed = execute(
        text,
        {"aggregates": [{"function": "sum", "column": "amount", "output": "result"}]},
    )
    record = summed["record"]
    assert record["result"]["rows"] == [{"result": "30.60"}]
    assert Decimal(record["result"]["rows"][0]["result"]) == Decimal("30.60")
    assert record["included_row_count"] == 3
    assert record["excluded_row_count"] == 1
    assert record["exclusions"]["header"] == 1
    assert record["units"] == "unspecified"
    assert record["dataset"]["version_id"] == "version"
    assert record["dataset"]["columns"][0]["type"] == "identifier"
    assert record["source_rows"] == [2, 3, 4]
    assert record["production_qualified"] is False
    averaged = execute(
        text,
        {"aggregates": [{"function": "avg", "column": "amount", "output": "result"}]},
    )
    assert Decimal(averaged["record"]["result"]["rows"][0]["result"]) == Decimal(
        "10.20"
    )
    assert execute(
        text,
        {"aggregates": [{"function": "min", "column": "amount", "output": "result"}]},
    )["record"]["result"]["rows"] == [{"result": "0.30"}]
    assert execute(
        text,
        {"aggregates": [{"function": "max", "column": "amount", "output": "result"}]},
    )["record"]["result"]["rows"] == [{"result": "20.20"}]
    assert execute(
        text,
        {"aggregates": [{"function": "count", "column": None, "output": "result"}]},
    )["record"]["result"]["rows"] == [{"result": "3"}]


def test_arabic_and_mixed_columns_group_filter_and_exclude_subtotals():
    text = "المنطقة,Quarter,المبيعات\nالقاهرة,Q1,10.50\nالإسكندرية,Q1,2.25\nالقاهرة,Q2,1.25\nالمجموع,Q2,14.00\n"
    grouped = execute(
        text,
        {
            "group_by": ["المنطقة"],
            "aggregates": [
                {"function": "sum", "column": "المبيعات", "output": "النتيجة"}
            ],
        },
        filename="مبيعات.csv",
    )
    rows = {
        row["المنطقة"]: row["النتيجة"] for row in grouped["record"]["result"]["rows"]
    }
    assert rows == {"القاهرة": "11.75", "الإسكندرية": "2.25"}
    assert grouped["record"]["exclusions"]["subtotal"] == 1
    assert grouped["record"]["included_row_count"] == 3
    filtered = execute(
        text,
        {
            "filters": [{"column": "Quarter", "operator": "eq", "value": "Q1"}],
            "aggregates": [
                {"function": "sum", "column": "المبيعات", "output": "result"}
            ],
        },
    )
    assert filtered["record"]["result"]["rows"] == [{"result": "12.75"}]
    assert filtered["record"]["exclusions"]["filter"] == 1
    assert filtered["record"]["periods_included"] == ["Q1"]
    assert filtered["record"]["filters"][0]["value"] == "Q1"


def test_arabic_indic_digits_and_negative_values_stay_exact():
    text = "البند,القيمة\nأ,١٢٫٥٠\nب,-1.50\nج,١٫٥٠\n"
    result = execute(
        text,
        {"aggregates": [{"function": "sum", "column": "القيمة", "output": "result"}]},
        filename="أرقام.csv",
    )
    assert result["record"]["result"]["rows"] == [{"result": "12.50"}]
    assert "١٢" not in result["record"]["result"]["rows"][0]["result"]


def test_leading_zero_identifiers_are_not_summed_and_are_preserved():
    source = dataset("id,amount\n001,10.00\n002,5.00\n")
    with pytest.raises(DatasetError, match="column_not_numeric"):
        bind_plan(
            OperationPlan.model_validate(
                {
                    "aggregates": [
                        {"function": "sum", "column": "id", "output": "result"}
                    ]
                }
            ),
            source,
        )
    projected = execute(
        "id,amount\n001,10.00\n002,5.00\n",
        {"project": ["id"]},
    )
    assert [row["id"] for row in projected["record"]["result"]["rows"]] == [
        "001",
        "002",
    ]


def test_margin_currency_and_forecast_are_not_guessed():
    source = dataset("item,revenue,cost\na,100.00,40.00\n")
    chosen = OperationPlan.model_validate(
        {"aggregates": [{"function": "sum", "column": "revenue", "output": "margin"}]}
    )
    assert margin_ambiguous("What is the margin?", chosen, source)
    assert margin_ambiguous("ما الهامش؟", chosen, source)
    stored = dataset("item,margin\na,0.25\n")
    direct = OperationPlan.model_validate(
        {"aggregates": [{"function": "sum", "column": "margin", "output": "result"}]}
    )
    assert margin_ambiguous("What is the margin?", direct, stored) is False
    currencies = dataset("item,currency,amount\na,USD,10.00\nb,EGP,5.00\n")
    mixed = bind_plan(
        OperationPlan.model_validate(
            {
                "aggregates": [
                    {"function": "sum", "column": "amount", "output": "result"}
                ]
            }
        ),
        currencies,
    )
    with pytest.raises(ClarifyError, match="mixed_currency"):
        run(currencies, mixed)
    single = execute(
        "item,currency,amount\na,USD,10.00\nb,USD,5.00\n",
        {"aggregates": [{"function": "sum", "column": "amount", "output": "result"}]},
    )
    assert single["record"]["result"]["rows"] == [{"result": "15.00"}]
    assert single["record"]["units"] == "USD"
    assert interpretation_limited("لماذا انخفضت المبيعات؟", None)
    assert interpretation_limited("Forecast next quarter", None)
    assert clarification("ambiguous_margin", "ما الهامش؟", ["revenue"])
    assert "لم يُحسب" in clarification("ambiguous_margin", "ما الهامش؟", ["revenue"])


def test_injection_attempts_do_not_become_sql_or_formulas():
    source = dataset("id,amount\n001,10.00\n")
    with pytest.raises(ValueError):
        OperationPlan.model_validate(
            {
                "aggregates": [
                    {"function": "sum", "column": "amount", "output": "result"}
                ],
                "sql": "DROP TABLE enterprise.users",
            }
        )
    chosen = bind_plan(
        OperationPlan.model_validate(
            {
                "filters": [
                    {
                        "column": "id",
                        "operator": "eq",
                        "value": "001'; DROP TABLE enterprise.users; --",
                    }
                ],
                "aggregates": [
                    {"function": "count", "column": "id", "output": "result"}
                ],
            }
        ),
        source,
    )
    sql, params = _where(source, chosen)
    assert "DROP" not in sql and "postgres" not in sql.casefold()
    assert params == ["001'; DROP TABLE enterprise.users; --"]
    exported = result_csv(
        ["label", "result"],
        [{"label": "=cmd|'/c calc'!A0", "result": "-1.50"}],
    )
    parsed = list(csv.reader(io.StringIO(exported.lstrip("\ufeff"))))
    assert parsed[1][0].startswith("'")
    assert parsed[1][1] == "-1.50"
    assert escape_cell("@cmd").startswith("'")
    assert escape_cell("\t=1").startswith("'")
    assert escape_cell("قلم") == "قلم"


def test_limits_stop_without_a_partial_result():
    wide = "a,b\n" + "".join(f"{index}," for index in range(101)).rstrip(",") + "\n"
    # 101 columns is rejected while classifying the header width.
    with pytest.raises(DatasetError, match="not_csv"):
        dataset("h" + ",h" * 100 + "\n" + "1" + ",1" * 100)
    with pytest.raises(ValueError):
        OperationPlan.model_validate(
            {
                "filters": [
                    {"column": "amount", "operator": "eq", "value": "1"}
                    for _ in range(9)
                ],
                "aggregates": [
                    {"function": "count", "column": None, "output": "result"}
                ],
            }
        )
    text = "region,amount\nA,1.00\nB,2.00\nC,3.00\n"
    source = dataset(text)
    chosen = bind_plan(
        OperationPlan.model_validate(
            {
                "group_by": ["region"],
                "aggregates": [
                    {"function": "sum", "column": "amount", "output": "result"}
                ],
            }
        ),
        source,
    )
    with pytest.raises(DatasetError, match="output_limit"):
        run(source, chosen, max_output_rows=1)

    class Slow:
        def execute(self, sql, params):
            import time

            time.sleep(0.2)
            return self

        def fetchall(self):
            return []

        def interrupt(self):
            self.interrupted = True

    with pytest.raises(DatasetError, match="runtime_limit"):
        _query(Slow(), "SELECT 1", [], 0.01)
    huge = ["n"] + ["1.00"] * 1001
    with pytest.raises(DatasetError, match="source_limit"):
        dataset("\n".join(huge))
    assert wide.startswith("a,b")


def test_sandbox_locks_duckdb_and_records_the_authorized_version():
    outcome = execute(
        "sku,qty\nA,2\nB,3\n",
        {"aggregates": [{"function": "sum", "column": "qty", "output": "result"}]},
    )
    sandbox = outcome["record"]["sandbox"]
    assert sandbox["external_access"] is False
    assert sandbox["extension_autoload"] is False
    assert sandbox["extension_autoinstall"] is False
    assert sandbox["settings_locked"] is True
    assert sandbox["threads"] == 1
    assert sandbox["postgresql_accessed"] is False
    assert sandbox["os_cgroup_enforced"] is False
    assert sandbox["job_local_snapshot"] is True
    assert outcome["record"]["operation"]["aggregates"][0]["function"] == "sum"
    assert outcome["citations"][0]["location"]["kind"] == "csv_row"
    assert outcome["citations"][0]["row"] == 2


def test_fixture_planner_selects_named_column_and_refuses_margin():
    profile = {
        "columns": [
            {"name": "المنطقة", "type": "text"},
            {"name": "المبيعات", "type": "decimal"},
        ]
    }
    chosen = fixture_plan("ما مجموع المبيعات؟", profile)
    assert chosen["aggregates"][0]["column"] == "المبيعات"
    assert chosen["aggregates"][0]["function"] == "sum"
    assert fixture_plan("What is the margin?", profile)["action"] == "clarify"
    assert fixture_plan("Forecast sales", profile)["action"] == "clarify"


def test_calculation_planner_keeps_the_groq_first_byok_chain():
    source = dataset("item,amount\na,1.00\n", filename="مبيعات.csv")
    gateway = OpenRouterGateway(privacy="nonconfidential-development")
    payload = gateway.payload(planner_request("ما مجموع amount؟", source))
    assert payload["model"] == MODELS[0] == "openai/gpt-oss-120b"
    assert payload["models"] == list(MODELS)
    assert payload["models"][1] == "google/gemini-3.8-flash"
    assert payload["provider"]["order"] == list(PROVIDERS)
    assert payload["provider"]["only"] == ["groq", "google-ai-studio"]
    assert payload["provider"]["allow_fallbacks"] is True
    assert "CALCULATION_PLANNER" in payload["messages"][0]["content"]
    assert "10.00" not in payload["messages"][1]["content"]
    instructions = assemble("analysis")
    assert {"ANA-004", "ANA-005", "ANA-006", "ANA-007"} <= set(instructions.rule_ids)
    assert "allowlisted" in instructions.text
