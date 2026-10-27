"""Authorized CSV calculations against PostgreSQL RLS. The model fixture is local."""

import csv
import io
import json
from decimal import Decimal
from urllib.parse import quote
from uuid import uuid4

from sqlalchemy import text

from enterprise_integration_tests.conftest import headers
from enterprise_integration_tests.test_ingestion_jobs import extraction_path, process


def plan(column="amount", function="sum", filters=None, group_by=None):
    body = {
        "version_id": None,
        "client_request_id": str(uuid4()),
        "plan": {
            "action": "calculate",
            "aggregates": [
                {"function": function, "column": column, "output": "result"}
            ],
        },
    }
    if filters:
        body["plan"]["filters"] = filters
    if group_by:
        body["plan"]["group_by"] = group_by
    return body


async def ready_csv(world, text, filename="sales.csv", user="alice"):
    response = await world["client"].post(
        f"/api/workspaces/{world['ids']['a']}/documents",
        headers=headers(world, user, filename),
        content=text.encode(),
    )
    assert response.status_code == 201, response.text
    record = response.json()
    assert await process(world, user)
    state = (
        await world["client"].get(
            extraction_path(world, record), headers=headers(world, user)
        )
    ).json()
    assert state["status"] == "extracted" and state["retrieval_ready"]
    return record


async def calculate(world, record, body, user="alice", workspace="a"):
    body = dict(body)
    body["version_id"] = record["version_id"]
    return await world["client"].post(
        f"/api/workspaces/{world['ids'][workspace]}/calculations",
        headers=headers(world, user),
        json=body,
    )


async def test_exact_csv_calculation_is_downloadable_and_cites_rows(world):
    record = await ready_csv(
        world,
        "id,amount\n001,10.10\n002,20.20\n003,0.30\n",
    )
    response = await calculate(world, record, plan())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "completed"
    assert body["production_qualified"] is False
    assert body["model_called"] is False
    assert body["record"]["result"]["rows"] == [{"result": "30.60"}]
    assert Decimal(body["record"]["result"]["rows"][0]["result"]) == Decimal("30.60")
    assert body["record"]["included_row_count"] == 3
    assert body["record"]["excluded_row_count"] == 1
    assert body["record"]["units"] == "unspecified"
    assert body["record"]["dataset"]["sha256"] == record["sha256"]
    assert body["record"]["dataset"]["version_id"] == record["version_id"]
    assert [item["row"] for item in body["citations"]] == [2, 3, 4]
    assert body["citations"][0]["quote"].startswith('["001"')
    preview = await world["client"].get(
        body["citations"][0]["preview_url"], headers=headers(world)
    )
    assert preview.status_code == 200 and "001" in preview.text
    table = await world["client"].get(
        body["downloads"]["table"], headers=headers(world)
    )
    parsed = list(csv.reader(io.StringIO(table.text.lstrip("\ufeff"))))
    assert parsed == [["result"], ["30.60"]]
    stored = await world["client"].get(
        body["downloads"]["record"], headers=headers(world)
    )
    assert stored.status_code == 200
    assert stored.json()["result"]["rows"][0]["result"] == "30.60"
    assert stored.json()["sandbox"]["postgresql_accessed"] is False
    assert stored.json()["sandbox"]["external_access"] is False


async def test_arabic_question_uses_fixture_plan_and_exact_group_totals(world):
    record = await ready_csv(
        world,
        "المنطقة,المبيعات\nالقاهرة,10.50\nالإسكندرية,2.25\nالقاهرة,1.25\nالمجموع,14.00\n",
        filename=quote("مبيعات.csv"),
    )
    response = await calculate(
        world,
        record,
        {"question": "ما مجموع المبيعات؟", "client_request_id": str(uuid4())},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_called"] is True
    assert body["record"]["result"]["rows"] == [{"result": "14.00"}]
    assert body["record"]["exclusions"]["subtotal"] == 1
    assert "14.00" in body["answer"]
    grouped = await calculate(
        world,
        record,
        plan(
            column="المبيعات",
            group_by=["المنطقة"],
        ),
    )
    rows = {
        row["المنطقة"]: row["result"]
        for row in grouped.json()["record"]["result"]["rows"]
    }
    assert rows == {"القاهرة": "11.75", "الإسكندرية": "2.25"}
    chat = await world["client"].post(
        f"/api/workspaces/{world['ids']['a']}/answers",
        headers=headers(world),
        json={
            "question": "/calculate ما مجموع المبيعات؟",
            "version_ids": [record["version_id"]],
            "client_request_id": str(uuid4()),
        },
    )
    assert chat.status_code == 200, chat.text
    events = [json.loads(line) for line in chat.text.splitlines()]
    answer = next(event for event in events if event["event"] == "answer")
    assert answer["mode"] == "spreadsheet_calculation"
    assert "14.00" in answer["answer"]
    assert {claim["location"]["row"] for claim in answer["claims"]} == {2, 3, 4}
    assert answer["metadata"]["fixture"] is True
    assert answer["metadata"]["production_qualified"] is False


async def test_margin_forecast_and_mixed_currency_do_not_invent_numbers(world):
    record = await ready_csv(world, "item,revenue,cost\na,100.00,40.00\n", "margin.csv")
    margin = await calculate(
        world,
        record,
        {"question": "What is the margin?", "client_request_id": str(uuid4())},
    )
    assert margin.status_code == 200
    assert margin.json()["status"] == "clarification"
    assert margin.json()["record"]["result"] is None
    rendered = json.dumps(margin.json()["record"])
    assert "60" not in rendered and "0.6" not in rendered
    forecast = await calculate(
        world,
        record,
        {
            "question": "Forecast next quarter revenue",
            "client_request_id": str(uuid4()),
        },
    )
    assert forecast.json()["status"] == "clarification"
    assert forecast.json()["record"]["result"] is None
    currencies = await ready_csv(
        world, "item,currency,amount\na,USD,10.00\nb,EGP,5.00\n", "currency.csv"
    )
    mixed = await calculate(world, currencies, plan())
    assert mixed.status_code == 200
    assert mixed.json()["status"] == "clarification"
    assert mixed.json()["record"]["result"] is None
    filtered = await calculate(
        world,
        currencies,
        plan(filters=[{"column": "currency", "operator": "eq", "value": "USD"}]),
    )
    assert filtered.json()["record"]["result"]["rows"] == [{"result": "10.00"}]
    assert filtered.json()["record"]["units"] == "USD"


async def test_injection_limits_and_denied_access_do_not_change_postgres(world):
    record = await ready_csv(world, "id,amount\n001,10.00\n002,5.00\n")
    async with world["owner"].connect() as conn:
        users_before = await conn.scalar(text("SELECT count(*) FROM enterprise.users"))
        documents_before = await conn.scalar(
            text("SELECT count(*) FROM enterprise.documents")
        )
    hostile = await calculate(
        world,
        record,
        {
            "client_request_id": str(uuid4()),
            "plan": {
                "action": "calculate",
                "filters": [
                    {
                        "column": "id",
                        "operator": "eq",
                        "value": "001'; DELETE FROM enterprise.users; --",
                    }
                ],
                "aggregates": [
                    {"function": "sum", "column": "amount", "output": "result"}
                ],
                "sql": "SELECT * FROM enterprise.users",
            },
        },
    )
    assert hostile.status_code == 422
    async with world["owner"].connect() as conn:
        assert (
            await conn.scalar(text("SELECT count(*) FROM enterprise.users"))
            == users_before
        )
        assert (
            await conn.scalar(text("SELECT count(*) FROM enterprise.documents"))
            == documents_before
        )
    denied = await world["client"].post(
        f"/api/workspaces/{world['ids']['b']}/calculations",
        headers=headers(world, "bob"),
        json=plan() | {"version_id": record["version_id"]},
    )
    assert denied.status_code == 404
    hidden = await world["client"].get(
        f"/api/workspaces/{world['ids']['a']}/calculations/{uuid4()}/table.csv",
        headers=headers(world, "bob"),
    )
    assert hidden.status_code == 404
    owned = await calculate(world, record, plan())
    private = await world["client"].get(
        owned.json()["downloads"]["table"], headers=headers(world, "viewer")
    )
    assert private.status_code == 404
    viewer = await calculate(world, record, plan(), user="viewer")
    assert viewer.status_code == 200
    assert viewer.json()["record"]["result"]["rows"] == [{"result": "15.00"}]
    async with world["owner"].begin() as conn:
        await conn.execute(
            text("DELETE FROM enterprise.memberships WHERE user_id=:id"),
            {"id": world["ids"]["alice"]},
        )
    revoked = await world["client"].get(
        owned.json()["downloads"]["record"], headers=headers(world)
    )
    assert revoked.status_code == 404


async def test_unready_csv_pdf_and_formula_export_are_explicit(world):
    ragged = await world["client"].post(
        f"/api/workspaces/{world['ids']['a']}/documents",
        headers=headers(world, filename="ragged.csv"),
        content=b"a,b\n1,2,3\n",
    )
    assert ragged.status_code == 201 and await process(world)
    blocked = await calculate(world, ragged.json(), plan(column="a"))
    assert blocked.status_code == 409
    pdf = await world["client"].post(
        f"/api/workspaces/{world['ids']['a']}/documents",
        headers=headers(world, filename="notes.pdf"),
        content=b"%PDF-1.4 not a real calculation source",
    )
    # Invalid PDF fails ingestion; calculation must not treat it as a dataset.
    if pdf.status_code == 201:
        await process(world)
        rejected = await calculate(world, pdf.json(), plan(column="a"))
        assert rejected.status_code in {404, 409}
    formula = await ready_csv(
        world, "label,amount\n=cmd,10.00\nsafe,2.50\n", "formula.csv"
    )
    exported = await calculate(
        world,
        formula,
        {
            "client_request_id": str(uuid4()),
            "plan": {"action": "calculate", "project": ["label", "amount"]},
        },
    )
    assert exported.status_code == 200, exported.text
    table = await world["client"].get(
        exported.json()["downloads"]["table"], headers=headers(world)
    )
    parsed = list(csv.reader(io.StringIO(table.content.decode().lstrip("\ufeff"))))
    assert parsed[1][0].startswith("'")
    assert parsed[1][1] == "10.00"
    assert any(row[0] == "safe" for row in parsed[1:])


class BoomGateway:
    async def stream(self, request):
        raise AssertionError("explicit plans must not call a model")
        yield  # pragma: no cover


async def test_explicit_plan_does_not_call_the_model(world):
    record = await ready_csv(world, "amount\n1.00\n2.00\n", "plain.csv")
    world["app"].state.gateway = BoomGateway()
    response = await calculate(world, record, plan(column="amount"))
    assert response.status_code == 200
    assert response.json()["model_called"] is False
    assert response.json()["record"]["result"]["rows"] == [{"result": "3.00"}]
