"""Run the acceptance corpus and write a versioned report."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from acceptance.corpus import load_corpus
from acceptance.execute import (
    AcceptanceConfigError,
    FixtureSession,
    HttpEnterpriseClient,
    LiveProviderError,
    ci_environment,
    load_elements,
)
from acceptance.metrics import score_case
from acceptance.report import build_report, write_report

ROOT = Path(__file__).resolve().parents[1]


def execution_record(*, mode: str, live_provider: bool) -> dict:
    return {
        "mode": mode,
        "gateway": (
            "backend.answers.fixtures.FixtureQuoteGateway"
            if mode == "fixture_gateway_in_process"
            else "configured_enterprise_api"
        ),
        "retrieval": (
            "local_token_overlap_using_backend.search.index.normalize_and_query_terms"
            if mode == "fixture_gateway_in_process"
            else "enterprise_api_search"
        ),
        "retrieval_limitation": (
            "Fixture retrieval is not PostgreSQL tsvector, dense retrieval, or a "
            "context-window position test. VAL-003 remains pending."
        ),
        "calculation": "trusted_duckdb_after_fixture_planner_or_gold_plan",
        "coverage": "backend.coverage_planner_and_verdict_with_fixture_gateway",
        "live_provider_called": live_provider,
        "weights_downloaded": False,
        "production_qualified": False,
        "qualifies_company_readiness": False,
        "qualifies_arabic_reliability": False,
        "qualifies_context_or_latency": False,
    }


async def run_cases(session, cases: list[dict], elements: dict) -> list[dict]:
    outcomes = []
    for case in cases:
        observation = await session.run(case, elements)
        outcomes.append(score_case(case, observation, elements))
    return outcomes


def run(
    corpus_dir: Path,
    output: Path,
    *,
    api_url: str | None = None,
    live_provider: bool = False,
    token: str | None = None,
    version_map: dict | None = None,
    workspace: str | None = None,
    transport=None,
) -> dict:
    if live_provider and ci_environment():
        raise LiveProviderError("Live providers are disabled in CI")
    if live_provider and not api_url:
        raise LiveProviderError("A live provider requires an enterprise API URL")
    manifest, cases = load_corpus(corpus_dir)
    elements = load_elements(corpus_dir)
    if api_url:
        if not token or not version_map or not workspace:
            raise AcceptanceConfigError(
                "HTTP acceptance needs a token, workspace, and version map"
            )
        session = HttpEnterpriseClient(
            api_url,
            token,
            live_provider=live_provider,
            version_map=version_map,
            workspace=workspace,
            transport=transport,
        )
        mode = "enterprise_http_live" if live_provider else "enterprise_http_fixture"
    else:
        session = FixtureSession(corpus_dir)
        mode = "fixture_gateway_in_process"
        live_provider = False

    async def _execute():
        if api_url:
            await session.check_gateway()
        return await run_cases(session, cases, elements)

    outcomes = asyncio.run(_execute())
    payload = build_report(
        manifest,
        cases,
        outcomes,
        execution_record(mode=mode, live_provider=live_provider),
    )
    if payload["production_qualified"] or any(
        gate["decision"] == "pass" or gate["production_qualified"]
        for gate in payload["gates"]
    ):
        raise AcceptanceConfigError("This runner cannot mark production qualified")
    write_report(output, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the synthetic acceptance corpus")
    parser.add_argument("--corpus", default=str(ROOT / "eval/acceptance/v1"))
    parser.add_argument(
        "--output",
        default=str(ROOT / "eval/acceptance/reports/v1-fixture-baseline.json"),
    )
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--live-provider", action="store_true")
    parser.add_argument("--version-map", default=None)
    parser.add_argument("--workspace", default=None)
    parser.add_argument(
        "--token-env",
        default="ACCEPTANCE_DEV_TOKEN",
        help="Environment variable that holds the development bearer token",
    )
    args = parser.parse_args(argv)
    version_map = None
    if args.version_map:
        version_map = json.loads(Path(args.version_map).read_text(encoding="utf-8"))
    token = os.environ.get(args.token_env) if args.api_url else None
    try:
        payload = run(
            Path(args.corpus),
            Path(args.output),
            api_url=args.api_url,
            live_provider=args.live_provider,
            token=token,
            version_map=version_map,
            workspace=args.workspace,
        )
    except (LiveProviderError, AcceptanceConfigError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(
        "cases="
        f"{payload['corpus']['cases']} arabic_or_mixed={payload['corpus']['arabic_or_mixed']} "
        "production_qualified=false"
    )
    for gate in payload["gates"]:
        print(f"{gate['id']}={gate['decision']}")
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
