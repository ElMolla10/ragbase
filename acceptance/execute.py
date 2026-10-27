"""Execute acceptance cases.

The default session uses the development fixture gateway in process. A live
provider is possible only through an explicit HTTP opt-in, and never in CI.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
from pathlib import Path

ELEMENT_RE = re.compile(r":::element ([^\s]+)([^\n]*)\n(.*?):::", re.S)
CI_VARIABLES = ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "BUILDKITE")


class LiveProviderError(RuntimeError):
    """Live generation was requested where this runner must not call it."""


class AcceptanceConfigError(RuntimeError):
    """The runner was pointed at an API without the configuration it needs."""


def ci_environment() -> bool:
    return any(os.environ.get(name) for name in CI_VARIABLES)


def load_elements(root: Path) -> dict[str, dict]:
    elements = {}
    fixture_dir = root / "fixtures"
    for path in sorted(fixture_dir.glob("*")):
        if path.suffix.lower() == ".csv" or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for match in ELEMENT_RE.finditer(text):
            element_id, flags, body = match.group(1), match.group(2), match.group(3)
            elements[element_id] = {
                "id": element_id,
                "text": body.strip(),
                "flags": flags.strip(),
                "fixture": f"fixtures/{path.name}",
            }
    return elements


def retrieve(elements: list[dict], question: str, *, limit: int = 10) -> list[dict]:
    """Token overlap using the enterprise normalizer. Not PostgreSQL search."""
    from backend.search.index import query_terms

    terms = query_terms(question)
    if not terms:
        return []
    ranked = []
    for element in elements:
        have = _tokens(element["text"])
        score = sum(1 for term in terms if term in have)
        if score:
            ranked.append((-score, element["id"], element))
    ranked.sort()
    return [item[2] for item in ranked[:limit]]


def _tokens(text: str) -> set[str]:
    from backend.search.index import normalize

    return set(re.findall(r"[^\W_]+", normalize(text), flags=re.UNICODE))


class FixtureSession:
    """In-process fixture gateway. Does not open a network connection."""

    def __init__(self, root: Path):
        self.root = root

    async def run(self, case: dict, elements: dict) -> dict:
        observation = _blank()
        try:
            if case["task"] in {"ocr_pending", "chart_pending"}:
                observation["pending"] = True
            elif case["task"] == "denied_access":
                _deny(case, observation)
            elif case["task"] == "calculation":
                await _calculate(self.root, case, observation)
            elif case["task"] == "coverage":
                await _cover(case, elements, observation)
            else:
                await _answer(case, elements, observation)
        except Exception:
            observation["error"] = "execution_failed"
        return observation


def _blank() -> dict:
    return {
        "pending": False,
        "retrieved_ids": [],
        "retrieved_texts": [],
        "citations": [],
        "insufficient": False,
        "denied": False,
        "denial_body": "",
        "leaked": False,
        "clarification": False,
        "numeric_value": None,
        "coverage_label": None,
        "coverage_complete": None,
        "coverage_sections": [],
        "error": None,
    }


def _scope(case: dict, elements: dict) -> tuple[list[dict], bool]:
    allowed = []
    saw_denied = False
    for document in case["documents"]:
        if case["actor"] not in document["authorized_for"]:
            saw_denied = True
            continue
        allowed.extend(
            element
            for element in elements.values()
            if element["fixture"] == document["fixture"]
        )
    return allowed, saw_denied and not allowed


def _deny(case: dict, observation: dict) -> None:
    observation["denied"] = True
    observation["denial_body"] = "Selected source not available"
    observation["insufficient"] = True
    observation["leaked"] = any(
        snippet and snippet in observation["denial_body"]
        for snippet in case["gold"]["forbidden_snippets"]
    )


async def _answer(case: dict, elements: dict, observation: dict) -> None:
    from backend.answers.fixtures import FixtureQuoteGateway
    from backend.gateway.contracts import GatewayRequest
    from backend.policy import assemble

    allowed, denied = _scope(case, elements)
    if denied:
        _deny(case, observation)
        return
    hits = retrieve(allowed, case["question"])
    observation["retrieved_ids"] = [hit["id"] for hit in hits]
    observation["retrieved_texts"] = [hit["text"] for hit in hits]
    evidence = tuple(
        json.dumps({"source_id": hit["id"], "text": hit["text"]}, ensure_ascii=False)
        for hit in hits
    )
    request = GatewayRequest(
        assemble("retrieval"),
        case["question"],
        evidence=evidence,
        max_output_tokens=256,
    )
    payload = json.loads(await _read_gateway(FixtureQuoteGateway(), request))
    if payload.get("insufficient") or not payload.get("claims"):
        observation["insufficient"] = True
        return
    observation["citations"] = [
        {"id": claim.get("source_id"), "text": claim.get("quote") or ""}
        for claim in payload["claims"]
    ]


async def _calculate(root: Path, case: dict, observation: dict) -> None:
    from backend.analysis.engine import run
    from backend.analysis.plan import (
        OperationPlan,
        bind_plan,
        interpretation_limited,
        margin_ambiguous,
    )
    from backend.analysis.planner import propose
    from backend.analysis.profile import ClarifyError, build_dataset
    from backend.answers.fixtures import FixtureQuoteGateway

    fixture = root / case["documents"][0]["fixture"]
    dataset = build_dataset(
        version_id=case["documents"][0]["document_id"],
        document_id=case["documents"][0]["document_id"],
        filename=fixture.name,
        sha256="ab" * 32,
        extractor="csv-utf8-strings-v1",
        warnings=["csv_strings_only_no_header_type_or_formula_inference"],
        pages=_csv_pages(fixture.read_text(encoding="utf-8")),
    )
    plan_payload = case["gold"]["plan"]
    plan = OperationPlan.model_validate(plan_payload) if plan_payload else None
    if interpretation_limited(case["question"], plan):
        _clarify(observation)
        return
    if plan is None:
        plan = await propose(FixtureQuoteGateway(), case["question"], dataset)
    if plan.action != "calculate":
        _clarify(observation)
        return
    plan = bind_plan(plan, dataset)
    if margin_ambiguous(case["question"], plan, dataset):
        _clarify(observation)
        return
    try:
        outcome = run(dataset, plan)
    except ClarifyError:
        _clarify(observation)
        return
    observation["numeric_value"] = outcome["record"]["result"]["rows"][0]["result"]


def _clarify(observation: dict) -> None:
    observation["clarification"] = True
    observation["numeric_value"] = None
    observation["insufficient"] = True


def _csv_pages(text: str) -> list[dict]:
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


async def _cover(case: dict, elements: dict, observation: dict) -> None:
    from backend.answers.fixtures import FixtureQuoteGateway
    from backend.coverage.planner import (
        MAX_MAP_CALLS,
        assign_aliases,
        build_sections,
        expand_sections,
        split_cited,
        verdict,
    )
    from backend.coverage.prompt import (
        fits_request,
        make_coverage_request,
        parse_section,
        plan_map,
    )
    from backend.gateway.contracts import GatewayError
    from backend.policy import assemble

    pages = []
    ordinal = 1
    for document in case["documents"]:
        owned = [
            element
            for element in elements.values()
            if element["fixture"] == document["fixture"]
        ]
        for element in owned:
            ocr = "ocr" in element["flags"].split()
            content = "" if ocr else element["text"]
            page = {
                "version_id": document["document_id"],
                "ordinal": ordinal,
                "content": content,
                "location": {"kind": "markdown_section", "section": element["id"]},
                "warnings": (
                    ["no_native_text_ocr_or_visual_review_required"] if ocr else []
                ),
                "job_warnings": [],
                "filename": document["fixture"],
                "document_id": document["document_id"],
                "sha256": "ab" * 32,
                "chunks": [],
            }
            if content.strip():
                page["chunks"] = [
                    {
                        "id": element["id"],
                        "part": 0,
                        "char_start": 0,
                        "char_end": len(content),
                        "content": content,
                        "index_version": "native-lexical-v1",
                    }
                ]
            pages.append(page)
            ordinal += 1
    instructions = assemble("coverage")
    sections, _notes = build_sections(pages, "acceptance")
    output_tokens, budget = 256, 4096

    def fits_text(content: str) -> bool:
        return fits_request(
            instructions,
            case["question"],
            [{"alias": "S1", "text": content}],
            kind="section",
            output_tokens=output_tokens,
            budget=budget,
        )

    sections = expand_sections(sections, fits_text)
    batches = plan_map(
        sections,
        instructions=instructions,
        question=case["question"],
        output_tokens=output_tokens,
        budget=budget,
        max_calls=MAX_MAP_CALLS,
    )
    assign_aliases(batches)
    gateway = FixtureQuoteGateway()
    summaries = []
    for batch in batches:
        items = [
            {"alias": f"S{index}", "text": section["content"]}
            for index, section in enumerate(batch, 1)
        ]
        request = make_coverage_request(
            instructions,
            case["question"],
            items,
            kind="section",
            max_output_tokens=output_tokens,
        )
        try:
            raw = await _read_gateway(gateway, request)
            text_value, aliases = parse_section(raw, batch)
        except GatewayError:
            _mark(batch, "gap", "section_processing_failed")
            continue
        cited, missed = split_cited(batch, aliases)
        if cited:
            _mark(cited, "covered", None)
            summaries.append({"text": text_value, "aliases": aliases})
        if missed:
            _mark(missed, "gap", "section_processing_failed")
    root = None
    extra = []
    if summaries:
        root = await _reduce(
            summaries, instructions, case["question"], output_tokens, budget, gateway
        )
        if root is None:
            extra.append("reduction_incomplete")
    complete = verdict(sections, reduction_complete=root is not None and not extra)
    if extra:
        complete = False
    observation["coverage_complete"] = complete
    observation["coverage_label"] = "complete" if complete else "partial"
    observation["coverage_sections"] = [
        {
            "id": (section.get("location") or {}).get("section"),
            "status": section["status"],
            "gap_code": section.get("gap_code"),
        }
        for section in sections
    ]


def _mark(sections: list[dict], status: str, code: str | None) -> None:
    from backend.coverage.planner import fail_section

    for section in sections:
        if status == "gap" and section["status"] == "covered":
            continue
        if status == "covered":
            section["status"] = "covered"
            section["gap_code"] = None
        elif section["status"] == "pending":
            fail_section(section, code)


async def _reduce(summaries, instructions, question, output_tokens, budget, gateway):
    from backend.coverage.planner import MAX_REDUCTION_CALLS
    from backend.coverage.prompt import (
        make_coverage_request,
        parse_reduction,
        plan_reduction,
    )
    from backend.gateway.contracts import GatewayError

    level = summaries
    calls = 0
    while True:
        remaining = MAX_REDUCTION_CALLS - calls
        groups, overflow = plan_reduction(
            level,
            instructions=instructions,
            question=question,
            output_tokens=output_tokens,
            budget=budget,
            max_calls=remaining,
        )
        if (
            overflow
            or not groups
            or (len(level) > 1 and all(len(group) == 1 for group in groups))
        ):
            return None
        reduced = []
        for group in groups:
            items = [
                {"alias": alias, "text": summary["text"]}
                for summary in group
                for alias in summary["aliases"]
            ]
            request = make_coverage_request(
                instructions,
                question,
                items,
                kind="reduction",
                max_output_tokens=output_tokens,
            )
            try:
                allowed = {alias for summary in group for alias in summary["aliases"]}
                text_value, aliases = parse_reduction(
                    await _read_gateway(gateway, request), allowed
                )
            except GatewayError:
                return None
            calls += 1
            reduced.append({"text": text_value, "aliases": aliases})
        level = reduced
        if len(level) == 1:
            return level[0]


async def _read_gateway(gateway, request) -> str:
    from backend.gateway.contracts import GatewayError

    parts = []
    finished = False
    async for event in gateway.stream(request):
        if event.kind == "text":
            parts.append(event.text)
        elif event.kind == "done":
            if event.finish_reason != "stop":
                raise GatewayError("Incomplete answer")
            finished = True
    if not finished:
        raise GatewayError("Incomplete answer")
    return "".join(parts)


class HttpEnterpriseClient:
    """Call a configured enterprise API. Live mode is opt-in and blocked in CI."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        live_provider: bool,
        version_map: dict[str, str],
        workspace: str,
        transport=None,
    ):
        if live_provider and ci_environment():
            raise LiveProviderError("Live providers are disabled in CI")
        if not token or len(token) < 16:
            raise AcceptanceConfigError(
                "Acceptance HTTP mode needs a development token"
            )
        if not version_map or not workspace:
            raise AcceptanceConfigError(
                "Acceptance HTTP mode needs a workspace and version map"
            )
        self.base_url = base_url.rstrip("/")
        self._token = token
        self.live_provider = live_provider
        self.version_map = version_map
        self.workspace = workspace
        self._transport = transport

    def redact(self, message: str) -> str:
        if self._token and self._token in message:
            return message.replace(self._token, "[redacted]")
        return message

    async def check_gateway(self) -> str:
        import httpx

        async with httpx.AsyncClient(
            transport=self._transport, timeout=10, trust_env=False
        ) as client:
            response = await client.get(self.base_url + "/health/ready")
        if response.status_code != 200:
            raise AcceptanceConfigError("Enterprise readiness check failed")
        generation = response.json().get("generation")
        if generation == "fixture":
            return generation
        if generation == "openrouter":
            if not self.live_provider:
                raise LiveProviderError("Live provider requires explicit opt-in")
            if ci_environment():
                raise LiveProviderError("Live providers are disabled in CI")
            return generation
        raise LiveProviderError("Acceptance cannot use this generation mode")

    async def run(self, case: dict, elements: dict) -> dict:
        import httpx

        observation = _blank()
        if case["task"] in {"ocr_pending", "chart_pending"}:
            observation["pending"] = True
            return observation
        versions = []
        for document in case["documents"]:
            version = self.version_map.get(document["fixture"])
            if version:
                versions.append(version)
        if not versions:
            observation["error"] = "api_versions_not_configured"
            return observation
        headers = {"authorization": f"Bearer {self._token}"}
        try:
            async with httpx.AsyncClient(
                transport=self._transport, timeout=30, trust_env=False
            ) as client:
                if case["task"] == "denied_access":
                    response = await client.post(
                        f"{self.base_url}/api/workspaces/{self.workspace}/search",
                        headers=headers,
                        json={"question": case["question"], "version_ids": versions},
                    )
                    body = response.text
                    observation["denial_body"] = body
                    observation["denied"] = response.status_code in {401, 403, 404}
                    observation["leaked"] = any(
                        snippet in body
                        for snippet in case["gold"]["forbidden_snippets"]
                    )
                    return observation
                response = await client.post(
                    f"{self.base_url}/api/workspaces/{self.workspace}/search",
                    headers=headers,
                    json={"question": case["question"], "version_ids": versions},
                )
        except httpx.HTTPError as exc:
            observation["error"] = "execution_failed"
            observation["denial_body"] = self.redact(str(exc))
            return observation
        if response.status_code in {401, 403, 404}:
            observation["denied"] = True
            observation["denial_body"] = response.text
            return observation
        if response.status_code != 200:
            observation["error"] = "execution_failed"
            return observation
        hits = response.json().get("hits") or []
        observation["retrieved_ids"] = [hit.get("source_id", "") for hit in hits[:10]]
        observation["retrieved_texts"] = [hit.get("content", "") for hit in hits[:10]]
        if not hits:
            observation["insufficient"] = True
        return observation
