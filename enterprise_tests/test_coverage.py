"""Exhaustive coverage planning and citation contracts. No provider calls."""

import json
from uuid import uuid4

import pytest

from backend.answers.fixtures import FixtureQuoteGateway
from backend.answers.grounding import validate_answer
from backend.coverage.planner import (
    build_sections,
    expand_sections,
    gap_counts,
    scope_notice,
    split_cited,
    verdict,
)
from backend.coverage.prompt import (
    fits_request,
    make_coverage_request,
    parse_section,
    plan_map,
    release_answer,
)
from backend.gateway.contracts import GatewayError
from backend.policy import assemble
from enterprise_tests.test_quote_answers import hit


def page(content, *, chunks=None, warnings=None, kind="text", ordinal=1, version="v1"):
    location = {"kind": kind, "line_start": 1, "line_end": 2}
    if kind == "pdf_page":
        location = {"kind": kind, "page": ordinal}
    elif kind == "csv_row":
        location = {
            "kind": kind,
            "row": ordinal,
            "line_start": ordinal,
            "line_end": ordinal,
        }
    return {
        "version_id": version,
        "document_id": "d1",
        "filename": "synthetic.txt",
        "sha256": "a" * 64,
        "ordinal": ordinal,
        "content": content,
        "location": location,
        "warnings": warnings or [],
        "job_warnings": [],
        "chunks": chunks
        if chunks is not None
        else [
            {
                "id": uuid4(),
                "part": 0,
                "char_start": 0,
                "char_end": len(content),
                "content": content,
                "index_version": "native-lexical-v1",
            }
        ],
    }


def test_ordinary_answer_is_not_an_exhaustive_result():
    result = validate_answer(
        json.dumps(
            {
                "insufficient": False,
                "answer": "The note records rainfall. [S1]",
            }
        ),
        [hit("The note records rainfall.")],
    )
    assert "exhaustive" in result["scope_notice"]
    assert result.get("coverage_complete") is not True


def test_unreadable_page_is_a_gap_even_when_text_chunks_exist():
    chunk = {
        "id": uuid4(),
        "part": 0,
        "char_start": 0,
        "char_end": 5,
        "content": "ghost",
        "index_version": "native-lexical-v1",
    }
    sections, notes = build_sections(
        [
            page(
                "",
                chunks=[chunk],
                warnings=["no_native_text_ocr_or_visual_review_required"],
                kind="pdf_page",
            )
        ],
        "workspace",
    )
    assert len(sections) == 1
    assert sections[0]["gap_code"] == "unreadable_section"
    assert sections[0]["content"] == ""
    assert not verdict(sections, reduction_complete=True)
    assert "OCR" in " ".join(notes)


def test_arabic_and_mixed_text_are_not_normalized_before_the_model():
    arabic = "إِجمالي المبيعات ١٢٠ جنيه"
    mixed = "Policy 12: لا يجوز تجاوز الحد."
    sections, _ = build_sections(
        [page(arabic, ordinal=1), page(mixed, ordinal=2, version="v2")],
        "workspace",
    )
    request = make_coverage_request(
        assemble("coverage"),
        "لخص الوثيقة",
        [
            {"alias": "S1", "text": sections[0]["content"]},
            {"alias": "S2", "text": sections[1]["content"]},
        ],
        kind="section",
        max_output_tokens=256,
    )
    blob = "\n".join(request.evidence)
    assert "إِجمالي" in blob
    assert "١٢٠" in blob
    assert "لا يجوز" in blob
    assert "EVD-004" in request.instructions.rule_ids
    assert "Do not claim complete coverage" in request.instructions.text
    assert request.images == () and request.tools == ()
    request.validate()


def test_oversized_text_is_split_without_dropping_characters():
    instructions = assemble("coverage")
    question = "Summarize the selected documents completely."
    original = "م" * 512
    sections, _ = build_sections(
        [page("Short rainfall note.", ordinal=1), page(original, ordinal=2)],
        "workspace",
    )

    def fits(content):
        return fits_request(
            instructions,
            question,
            [{"alias": "S1", "text": content}],
            kind="section",
            output_tokens=256,
            budget=4096,
        )

    expanded = expand_sections(sections, fits)
    pieces = [section["content"] for section in expanded if section["ordinal"] == 2]
    assert "".join(pieces) == original
    assert len(pieces) > 1
    assert all(fits(piece) for piece in pieces)
    assert all(section["status"] == "pending" for section in expanded)
    blocked = expand_sections(
        build_sections([page(original)], "workspace")[0], lambda _content: False
    )
    assert blocked[0]["gap_code"] == "section_exceeds_budget"
    assert blocked[0]["content"] == original
    assert not verdict(blocked, reduction_complete=True)
    notice = scope_notice(blocked, complete=False, notes=[], versions=["v1"])
    assert "not complete" in notice
    assert "section_exceeds_budget" in notice


def test_batch_limit_is_reported_instead_of_a_silent_partial_review():
    instructions = assemble("coverage")
    question = "Summarize"
    content = None
    for count in range(1, 400):
        trial = "rainfall " * count
        pair, _ = build_sections(
            [page(trial, ordinal=1), page(trial, ordinal=2)], "workspace"
        )
        one = fits_request(
            instructions,
            question,
            [{"alias": "S1", "text": pair[0]["content"]}],
            kind="section",
            output_tokens=256,
            budget=4096,
        )
        two = fits_request(
            instructions,
            question,
            [
                {"alias": "S1", "text": pair[0]["content"]},
                {"alias": "S2", "text": pair[1]["content"]},
            ],
            kind="section",
            output_tokens=256,
            budget=4096,
        )
        if one and not two:
            content = trial
            break
    assert content, "expected a section that fits alone but not beside another"
    sections, _ = build_sections(
        [page(content, ordinal=ordinal) for ordinal in range(1, 5)], "workspace"
    )
    groups = plan_map(
        sections,
        instructions=instructions,
        question=question,
        output_tokens=256,
        budget=4096,
        max_calls=1,
    )
    assert len(groups) == 1
    assert gap_counts(sections)["batch_limit"] >= 1
    assert not verdict(sections, reduction_complete=True)


def test_uncited_sections_are_not_treated_as_covered():
    sections, _ = build_sections(
        [page("Alpha rainfall."), page("Beta runoff.", ordinal=2)], "ws"
    )
    sections[0]["global_alias"] = "S1"
    sections[1]["global_alias"] = "S2"
    cited, missed = split_cited(sections, ["S1"])
    assert [section["global_alias"] for section in cited] == ["S1"]
    assert [section["global_alias"] for section in missed] == ["S2"]


def test_complete_requires_every_section_and_a_finished_reduction():
    sections, notes = build_sections(
        [page("Rainfall is 12 mm.", kind="pdf_page")], "ws"
    )
    for section in sections:
        section["status"] = "covered"
    assert verdict(sections, reduction_complete=True)
    assert not verdict(sections, reduction_complete=False)
    notice = scope_notice(
        sections, complete=True, notes=notes, versions=["v"], extra_gaps=()
    )
    assert notice.startswith("Complete coverage of 1 extracted sections")
    assert "not complete" not in notice
    assert "OCR" in notice


@pytest.mark.asyncio
async def test_fixture_cites_every_coverage_alias_and_ignores_marker_in_evidence():
    request = make_coverage_request(
        assemble("coverage"),
        "لخص",
        [
            {"alias": "S1", "text": "إِجمالي المبيعات ١٢٠"},
            {"alias": "S2", "text": "Quarter 2 sales 80"},
        ],
        kind="section",
        max_output_tokens=256,
    )
    events = [event async for event in FixtureQuoteGateway().stream(request)]
    body = json.loads(events[0].text)
    assert "[S1]" in body["answer"] and "[S2]" in body["answer"]
    assert events[1].finish_reason == "stop"
    ordinary = make_coverage_request(
        assemble("retrieval"),
        "sales",
        [{"alias": "S1", "text": "COVERAGE_WORKFLOW Quarter 1 sales: 120"}],
        kind="section",
        max_output_tokens=256,
    )
    # Retrieval instructions do not contain the coverage marker, so the fixture
    # keeps the legacy exact-quote contract even if evidence contains the word.
    ordinary = type(ordinary)(
        assemble("retrieval"),
        ordinary.question,
        ordinary.evidence,
        max_output_tokens=256,
    )
    quoted = [event async for event in FixtureQuoteGateway().stream(ordinary)]
    assert "claims" in quoted[0].text


def test_section_citations_remap_to_global_aliases_and_reject_unknowns():
    first, second = page("Alpha rainfall."), page("Beta runoff.", ordinal=2)
    sections, _ = build_sections([first, second], "ws")
    sections[0]["global_alias"] = "S4"
    sections[1]["global_alias"] = "S5"
    text, aliases = parse_section(
        json.dumps({"insufficient": False, "answer": "Both notes matter. [S2] [S1]"}),
        sections,
    )
    assert aliases == ["S5", "S4"]
    assert "[S5]" in text and "[S4]" in text
    with pytest.raises(GatewayError):
        parse_section(
            json.dumps({"insufficient": False, "answer": "Invented [S9]"}),
            sections,
        )
    with pytest.raises(GatewayError):
        parse_section(
            json.dumps({"insufficient": False, "answer": "Page claim (Page 4). [S1]"}),
            sections,
        )


def test_released_arabic_answer_keeps_wording_and_original_passage():
    source = hit("إِجمالي المبيعات ١٢٠ جنيه")
    source["global_alias"] = "S1"
    source["location"] = {"kind": "text", "line_start": 1, "line_end": 1, "part": 0}
    result = release_answer("يلخص القسم المبيعات. [S1]", [source])
    assert result["answer"].startswith("يلخص القسم المبيعات.")
    assert "١٢٠" in result["claims"][0]["quote"]
    assert "إِجمالي" in result["claims"][0]["quote"]
    assert result["claims"][0]["source_id"] == source["source_id"]
