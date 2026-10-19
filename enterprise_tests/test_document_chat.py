"""Natural answers still require checked citations and honest overview intent."""

import json

import pytest

from backend.answers.grounding import overview_question, validate_answer
from backend.gateway.contracts import GatewayError
from enterprise_tests.test_quote_answers import hit


@pytest.mark.parametrize(
    "question",
    [
        "whats the paper about?",
        "What's this paper about?",
        "What is this document about?",
        "Summarize this report",
        "Give me an overview",
        "لخص البحث",
        "الورقة بتتكلم عن إيه؟",
    ],
)
def test_overview_does_not_require_a_document_keyword_match(question):
    assert overview_question(question)


def test_generated_explanation_is_returned_with_support_not_replaced_by_quote():
    source = hit("This study examines rainfall forecasting.")
    raw = json.dumps(
        {
            "insufficient": False,
            "claims": [
                {
                    "source_id": source["source_id"],
                    "quote": "rainfall forecasting",
                    "explanation": "The paper investigates predicting rainfall.",
                }
            ],
        }
    )
    result = validate_answer(raw, [source])
    assert result["answer"] == "The paper investigates predicting rainfall."
    assert result["claims"][0]["quote"] == "rainfall forecasting"
    assert "model interpretations" in result["limitation"]


def test_explanation_cannot_bypass_quote_or_identity_checks():
    source = hit("Original evidence.")
    for quote in ("invented evidence", " "):
        raw = json.dumps(
            {
                "insufficient": False,
                "claims": [
                    {
                        "source_id": source["source_id"],
                        "quote": quote,
                        "explanation": "A fluent answer is not enough.",
                    }
                ],
            }
        )
        with pytest.raises(GatewayError):
            validate_answer(raw, [source])


def test_passage_citation_needs_no_brittle_pdf_quote_reproduction():
    source = hit("A PDF with hyphen-\nated words and  irregular spacing.")
    raw = json.dumps(
        {
            "insufficient": False,
            "claims": [
                {
                    "source_id": "S1",
                    "explanation": "The cited passage discusses hyphenated words.",
                }
            ],
        }
    )
    result = validate_answer(raw, [source])
    claim = result["claims"][0]
    assert claim["source_id"] == source["source_id"]
    assert claim["quote"] == source["content"]
    assert claim["support_kind"] == "source_passage"
    assert result["answer"].startswith("The cited passage")


def test_unknown_short_citations_still_fail_closed():
    with pytest.raises(GatewayError):
        validate_answer(
            json.dumps(
                {
                    "insufficient": False,
                    "claims": [{"source_id": "S999", "explanation": "Not authorized."}],
                }
            ),
            [hit()],
        )
