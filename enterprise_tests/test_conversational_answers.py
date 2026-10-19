"""Public-style prose keeps trusted inline locations and valid source support."""

import json

import pytest

from backend.answers.grounding import validate_answer
from backend.gateway.contracts import GatewayError
from enterprise_tests.test_quote_answers import hit


def test_cohesive_answer_gets_inline_real_page_instead_of_claim_footer():
    source = hit("The paper describes an integrated storage architecture.")
    source["location"] = {"kind": "pdf_page", "page": 3}
    result = validate_answer(
        json.dumps(
            {
                "insufficient": False,
                "answer": "The paper explains an integrated approach to storage. [S1]",
            }
        ),
        [source],
    )
    assert (
        result["answer"]
        == "The paper explains an integrated approach to storage. (Page 3)"
    )
    assert result["claims"][0]["quote"] == source["content"]
    assert result["claims"][0]["source_id"] == source["source_id"]


@pytest.mark.parametrize(
    "answer",
    [
        "Invented citation [S999]",
        "Unsupported uncited answer.",
        "Correct alias [S1], but invented page (Page 999).",
    ],
)
def test_unknown_missing_or_model_invented_page_citations_are_rejected(answer):
    with pytest.raises(GatewayError):
        validate_answer(json.dumps({"insufficient": False, "answer": answer}), [hit()])


def test_honest_missing_information_can_be_explained_naturally():
    result = validate_answer(
        json.dumps(
            {
                "insufficient": True,
                "answer": "The supplied pages do not describe deployment costs.",
            }
        ),
        [hit()],
    )
    assert result["insufficient"] and result["claims"] == []
    assert "do not describe" in result["answer"]


def test_seven_legitimate_inline_citations_do_not_hit_old_six_claim_limit():
    sources = [hit(f"Source evidence {number}") for number in range(7)]
    for number, source in enumerate(sources, 1):
        source["location"] = {"kind": "pdf_page", "page": number}
    result = validate_answer(
        json.dumps(
            {
                "insufficient": False,
                "answer": " ".join(
                    f"Statement {number} [S{number}]." for number in range(1, 8)
                ),
            }
        ),
        sources,
    )
    assert len(result["claims"]) == 7 and "(Page 7)" in result["answer"]
