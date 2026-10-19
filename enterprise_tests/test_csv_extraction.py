"""Actual bounded child-parser checks; values stay inert strings."""

import json

import pytest

from backend.ingestion.extract import extract


def test_csv_preserves_arabic_identifiers_empty_cells_and_physical_lines():
    body = '\ufeffid,label,value\r\n001,"مبيعات, مصر",\r\n002,"two\nlines",-12.50\r\n'
    result = extract(body.encode(), "fixture.CSV")
    assert result["status"] == "extracted"
    assert json.loads(result["pages"][1]["content"]) == ["001", "مبيعات, مصر", ""]
    assert json.loads(result["pages"][2]["content"]) == ["002", "two\nlines", "-12.50"]
    assert result["pages"][2]["location"] == {
        "kind": "csv_row",
        "row": 3,
        "line_start": 3,
        "line_end": 4,
        "column_count": 3,
    }


def test_csv_formula_like_values_remain_literal_text():
    result = extract(b'id,value\n001,"=HYPERLINK(""https://example.com"")"\n', "f.csv")
    assert result["status"] == "extracted"
    assert json.loads(result["pages"][1]["content"])[1].startswith("=HYPERLINK")
    assert "csv_formula_like_text_not_executed" in result["pages"][1]["warnings"]


def test_ragged_csv_requires_review_and_retains_all_cells():
    result = extract(b"a,b\n1,2,3\n", "ragged.csv")
    assert result["status"] == "needs_review"
    assert json.loads(result["pages"][1]["content"]) == ["1", "2", "3"]
    assert "csv_inconsistent_column_count" in result["warnings"]


@pytest.mark.parametrize(
    "body,code",
    [
        (b'a,"unterminated', "invalid_csv"),
        (b"a,\x00", "text_limit_or_binary"),
        (b"\xff", "parser_resource_or_encoding_limit"),
        (b"\n\n", "invalid_csv"),
        (b"a\n" * 1001, "csv_structure_limit"),
        (b",".join([b"a"] * 101), "csv_structure_limit"),
        (b"x" * 1_000_001, "extracted_text_limit"),
    ],
)
def test_csv_rejections_publish_no_partial_rows(body, code):
    assert extract(body, "invalid.csv") == {"error_code": code}


@pytest.mark.parametrize(
    "name", ["macros.xlsm", "legacy.xls", "slides.pptx", "word.docx"]
)
def test_unimplemented_formats_fail_closed(name):
    assert extract(b"untrusted bytes", name) == {"error_code": "unsupported_format"}
