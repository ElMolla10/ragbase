"""Generated PDF fixtures cover reading order and table cell relationships."""

import fitz

from backend.ingest import extract_text_from_pdf


def test_two_columns_read_left_before_right():
    with fitz.open() as doc:
        page = doc.new_page(width=600, height=800)
        # Deliberately write objects in the wrong order.
        page.insert_text((330, 100), "RIGHT first\nRIGHT second")
        page.insert_text((40, 100), "LEFT first\nLEFT second")
        text = extract_text_from_pdf(doc.tobytes())[0].text
    assert text.index("LEFT second") < text.index("RIGHT first")


def test_ruled_table_preserves_rows():
    with fitz.open() as doc:
        page = doc.new_page()
        for x in (50, 200, 350):
            page.draw_line((x, 100), (x, 190))
        for y in (100, 130, 160, 190):
            page.draw_line((50, y), (350, y))
        for y, cells in zip((120, 150, 180), (("Model", "Accuracy"), ("E5", "92%"), ("MiniLM", "81%")), strict=True):
            for x, cell in zip((60, 210), cells, strict=True):
                page.insert_text((x, y), cell)
        text = extract_text_from_pdf(doc.tobytes())[0].text
    assert "Model | Accuracy" in text
    assert "E5 | 92%" in text
    assert text.count("MiniLM") == 1


def test_table_failure_keeps_text(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("table detector unavailable")

    monkeypatch.setattr(fitz.Page, "find_tables", fail)
    with fitz.open() as doc:
        doc.new_page().insert_text((72, 72), "Text must survive")
        assert "Text must survive" in extract_text_from_pdf(doc.tobytes())[0].text
