"""Conservative reading order and ruled-table extraction using PyMuPDF."""

import logging

import fitz

logger = logging.getLogger(__name__)


def page_text(page: fitz.Page) -> str:
    """Read common two-column layouts by column, preserving full-width blocks.

    Table detection is best effort: a failure falls back to ordinary text.
    Blocks that straddle table boundaries are retained to avoid losing prose.
    """
    blocks = [
        (fitz.Rect(b[:4]), b[4].strip())
        for b in page.get_text("blocks")
        if b[6] == 0 and b[4].strip()
    ]
    tables = []
    try:
        for table in page.find_tables().tables:
            rows = table.extract()
            text = "\n".join(
                " | ".join(" ".join((cell or "").split()) for cell in row)
                for row in rows
            )
            if text.strip():
                tables.append((fitz.Rect(table.bbox), text))
    except Exception:
        logger.warning("Table detection failed on page %s; using text", page.number + 1)

    blocks = [
        (rect, text)
        for rect, text in blocks
        if not any(table_rect.contains(rect) for table_rect, _ in tables)
    ] + tables
    midpoint = page.rect.width / 2
    left = [b for b in blocks if b[0].x1 <= midpoint]
    right = [b for b in blocks if b[0].x0 >= midpoint]
    # Only infer columns when both sides have vertically overlapping content.
    two_columns = (
        left
        and right
        and any(
            lb[0].y0 < rb[0].y1 and rb[0].y0 < lb[0].y1 for lb in left for rb in right
        )
    )
    if not two_columns:
        ordered = sorted(blocks, key=lambda b: (b[0].y0, b[0].x0))
    else:
        spanning = sorted(
            [b for b in blocks if b[0].x0 < midpoint < b[0].x1],
            key=lambda b: b[0].y0,
        )
        ordered = []
        remaining = left + right
        for separator in spanning:
            band = [b for b in remaining if b[0].y0 < separator[0].y0]
            ordered.extend(sorted(band, key=lambda b: (b[0].x0 >= midpoint, b[0].y0)))
            remaining = [b for b in remaining if b[0].y0 >= separator[0].y0]
            ordered.append(separator)
        ordered.extend(sorted(remaining, key=lambda b: (b[0].x0 >= midpoint, b[0].y0)))
    return "\n\n".join(text for _, text in ordered)
