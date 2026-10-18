"""In-memory synthetic PDF bytes; no company material or output artifacts."""

import io

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def pdf_bytes(texts=("Synthetic sales: 120",), *, encrypted=False, compressed=False):
    writer = PdfWriter()
    font = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )
    for content in texts:
        page = writer.add_blank_page(width=400, height=400)
        if content:
            page[NameObject("/Resources")] = DictionaryObject(
                {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
            )
            stream = DecodedStreamObject()
            escaped = (
                content.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            )
            stream.set_data(f"BT /F1 12 Tf 20 350 Td ({escaped}) Tj ET".encode("ascii"))
            page[NameObject("/Contents")] = writer._add_object(
                stream.flate_encode() if compressed else stream
            )
    if encrypted:
        writer.encrypt("synthetic-password")
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()
