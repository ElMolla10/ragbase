"""Standalone parser child. No database, credentials, models or remote inputs.

Invoked by absolute path with Python isolated mode. Linux resource limits are
mandatory for execution; Windows development uses the Linux test/worker image.
"""

import io
import json
import logging
import re
import sys


class RejectedError(ValueError):
    pass


def parse(data, kind):
    if kind == "text":
        content = data.decode("utf-8")
        if len(content) > 1_000_000 or re.search(
            r"[\x00-\x08\x0b\x0c\x0e-\x1f]", content
        ):
            raise RejectedError("text_limit_or_binary")
        return {
            "extractor": "utf8-v1",
            "status": "extracted",
            "warnings": [],
            "pages": [
                {
                    "ordinal": 1,
                    "content": content,
                    "warnings": [],
                    "location": {
                        "kind": "text",
                        "line_start": 1,
                        "line_end": len(content.splitlines()),
                    },
                }
            ],
        }
    from pypdf import PdfReader, __version__

    reader = PdfReader(io.BytesIO(data), strict=True)
    if reader.is_encrypted:
        raise RejectedError("encrypted_pdf_unsupported")
    if not 1 <= len(reader.pages) <= 100:
        raise RejectedError("pdf_page_limit")
    pages, total, needs_review = [], 0, False
    for ordinal, page in enumerate(reader.pages, 1):
        contents = page.get_contents()
        if contents is not None and len(contents.get_data()) > 8_000_000:
            raise RejectedError("pdf_content_stream_limit")
        content = page.extract_text() or ""
        if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", content):
            raise RejectedError("text_limit_or_binary")
        total += len(content)
        if len(content) > 100_000 or total > 1_000_000:
            raise RejectedError("extracted_text_limit")
        warnings = []
        if not content.strip():
            warnings.append("no_native_text_ocr_or_visual_review_required")
            needs_review = True
        pages.append(
            {
                "ordinal": ordinal,
                "content": content,
                "warnings": warnings,
                "location": {"kind": "pdf_page", "page": ordinal},
            }
        )
    return {
        "extractor": "pypdf-" + __version__ + "-native-v1",
        "status": "needs_review" if needs_review else "extracted",
        "pages": pages,
        "warnings": ["native_text_only_layout_tables_images_and_ocr_not_validated"],
    }


def main():
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (384 * 1024**2, 384 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024**2, 8 * 1024**2))
    logging.disable(logging.CRITICAL)
    try:
        data = sys.stdin.buffer.read(5_000_001)
        if len(data) > 5_000_000:
            raise RejectedError("upload_byte_limit")
        result = parse(data, sys.argv[1])
    except RejectedError as exc:
        result = {"error_code": str(exc)}
    except (MemoryError, UnicodeError):
        result = {"error_code": "parser_resource_or_encoding_limit"}
    except Exception:
        result = {"error_code": "invalid_or_unsupported_pdf"}
    encoded = json.dumps(result, ensure_ascii=True).encode()
    if len(encoded) > 8_000_000:
        encoded = b'{"error_code":"extraction_output_limit"}'
    sys.stdout.buffer.write(encoded)


if __name__ == "__main__":
    main()
