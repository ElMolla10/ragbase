"""Run the parser with a wall deadline and bounded output, outside the API."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ERROR_CODES = {
    "text_limit_or_binary",
    "encrypted_pdf_unsupported",
    "pdf_page_limit",
    "pdf_content_stream_limit",
    "extracted_text_limit",
    "upload_byte_limit",
    "parser_resource_or_encoding_limit",
    "invalid_or_unsupported_pdf",
    "extraction_output_limit",
    "parser_timeout",
    "parser_failed",
    "artifact_unavailable",
}


def extract(data: bytes, filename: str):
    kind = "pdf" if filename.lower().endswith(".pdf") else "text"
    with tempfile.TemporaryFile() as output:
        try:
            process = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    str(Path(__file__).with_name("parser_process.py")),
                    kind,
                ],
                input=data,
                stdout=output,
                stderr=subprocess.DEVNULL,
                env={"LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"},
                timeout=30,
                check=False,
            )
            if process.returncode != 0 or output.tell() > 8_000_000:
                return {"error_code": "parser_failed"}
            output.seek(0)
            result = json.loads(output.read(8_000_001))
            if "error_code" in result and result["error_code"] not in ERROR_CODES:
                return {"error_code": "parser_failed"}
            return result
        except subprocess.TimeoutExpired:
            return {"error_code": "parser_timeout"}
        except (OSError, ValueError):
            return {"error_code": "parser_failed"}
