# Phase 4 / P3 progress

Updated 19 October 2026. Phase 4 is **in progress**, not complete.

## Delivered slice: bounded CSV ingestion

UTF-8 comma-separated CSV originals now use the existing authenticated upload,
immutable storage, durable job, isolated parser subprocess and scoped retrieval
path. A UTF-8 BOM is accepted. Limits: 5 MB upload, 1,000 CSV records, 100 columns,
100,000 characters per field/rendered record and 1,000,000 extracted characters;
existing parser CPU, address-space, wall-time and output limits still apply.
One CSV record can span physical lines. No new parser dependency or model weight
is needed.

Each source record stores a canonical JSON array of strings and retains CSV row,
physical line range and column count. Cell order, leading zeros, decimal/date
text, empty cells, Unicode, quoted commas and quoted multiline content remain
intact. There is no numeric/date/header inference or formula evaluation. Formula-
like strings are marked in record warnings and stay literal. This is extraction
and exact-quote lookup, not a calculation feature or spreadsheet export.

Malformed CSV fails without partial published rows. Inconsistent record widths
retain their cells but produce `needs_review`; existing retrieval checks block
such versions. Unsupported Office/macro suffixes remain rejected. This slice
does not claim schema inference, units interpretation or general table analysis.

## Test client

`python scripts/try_enterprise.py` prompts for a local file, uploads through the
authenticated API, waits for extraction/indexing and then accepts questions.
It displays validated quotations and precise source previews using the existing
query client. It supports an existing `--version`, optional `--file` and a single
`--question`. Files are bounded before upload. Tokens are read from ignored local
development configuration and never displayed. Uploaded originals/jobs persist
when the client exits or times out.

The current enterprise app has no graphical chat interface. The client is a
development convenience using synthetic Alice/workspace A, not company SSO or a
production authentication flow. The active Groq-first BYOK route sends questions
and retrieved evidence externally: only nonconfidential testing is approved.

## Evidence

- 181 unit/contract tests and 62 isolated PostgreSQL integration tests pass.
- Parser tests cover Arabic, quoted multiline fields, BOM, leading-zero IDs,
  empty cells, literal formulas, malformed input, encoding, row/column/text
  limits and unimplemented Office formats.
- Integration tests cover upload -> job -> extraction -> lexical search ->
  row-specific preview, byte-identical original download, cross-workspace denial,
  invalid-encoding rejection and failed/review-required versions blocked from Q&A.
- The running API/worker were rebuilt and recreated. The interactive client
  uploaded a synthetic CSV and returned `["001", "Quarter 1 sales", "120"]`,
  validated against CSV row 2 with the correct source preview. The answer stage
  took 1.00 seconds in that run; this is not a performance acceptance benchmark.

## Remaining Phase 4 work

Bounded native-first Arabic/English OCR, DOCX revisions/notes and table handling,
PPTX slide/notes lineage, XLSX values/formulas/caches/types/merged and hidden
content metadata, archive expansion safeguards, reviewed format fidelity cases,
and CPU/resource testing remain open. No OCR model or language pack was downloaded
and no target-server or company-document validation was performed by this slice.
