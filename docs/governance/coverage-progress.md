# Exhaustive coverage progress

Updated 27 October 2026. This is a local development check for constitution rule
EVD-004. It is not production qualification, an OCR result, or a live-model
measurement.

## What was added

Ordinary overview answers stay bounded. They still say they are not an exhaustive
review. Search still reports `exhaustive: false`.

Exhaustive coverage is a separate authenticated route:

- `POST /api/workspaces/{id}/coverage`
- `GET /api/workspaces/{id}/coverage/{run_id}`
- `POST /api/workspaces/{id}/coverage/{run_id}/cancel`

The caller selects one to eight authorized versions. The workflow loads every
extracted page of those versions, requires every indexed chunk, and records a
gap for a blank, unreadable, or unindexed page. A section that does not fit the
active byte budget is split into smaller pieces that do fit. Characters are not
dropped. A piece that still cannot fit, or work that exceeds the map/reduction
call caps, stays a named gap. The result is labeled complete only when every
required section was cited by its section summary and the reduction finished.

The streamed answer cites application-resolved page, row, or line locations and
attaches the original passage text. The durable record stores a question hash,
section status, gap code, and location. It does not store the question, passage
text, or summary prose. `GET` returns that map only.

Coverage yields the generation slot between batches when an interactive answer
is queued or running, or when an ingestion job is extracting. One user can have
one active coverage run. Two coverage runs can be active in total. Cancellation
and loss of membership do not release summary text. Interrupted runs are not
resumed.

OpenRouter routing is unchanged: GPT-OSS 120B via Groq, then Gemini 3.8 Flash
via Google AI Studio. Tests use the fixture gateway and do not call a provider
or download weights.

## Local evidence

On this checkout, after installing dependencies so the isolated parser could
import `pypdf`:

- `ruff check .` and `ruff format --check .` passed.
- `python scripts/build_policy.py` passed.
- `python -m pytest enterprise_tests -q`: 208 passed.
- `python -m pytest enterprise_integration_tests -q` against a synthetic
  PostgreSQL 16 database migrated to `0006_coverage_runs`: 74 passed.

The new tests include Arabic and mixed Arabic/English text, a long Arabic
document, multi-section PDFs, CSV rows with formula-like text left literal,
partial coverage, unreadable pages, denied access, cancellation, and yielding
to interactive work. No live Groq or Google call was made.

## Not claimed

- Layout, tables, images, OCR, or Office files were not reviewed. A PDF can be
  coverage-complete for extracted native text and still carry the native-text
  warning.
- The byte budget is not a tokenizer. Fixture/local mode uses 4,096 charges
  with 256 reserved output tokens. OpenRouter mode still uses 16,000 charges
  with 2,048 reserved output tokens.
- Arabic characters were preserved in evidence and quotes. Arabic answer
  quality was not measured.
- Summary prose is not stored, so a disconnected client can retrieve the map
  but not the prose.
- Policy manifest hashes for the constitution and core template were corrected
  from CRLF digests to the LF bytes stored in git. `CONSTITUTION.md` and the
  template text were not edited. `.gitattributes` already requires LF so those
  hashes survive Windows and Linux checkouts.
