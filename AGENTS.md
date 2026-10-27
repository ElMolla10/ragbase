# Enterprise development instructions

Read `CONSTITUTION.md` before making changes. It is the authoritative project
policy; do not duplicate it in model-specific instruction files. Read
`docs/execution-plan.md` for delivery order and `docs/governance/rule-ledger.json`
for implementation/verification state.

- This checkout is the enterprise implementation; preserve public-demo deployment
  boundaries. The legacy `backend.main:app` is not an enterprise deployment.
- Keep company documents and sensitive evaluations out of this repository.
- Do not download model weights during ordinary development or tests.
- Keep server-dependent evaluations pending until actual compute is available.
- Run relevant tests and lint; record what was measured and what remains pending.
- Use one canonical constitution; derive bounded runtime instructions and enforce
  authorization, data scope, execution, and network limits in trusted code.

The current development entry point is `backend.enterprise:app`. With explicit
enterprise configuration it exposes authenticated PDF/TXT/Markdown/CSV originals,
durable native extraction jobs, page evidence and downloads backed by PostgreSQL
RLS. A separate bounded worker uses explicit development actor scopes. Versioned
lexical retrieval, bounded overview context and natural answers with checked supporting quotations/citations are implemented. Do not reinstate keyword-only failure or quote-only output. Ordinary overviews stay bounded and must not be labeled complete. Explicit exhaustive coverage is a separate authenticated workflow over every required section of the selected authorized versions; it records gaps and uses a complete label only when coverage finished. Default
generation is disabled; the explicitly opted-in OpenRouter relay is development-only. Read docs/openrouter-byok.md for the current hosted-generation implementation and pending free-access validation; never expose credentials or silently change modes. Company
Phase 4 is in progress: bounded CSV ingestion is implemented; OCR and Office
formats remain pending. Authorized CSV calculations are a separate development
slice: one ready CSV version, an allowlisted operation plan, and DuckDB on a
job-local snapshot with external access and extension autoload disabled and
settings locked. The model may propose the plan through the existing Groq-then-Gemini
OpenRouter client; trusted code calculates the number. Free SQL and arbitrary
Python stay disabled, and generated analytical SQL is never sent to PostgreSQL.
`/calculate` on the answer stream and `POST /api/workspaces/{id}/calculations`
are authenticated and creator-scoped. Margin, mixed currency, and forecast/cause
requests ask for clarification. DuckDB `memory_limit` is not an OS cgroup.
The development graphical UI is served by this app with an Arabic/English toggle.
It can start `/calculate` for one ready CSV and displays the server record,
result table, clarification, and authenticated CSV/JSON downloads. It does not
calculate. SSO, dense retrieval, charts, joins, XLSX formulas, Arabic
qualification and production acceptance remain pending. Never
add unprotected routes or label local smoke results production qualification.
