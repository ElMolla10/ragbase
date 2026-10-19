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
lexical retrieval, bounded overview context and natural answers with checked supporting quotations/citations are implemented. Do not reinstate keyword-only failure or quote-only output. Default
generation is disabled; the explicitly opted-in OpenRouter relay is development-only. Read docs/openrouter-byok.md for the current hosted-generation implementation and pending free-access validation; never expose credentials or silently change modes. Company
Phase 4 is in progress: bounded CSV ingestion is implemented; OCR and Office
formats remain pending. SSO, dense retrieval, general analysis and the graphical UI remain pending. Never
add unprotected routes or label local smoke results production qualification.
