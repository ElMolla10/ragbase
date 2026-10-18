# Execution backlog

Updated: 18 October 2026. Work package definitions live in `docs/execution-plan.md`.

| Package | State | Next evidence needed |
|---|---|---|
| A0 baseline/governance | Implemented for initial local checkout | Continue updating ledger and baseline decisions |
| A1 foundation | In progress | Storage/embedding/OCR injection, migrations, full gateway/capability/token-budget contracts, larger dev-lite stack |
| A2 identity/access/storage | Not started | Scoped identity, SSO path, durable storage, RLS, revocation and serving-role tests |
| A3 durable ingestion | Not started | Jobs/outbox/retries, originals, text PDF adapters and source previews |
| A4 retrieval/chat | Not started | ACL-aware search, cited streaming, real-versus-mocked metrics |
| A5 formats | Not started | Office/workbook adapters, format gold fixtures and source mappings |
| A6 analysis | Not started | Typed trusted calculations, sandbox, numerical gold cases and exports |
| A7 coverage/UX | Not started | Coverage workflows, bounded vision, history, cancellation and RTL UX |
| A8 evaluations/packaging | Not started | Real benchmark runners, scorecards, offline bundles, backup/restore |
| B0 inventory | Pending company access | Hardware inventory, representative internal corpus and IT decisions |
| B1 candidate benchmarks | Pending suitable compute | A/B/C Arabic, context, vision, memory, latency and concurrency |
| B2 real integration/swap | Pending preceding gates | Validated local components and two real model swaps |
| B3 hardening/pilot | Pending preceding gates | Offline cold start, load, restoration and acceptance report |

Next implementation sequence:

1. Finish A1 migrations and enterprise configuration/injection contracts without
   broadening the unprotected development API.
2. Implement A2 workspace identity, storage and actual serving-role RLS tests in
   a separate enterprise database namespace.
3. Add A3 durable ingestion only behind those controls.

No current health check constitutes G1 completion for the entire planned
application. No current contract test establishes G7/G8/G9 model acceptance.
