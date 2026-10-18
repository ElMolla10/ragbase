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

The current foundation entry point is `backend.enterprise:app`. It exposes only
development health/status and has no document or analysis API yet. Do not add
unprotected data routes to this scaffold.
