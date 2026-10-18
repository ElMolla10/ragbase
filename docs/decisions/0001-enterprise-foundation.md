# 0001 — Isolated foundation and pending model validation

> Historical foundation record. The CPU-only execution plan v2.0 and constitution v2.0.0 supersede earlier hardware, model, context and startup assumptions. Existing source WIP must be reviewed and tested before these older startup instructions are reused.

Date: 18 October 2026. Status: implemented foundation decision.

The reviewed upstream and implementation base are both
`2f0a0204a14f6659b97a061168ccb0bdd8c061f7`. Work lives on local
`enterprise-local`; no remote branch or public deployment has been changed.

Use `backend.enterprise:app` as the new foundation instead of weakening the
existing demo's settings or startup path. It currently exposes only health and
readiness. Data routes wait for scoped identity, durable storage, and migrations.
Existing demo routes and startup remain legacy and do not satisfy the constitution.

Governance manifests use JSON rather than YAML so validation and policy assembly
can use the Python standard library. This changes file extensions in the plan,
not their responsibilities. The constitution remains the sole policy source;
reviewed templates are derived assets with hashes and rule references.

Use LF for constitution/policy assets to preserve hashes across Windows and
Linux. Detect stale assets rather than refresh hashes automatically at startup.

Prepare three model manifests, all production-disabled. The HTTP adapter is a
text-only contract scaffold; actual tokenizer budgeting, capabilities, nonstreaming,
credentials, structured tools, vision, and real model swapping remain future work.
Never interpret fixtures or manifests as a model benchmark.

Keep an isolated Compose project on loopback port 8800, separate from the
already running user's RAGBase services on ports 8000/8501/5432. Do not use
existing services, credentials, databases, or model caches for these changes.

Direct foundation dependencies are pinned to versions observed in the local
test environment. A complete transitive lock, image digests, offline bundle, and
dependency/license review remain packaging gates before production.
