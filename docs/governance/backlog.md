# Execution backlog — CPU-only revision

Updated: 18 October 2026. Canonical plan: execution-plan.md v2.0; constitution: 2.0.0. Seven main phases, P0–P6.

| Phase | Current state | Remaining acceptance/work |
|---|---|---|
| P0 / Phase 1: Discovery | Plan/constitution and laptop inventory complete | Actual server inventory, company requirements and reviewed acceptance cases |
| P1 / Phase 2: Foundation | Local scoped storage implemented and tested | Production SSO/provisioning, full authorization and release controls |
| P2 / Phase 3: PDF Q&A | Local implementation complete; 156 tests and live CPU smoke pass | Actual-server baseline and reviewed company-case acceptance remain pending |
| P3 / Phase 4: OCR/Office | Pending | Arabic/English OCR and fidelity-preserving DOCX/PPTX/XLSX/CSV adapters |
| P4 / Phase 5: Analysis | Pending | Typed calculations, data validation, sandbox, lineage and exports |
| P5 / Phase 6: Coverage/model qualification | Coverage workflow and a development RTL browser shell exist locally. The shell displays server CSV calculation records | Arabic qualification, hybrid retrieval, optional vision, model swaps, and production acceptance |
| P6 / Phase 7: Hardening/pilot | Pending | Company SSO, offline bundle, network denial, load envelope, backup restoration and acceptance |

Current evidence: [Phase 3 completion record](phase3-completion.md). Earlier storage/native-extraction reports are historical checkpoints. Local completion does not qualify company-server capacity or production use. No model weights were downloaded. The installed Qwen3 1.7B was used only as a development integration model; Gemma is absent from the current inventory and all production candidates remain unqualified.

Next implementation phase is P3 / Phase 4. Keep outstanding P0/server and production gates open; do not invent their results or silently turn a local smoke into a release acceptance.
