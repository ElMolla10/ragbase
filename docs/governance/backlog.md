# Execution backlog â€” CPU-only revision

Updated: 18 October 2026. Canonical package definitions: `docs/execution-plan.md`, version 2.0. Constitution: 2.0.0. The previous A/B package sequence is superseded by P0â€“P6.

| Phase | State | Next evidence needed |
|---|---|---|
| P0 hardware/policy reconciliation | Laptop inventory complete; actual server discovery pending | Actual server CPU/ISA/OS/visible RAM, utilization, storage and internal requirements |
| P1 lightweight foundation/scoped storage | Local storage slice running; 103 checks passed; production gates pending | Company SSO/provisioning, broader authorization, backup/restore and production controls; see cpu-storage-progress.md |
| P2 CPU baseline/PDF slice | Durable native extraction running; partial P2 only | Versioned chunks, authorized retrieval, citations, fixture answers, measured local inference and bilingual gold cases |
| P3 CPU OCR/Office formats | Not implemented | Native-first adapters, bounded Arabic/English OCR, preserved structure/cells/locations |
| P4 deterministic analysis | Not implemented | Typed tools, data validation, sandbox, process RAM and spill quotas, numerical gold fixtures |
| P5 coverage/vision/model comparison | Not implemented | Whole-document coverage, 9B/35B/27B/4B plus existing Gemma comparison, qualified images, real swaps |
| P6 offline hardening/pilot | Pending preceding gates | SSO, offline cold start, scheduler/load envelope, safe assets, backups/restore and acceptance report |

Execution resumed on the project owner’s subsequent instruction. See cpu-storage-progress.md for measured implementation evidence. Historical foundation counts remain superseded.

Next execution continues with P2 retrieval and cited-answer integration; durable jobs and bounded native extraction are now running while company-server discovery remains pending. The development storage boundary and policy/profile checks now pass. No model weights have been downloaded. The already-installed `gemma4:12b` can be considered for later short laptop tests after resource checks; it has not been benchmarked.

Current P2 evidence: [durable native extraction progress](cpu-ingestion-progress.md). P0 server discovery and all later release gates remain open.
