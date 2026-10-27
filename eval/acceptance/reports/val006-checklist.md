# VAL-006 recovery and offline checklist

Decision: **pending**. This file does not qualify offline operation or backup recovery.

| Item | Mode | Status | Check |
|---|---|---|---|
| `local-boot-offline` | local | pending | Boot a prepared local-mode release without public internet. |
| `local-enabled-formats` | local | pending | Exercise each enabled document format on the prepared release. |
| `local-arabic-ocr` | local | pending | Run Arabic OCR on the prepared release. OCR is not implemented yet. |
| `local-evidence-question` | local | pending | Ask an evidence-backed question and keep the citation. |
| `local-workbook-calculation` | local | pending | Perform a workbook calculation through the trusted calculator. |
| `local-open-citation` | local | pending | Open a citation from the answer. |
| `local-export` | local | pending | Export a result table. |
| `local-restart` | local | pending | Restart services and confirm authorized data is still available. |
| `local-restore` | local | pending | Restore a backup onto a separate check, including database and files. |
| `local-network-denial` | local | pending | Record network-denial evidence for local mode. |
| `local-no-public-dependency` | local | pending | Show that local mode did not require a public connection. |
| `remote-relay-only` | remote | pending | Show that only the generation relay reaches the approved provider. |
| `remote-missing-key` | remote | pending | Missing or invalid provider keys fail safely. |
| `remote-upstream-outage` | remote | pending | An upstream outage fails safely. |
| `remote-local-services` | remote | pending | Local document services stay usable when generation is down. |
| `remote-cancellation` | remote | pending | Cancellation closes local transports. Remote billing may already have started. |
| `remote-not-offline` | remote | pending | Hosted inference is not described as offline-qualified. |
