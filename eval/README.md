# Evaluation corpus

Frozen papers and golden queries for the RAGBase study. Do not edit these files after the baseline run.

| File | Pages | What it covers |
|---|---|---|
| `english_hypercore_ops.pdf` | 4 | English operations guide |
| `arabic_mixed_notes.pdf` | 3 | Arabic notes with English terms |
| `limits_table.pdf` | 3 | Limits and a cluster table |

`golden.jsonl` has 100 queries. Each line has `query`, `doc`, `expected_pages`, `expected_facts`, `type`, and `lang`.

These are synthetic documents written for evaluation. They are not vendor manuals.
