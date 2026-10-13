# Evaluation corpus

Frozen papers and golden queries for the RAGBase study. Do not edit these files after the baseline run.

| File | Pages | What it covers |
|---|---|---|
| `english_hypercore_ops.pdf` | 4 | English operations guide |
| `arabic_mixed_notes.pdf` | 3 | Arabic notes with English terms |
| `limits_table.pdf` | 3 | Limits and a cluster table |

`golden.jsonl` has 100 queries. Each line has `query`, `doc`, `expected_pages`, `expected_facts`, `type`, and `lang`.

These are synthetic documents written for evaluation. They are not vendor manuals.

## Baseline

Retrieval only:

```bash
python -m scripts.eval
```

Retrieval plus citation and refusal scoring:

```bash
python -m scripts.eval --generate
```

Results land in `eval/results/baseline.json`.

Metrics:

- `recall_at_5`: expected page appears in the top 5 chunks
- `citation_accuracy`: generated answer cites a correct page
- `refusal_rate`: negative queries are refused

CI checks retrieval only and fails if recall@5 drops below 0.35.
