# Text2SQL Assistant

This repository now has two layers:

- a small, runnable Text2SQL prototype in `src/text2sql_assistant/`
- preserved course snapshots in `legacy/A2/` and `legacy/A4/`

## What to read first

1. [`legacy/README.md`](legacy/README.md) for the overall archive layout.
2. [`legacy/A2/README.md`](legacy/A2/README.md) for the A2 perceptron baseline.
3. [`legacy/A4/README.md`](legacy/A4/README.md) for the ATIS pipeline.

## Current demo layer

The active demo is the simplified Text2SQL assistant used in this repo:

- `baseline`: nearest-neighbor SQL reuse
- `classification`: template lookup plus slot filling
- `generation`: retrieval plus template completion
- `llm`: lightweight ensemble between baseline and classification

### Run it

```bash
source .venv/bin/activate
python scripts/demo.py
python scripts/demo.py --question "show flights from boston to denver" --mode all
streamlit run app/streamlit_app.py
```

### Data shape

The demo loader accepts JSON or JSONL rows with:

- `question`
- `sql`
- `split` (`train`, `dev`, `test`)
- `source` (optional)
- `template_sql` (optional)

### Outputs

Artifacts are written to `artifacts/runs/<timestamp>/` and include:

- `predictions.jsonl`
- `summary.json`
- `errors.jsonl`

## Repository layout

```text
text2sql-assistant/
  app/
    streamlit_app.py
  data/
    sample_text2sql.jsonl
  legacy/
    A2/
    A4/
  scripts/
    demo.py
    preprocess.py
    evaluate.py
  src/text2sql_assistant/
    common.py
    data.py
    baseline_a2.py
    a4_models.py
    pipeline.py
    types.py
  artifacts/
    runs/
```

## Notes

- The current prototype stays intentionally lightweight and easy to run.
- The legacy trees keep the original assignment code, data, checkpoints, and processed splits visible for reference.
- The OpenRouter-based legacy scripts auto-load `OPENROUTER_API_KEY` from the repo root `.env` file.
