# Text2SQL Assistant

This repo turns the COMP5046 `A2 + A4` course work into one demo-oriented Text2SQL assistant.

## What it is

- `A2` becomes the baseline retriever.
- `A4` becomes the main system with three adapters:
  - classification-style template matching
  - generation-style SQL filling
  - LLM-style ensemble fallback
- A single CLI demo shows predictions, exact match scores, and saves artifacts.

## Layout

```text
text2sql-assistant/
  app/                # optional future UI layer
  data/
    sample_text2sql.jsonl
    raw/              # bring your own course data here
    processed/
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

## Quick start

Run an interactive demo:

```bash
python scripts/demo.py
```

Run the Streamlit UI:

```bash
streamlit run app/streamlit_app.py
```

If you want an isolated environment first:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
streamlit run app/streamlit_app.py
```

Run a single query:

```bash
python scripts/demo.py --question "show flights from boston to denver" --mode all
```

Evaluate the sample dataset:

```bash
python scripts/evaluate.py --dataset data/sample_text2sql.jsonl
```

## How to plug in the real course data

The loader accepts JSONL with these fields:

- `question`
- `sql`
- `split` (`train`, `dev`, `test`)
- `source` (optional)
- `template_sql` (optional, helpful for classification/generation)

If your A4 data already has different field names, add a small adapter in `scripts/preprocess.py` and emit the standard JSONL shape.

## Output

All demo and evaluation outputs go to `artifacts/runs/<timestamp>/`.

- `predictions.jsonl`
- `summary.json`
- `errors.jsonl`
- `stdout.log`

## Notes

- The core pipeline is standard-library only; the Streamlit UI adds one lightweight frontend dependency.
- The sample dataset is intentionally small and synthetic so the demo works immediately.
- Replace the sample data with the real A2/A4 data when you are ready.
