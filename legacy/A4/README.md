# A4 Snapshot

This folder preserves the original ATIS-based Text2SQL assignment material.

## Layout

- `data/`: raw ATIS source files.
- `build_dataset.py`: turns the raw ATIS JSON into classification and generation JSONL files.
- `Classification/`: template classification experiments and checkpoints.
- `Generation/`: seq2seq generation experiments and checkpoints.
- `LLM/`: prompt-based OpenRouter experiments.
- `processed_data/`: precomputed splits and template maps.
- `A4/`: a nested copy of the original workspace, preserved as-is for fidelity.

## How the original pipeline works

1. `build_dataset.py` reads `data/atis.json`.
2. It emits question-split and query-split JSONL files under `processed_data/`.
3. Classification models predict a `template_id`, then fill variables back into the template SQL.
4. Generation models train seq2seq networks to map question text directly to SQL.
5. LLM scripts build prompts from training examples and call OpenRouter for zero-shot / few-shot evaluation.

## Common entrypoints

From the repo root:

```bash
source .venv/bin/activate
cd legacy/A4
python build_dataset.py
python A4/build_dataset.py
python A4/Classification/a4_experiment.py
python A4/Generation/LSTM/LSTM_train_question_split.py
python A4/Generation/Transformer/train_transformer_question_split.py
cd LLM
python run_model.py --train-file generation_train.jsonl --test-file generation_test.jsonl
```

For the OpenRouter scripts, the repo root `.env` file is loaded automatically. If you prefer shell env vars, you can still export `OPENROUTER_API_KEY` manually:

```bash
export OPENROUTER_API_KEY="your-key-here"
```

Then run the prompt-based helpers from `legacy/A4/LLM/`.

## Notes

- The outer `legacy/A4/` tree and the nested `legacy/A4/A4/` tree both exist because the source workspace had both variants.
- Historical checkpoints and evaluation artifacts are preserved to make the snapshot self-contained.
