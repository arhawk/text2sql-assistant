# A2 Snapshot

This folder preserves the original COMP5046 A2 Q6 solution code.

## Files

- `importData.py`: downloads the original `a2_data.json` file.
- `q6_combined.py`: contains the full perceptron-style model, inference, metrics, and `main(...)` runner.
- `data.q6_combined.sample.json`: sample input used by the assignment.
- `a2_data.json`: the downloaded assignment dataset.

## What the code does

- `read_data(...)` parses the assignment JSON into `train/dev/test` splits and a label set.
- `CodeModel` stores token/SQL weights for a structured perceptron.
- `find_best_code(...)` scores every candidate SQL label and returns the highest-scoring one.
- `learn(...)` performs perceptron updates until the prediction matches the gold SQL.
- `get_confusion_matrix(...)` and the metric helpers compute accuracy, precision, recall, and macro F1.
- `main(...)` trains for a fixed number of iterations and evaluates on dev/test.

## Run

```bash
source .venv/bin/activate
cd legacy/A2
python importData.py
python q6_combined.py
```

## Notes

- The style is intentionally close to the original submission, so the file reads more like an assignment solution than a packaged library.
- If you only want the logic, start with [`q6_combined.py`](./q6_combined.py).
