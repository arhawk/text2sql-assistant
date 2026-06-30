from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text2sql_assistant.common import normalize_sql, save_json
from text2sql_assistant.data import load_examples
from text2sql_assistant.pipeline import build_models, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Text2SQL Assistant")
    parser.add_argument("--dataset", default="data/sample_text2sql.jsonl", help="Path to a JSON or JSONL dataset")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["baseline", "classification", "generation", "llm", "all"],
        help="Which model to evaluate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = REPO_ROOT / args.dataset
    examples = load_examples(dataset_path)
    train_examples = [ex for ex in examples if ex.split == "train"]
    test_examples = [ex for ex in examples if ex.split == "test"]
    models = build_models(train_examples)

    if not test_examples:
        print("No test examples found.")
        return

    modes = ["baseline", "classification", "generation", "llm"] if args.mode == "all" else [args.mode]
    results = []
    for mode in modes:
        summary, preds = evaluate(models, test_examples, mode)
        results.append(summary.to_dict())
        print(f"[{mode}] accuracy={summary.accuracy:.4f} exact={summary.exact_match}/{summary.total}")
        print(f"artifacts: {summary.artifact_dir}")

    save_json(REPO_ROOT / "artifacts" / "latest_summary.json", results)


if __name__ == "__main__":
    main()
