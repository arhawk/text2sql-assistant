from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text2sql_assistant.common import normalize_sql
from text2sql_assistant.data import load_examples
from text2sql_assistant.pipeline import build_models, predict_all, predict_one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text2SQL Assistant demo")
    parser.add_argument("--dataset", default="data/sample_text2sql.jsonl", help="Path to a JSON or JSONL dataset")
    parser.add_argument("--question", default=None, help="Single question to predict")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["baseline", "classification", "generation", "llm", "all"],
        help="Which model to run",
    )
    return parser.parse_args()


def print_prediction_table(predictions):
    for pred in predictions:
        print(f"\n[{pred.mode}]")
        print(f"score: {pred.score:.4f}")
        print(f"sql: {normalize_sql(pred.sql)}")
        print(f"matched question: {pred.matched_question}")
        if pred.notes:
            print(f"notes: {', '.join(pred.notes)}")


def main() -> None:
    args = parse_args()
    dataset_path = REPO_ROOT / args.dataset
    examples = load_examples(dataset_path)
    train_examples = [ex for ex in examples if ex.split == "train"]
    models = build_models(train_examples)

    if args.question:
        if args.mode == "all":
            predictions = predict_all(args.question, models)
            print_prediction_table(predictions)
        else:
            pred = predict_one(args.question, args.mode, models)
            print_prediction_table([pred])
        return

    print("Text2SQL Assistant")
    print("Type a question, or press Enter on an empty line to exit.")
    print("Available models: baseline, classification, generation, llm, all")

    while True:
        question = input("\nquestion> ").strip()
        if not question:
            break

        mode = input("mode [all]> ").strip() or "all"
        if mode == "all":
            predictions = predict_all(question, models)
            print_prediction_table(predictions)
        else:
            pred = predict_one(question, mode, models)
            print_prediction_table([pred])


if __name__ == "__main__":
    main()
