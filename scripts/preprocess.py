from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from text2sql_assistant.data import load_examples
from text2sql_assistant.pipeline import export_dataset_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Text2SQL splits")
    parser.add_argument("--dataset", default="data/sample_text2sql.jsonl", help="Source dataset path")
    parser.add_argument("--output", default="data/processed", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = REPO_ROOT / args.dataset
    output_path = REPO_ROOT / args.output
    examples = load_examples(dataset_path)
    export_dataset_split(examples, output_path)
    print(f"processed splits written to {output_path}")


if __name__ == "__main__":
    main()
