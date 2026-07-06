import json
import re
import os
import argparse
from pathlib import Path
import requests
from Prompt_fixed import PromptBuilder

MODEL = "mistralai/mistral-medium-3"

def load_env_file():
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        env_path = parent / ".env"
        if not env_path.is_file():
            continue
        with env_path.open("r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip("'\"")
                os.environ.setdefault(key, value)
        break


load_env_file()
API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

def get_headers():
    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return {
        "Authorization": f"Bearer {API_KEY}",
        "X-Title": "llama-sql"
    }

def clean_sql(sql):
    return re.sub(r"\s+", " ", sql.strip())

def run_eval(train_file, test_file, shot_type='zero-shot', shot_count=0):
    builder = PromptBuilder(train_file, mode=shot_type, shot_num=shot_count)
    correct = 0
    total = 0

    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            prompt = builder.build_prompt(item['text'])

            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers=get_headers(),  # ✅ 确保 headers 是新鲜的
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3
                }
            )

            if response.status_code != 200:
                print("Error:", response.json())
                continue

            prediction = response.json()['choices'][0]['message']['content']
            pred_sql = clean_sql(prediction)
            gold_sql = clean_sql(item['sql'])

            if pred_sql == gold_sql:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0
    print(f"[{shot_type} | {shot_count} shot] Accuracy: {acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Run the legacy OpenRouter Text2SQL evaluation")
    parser.add_argument("--train-file", default="generation_train.jsonl", help="Training prompt examples JSONL")
    parser.add_argument("--test-file", default="generation_test.jsonl", help="Evaluation JSONL")
    parser.add_argument("--shot-type", default="zero-shot", help="Prompting mode")
    parser.add_argument("--shot-count", type=int, default=0, help="Number of few-shot examples")
    args = parser.parse_args()
    run_eval(args.train_file, args.test_file, args.shot_type, args.shot_count)


if __name__ == "__main__":
    main()
