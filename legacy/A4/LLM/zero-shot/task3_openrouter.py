import json
import random
import re
import csv
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from openrouter_sdk import OpenRouter

# ========== 配置 ==========
MODEL = "meta-llama/llama-3.2-3b-instruct"
client = None


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

# ========== 工具函数 ==========
def clean_sql(sql):
    return re.sub(r'\s+', ' ', sql.strip()).lower()

def load_data(file_path):
    return [json.loads(line) for line in open(file_path)]

def substitute_vars(text, variables):
    if not variables:
        return text
    for k, v in variables.items():
        text = text.replace(k, v)
        text = text.replace(f'"{k}"', f'"{v}"')
    return text

def get_prompt(input_question, schema_text, examples=[]):
    shots_text = ""
    for ex in examples:
        question = substitute_vars(ex['text'], ex.get('variables', {}))
        sql = substitute_vars(ex['sql'][0], ex.get('variables', {}))
        shots_text += f"Q: {question}\nA: {sql}\n\n"
    return f"{schema_text}\n\n{shots_text}Q: {input_question}\nA:"

def evaluate(pred, refs):
    pred = clean_sql(pred)
    refs = [clean_sql(r) for r in refs]
    return pred in refs

def read_schema_from_csv(csv_file):
    schema = []
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            table = row['Table Name'].strip()
            field = row[' Field Name'].strip()
            if table == '-' or field == '-':
                continue
            schema.append(f"{table}.{field}")
    return "\n".join(schema)

# ========== 主逻辑 ==========
def run_eval(train_file, dev_file, schema_csv_file, shot_type="zero", shot_count=0):
    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    global client
    if client is None:
        client = OpenRouter(api_key=API_KEY, model=MODEL)
    train_data = load_data(train_file)
    dev_data = load_data(dev_file)
    schema_text = read_schema_from_csv(schema_csv_file)

    correct = 0
    total = len(dev_data)

    for item in tqdm(dev_data):
        question = substitute_vars(item['text'], item.get('variables', {}))
        if shot_type == "zero":
            shots = []
        else:
            shots = random.sample(train_data, shot_count)

        prompt = get_prompt(question, schema_text, shots)
        gen_sql = client.chat(prompt)

        if evaluate(gen_sql, item['sql']):
            correct += 1

    acc = correct / total
    print(f"[{shot_type}-shot | {shot_count}] Accuracy: {acc:.4f}")
