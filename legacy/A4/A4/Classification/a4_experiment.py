import json
import os
import re
import torch
from torch.utils.data import DataLoader
from FFN.FFN import FeedForwardClassifier
from LSTM.lstm import LSTMClassifier
from dataset import ClassificationDataset
import matplotlib.pyplot as plt

# === Paths ===
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
question_split_path = f'{base_path}/processed_data/question_split'
ffn_model_path = f'{base_path}/Classification/FFN/ffn_model_question.pt'
lstm_model_path = f'{base_path}/Classification/LSTM/lstm_model_question.pt'
dev_path = f'{question_split_path}/classification_dev.jsonl'

# === Load FFN checkpoint ===
checkpoint = torch.load(ffn_model_path, map_location='cpu')
word2id = checkpoint['word2id']
template2id = checkpoint['template2id']
id2template = {v: k for k, v in template2id.items()}

# ✅ 自动推断模型输出维度
num_classes = checkpoint['model_state_dict']['fc.6.bias'].shape[0]

# === Load dev set ===
dev_set = ClassificationDataset(dev_path, word2id=word2id)
dev_loader = DataLoader(dev_set, batch_size=32)
dev_data = [json.loads(line) for line in open(dev_path, encoding='utf-8')]

# === Variable substitution ===
def replace_variables(template_sql, variables):
    for var, val in sorted(variables.items(), key=lambda x: len(x[0]), reverse=True):
        template_sql = template_sql.replace(var, f'"{val}"' if not val.isdigit() else val)
    return template_sql

# === Matching functions ===
def normalize_sql_structure(sql):
    sql = sql.strip().lower()
    sql = re.sub(r'\s+', ' ', sql)
    if ' where ' not in sql:
        return sql
    head, where_clause = sql.split(' where ', 1)
    conditions = [c.strip() for c in where_clause.split(' and ')]
    conditions.sort()
    return head.strip() + ' where ' + ' and '.join(conditions)

def exact_match(pred, true):
    return pred.strip().lower() == true.strip().lower()

def relaxed_match(pred, true):
    return normalize_sql_structure(pred) == normalize_sql_structure(true)

# === Evaluate a model and optionally analyze mismatches ===
def evaluate_model(model, dev_loader, dev_data, analyze=False, model_name=''):
    model.eval()
    predicted_template_ids = []
    with torch.no_grad():
        for inputs, _ in dev_loader:
            outputs = model(inputs) if isinstance(model, FeedForwardClassifier) else model(inputs)[0]
            preds = torch.argmax(outputs, dim=1)
            predicted_template_ids.extend(preds.tolist())

    final_sqls, true_sqls = [], []
    for pred_id, item in zip(predicted_template_ids, dev_data):
        pred_template = id2template.get(pred_id, '[UNKNOWN TEMPLATE]')
        pred_sql = replace_variables(pred_template, item.get('variables', {}))
        true_template = id2template.get(item['template_id'], '[UNKNOWN TEMPLATE]')
        true_sql = replace_variables(true_template, item.get('variables', {}))
        final_sqls.append(pred_sql)
        true_sqls.append(true_sql)

    exact_correct = sum(exact_match(p, t) for p, t in zip(final_sqls, true_sqls))
    relaxed_correct = sum(relaxed_match(p, t) for p, t in zip(final_sqls, true_sqls))

    if analyze:
        mismatch_cases = []
        for p, t in zip(final_sqls, true_sqls):
            if not exact_match(p, t) and relaxed_match(p, t):
                mismatch_cases.append((p, t))
        print(f"\n🔍 [{model_name}] Exact wrong but Relaxed correct: {len(mismatch_cases)} cases")
        if mismatch_cases:
            print("🔬 Sample cases showing WHERE order mismatch only:\n")
            for i, (p, t) in enumerate(mismatch_cases[:3]):
                print(f"[Case {i+1}]\nPredicted: {p}\nExpected : {t}\n")

    return exact_correct, relaxed_correct, len(final_sqls)

# === Load and evaluate FFN ===
ffn = FeedForwardClassifier(len(word2id), 128, num_classes)
ffn.load_state_dict(checkpoint['model_state_dict'])
ffn_exact, ffn_relaxed, total = evaluate_model(ffn, dev_loader, dev_data, analyze=True, model_name="FFN")

# === Load and evaluate LSTM ===
lstm = LSTMClassifier(vocab_size=len(word2id), embed_dim=256, hidden_dim=256,
                      num_classes=num_classes, num_tags=5, dropout=0.3)
lstm.load_state_dict(torch.load(lstm_model_path, map_location='cpu'))
lstm_exact, lstm_relaxed, _ = evaluate_model(lstm, dev_loader, dev_data, analyze=True, model_name="LSTM")

# === Print Results ===
print(f"\n📊 Total Examples: {total}")
print(f"FFN  - Exact: {ffn_exact / total:.2%}, Relaxed: {ffn_relaxed / total:.2%}")
print(f"LSTM - Exact: {lstm_exact / total:.2%}, Relaxed: {lstm_relaxed / total:.2%}")

# === Plot ===
labels = ['FFN', 'LSTM']
x = range(len(labels))
bar_width = 0.35

exact = [ffn_exact / total, lstm_exact / total]
relaxed = [ffn_relaxed / total, lstm_relaxed / total]

plt.figure(figsize=(8, 5))
plt.bar(x, exact, width=bar_width, label='Exact Match', color='skyblue')
plt.bar([i + bar_width for i in x], relaxed, width=bar_width, label='Relaxed Match', color='orange')
plt.xticks([i + bar_width / 2 for i in x], labels)
plt.ylim(0.8, 1.0)
plt.title('FFN vs LSTM (Question-Split Evaluation)')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.grid(axis='y')
plt.show()
