import os
import json
import re
from collections import defaultdict
from tqdm import tqdm

# 输入输出路径
INPUT_PATH = 'data/atis.json'
OUTPUT_DIR = 'processed_data'


# 工具函数
def normalize_sql(sql: str) -> str:
    return re.sub(r'\s+', ' ', sql.strip())

def choose_shortest_sql(sql_list):
    return min(sql_list, key=len)

def replace_variables(text, variables):
    for var, value in sorted(variables.items(), key=lambda x: len(x[0]), reverse=True):
        text = text.replace(var, value)
    return text

def tokenize(text):
    return text.strip().split()

def get_variable_tags(text_tokens, variables):
    tags = ['O'] * len(text_tokens)
    for var_name, value in variables.items():
        value_tokens = tokenize(value)
        for i in range(len(text_tokens) - len(value_tokens) + 1):
            if text_tokens[i:i + len(value_tokens)] == value_tokens:
                tags[i] = f"B-{var_name}"
                for j in range(1, len(value_tokens)):
                    tags[i + j] = f"I-{var_name}"
    return tags


def process_dataset(dataset, split_key, mode, min_template_count=5):
    print(f"\n🔄 正在处理模式: {mode} ...")
    template2id = {}

    # 统计模板频率
    template_counts = defaultdict(int)
    for entry in dataset:
        for sentence in entry['sentences']:
            if sentence.get(split_key, 'train') == 'train':
                sql = normalize_sql(choose_shortest_sql(entry['sql']))
                template_counts[sql] += 1

    # 保留高频模板
    frequent_templates = {sql for sql, count in template_counts.items() if count >= min_template_count}
    print(f"✅ 高频模板数量: {len(frequent_templates)}")

    cls_data, gen_data = defaultdict(list), defaultdict(list)

    for entry in tqdm(dataset):
        sql = normalize_sql(choose_shortest_sql(entry['sql']))
        if sql not in template2id:
            template2id[sql] = len(template2id)
        template_id = template2id[sql]

        for sentence in entry['sentences']:
            split = sentence.get(split_key, entry.get(split_key, 'train'))
            if mode == 'question_split' and sql not in frequent_templates:
                continue
            if mode == 'query_split' and split == 'train' and sql not in frequent_templates:
                continue

            text = sentence['text']
            variables = sentence.get('variables', {})
            real_text = replace_variables(text, variables)
            real_sql = replace_variables(sql, {
                k: (f'"{v}"' if not v.isdigit() else v) for k, v in variables.items()
            })

            tokens = tokenize(real_text)
            tags = get_variable_tags(tokens, variables)

            cls_data[split].append({
                'text': real_text,
                'tokens': tokens,
                'template_id': template_id,
                'tags': tags
            })

            gen_data[split].append({
                'text': real_text,
                'sql': real_sql
            })

    return cls_data, gen_data, template2id


def save_jsonl(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"📂 正在加载数据集: {INPUT_PATH}")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"✅ 加载完成，共 {len(dataset)} 条数据")

    for mode in ['question_split', 'query_split']:
        split_key = 'question-split' if mode == 'question_split' else 'query-split'
        cls_data, gen_data, template2id = process_dataset(dataset, split_key, mode)

        mode_dir = os.path.join(OUTPUT_DIR, mode)
        os.makedirs(mode_dir, exist_ok=True)

        for split in ['train', 'dev', 'test']:
            save_jsonl(cls_data[split], os.path.join(mode_dir, f'classification_{split}.jsonl'))
            save_jsonl(gen_data[split], os.path.join(mode_dir, f'generation_{split}.jsonl'))
            print(f"💾 已保存 {split} 数据 ({mode}): {len(cls_data[split])} 分类样本，{len(gen_data[split])} 生成样本")

        with open(os.path.join(mode_dir, 'templates.json'), 'w', encoding='utf-8') as f:
            json.dump(template2id, f, indent=2)

    print("\n✅ 全部模式处理完毕，数据已保存！")