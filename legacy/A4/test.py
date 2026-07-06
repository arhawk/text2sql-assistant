# check_labels.py
import json
import os


def analyze_file(filepath):
    labels = []
    with open(filepath) as f:
        for line in f:
            data = json.loads(line)
            labels.append(data['template_id'])

    print(f"📊 文件: {os.path.basename(filepath)}")
    print(f"  标签范围: {min(labels)} ~ {max(labels)}")
    print(f"  唯一标签数: {len(set(labels))}")
    print(f"  样本数量: {len(labels)}")
    print("-" * 60)


if __name__ == '__main__':
    for mode in ['question_split', 'query_split']:
        print(f"\n🔍 分析模式: {mode.upper()}")
        for split in ['train', 'dev', 'test']:
            path = f"processed_data/{mode}/classification_{split}.jsonl"
            if os.path.exists(path):
                analyze_file(path)