import os
import json
import re
from collections import defaultdict
from tqdm import tqdm

# 输入输出路径
INPUT_PATH = 'data/atis.json'
OUTPUT_DIR = 'processed_data'


def normalize_sql(sql: str) -> str:
    """对SQL语句进行标准化，移除多余空格"""
    return re.sub(r'\s+', ' ', sql.strip())


def choose_shortest_sql(sql_list):
    """选择最短的SQL语句作为模板"""
    return min(sql_list, key=lambda x: len(x))


def replace_variables(text, variables):
    """替换文本中的变量为实际值"""
    # 按变量名长度降序排序，避免子字符串替换问题
    for var, value in sorted(variables.items(), key=lambda x: len(x[0]), reverse=True):
        text = text.replace(var, value)
    return text


def tokenize(text):
    """简单的空格分词"""
    return text.strip().split()


def get_variable_tags(text_tokens, variables):
    """为每个token分配变量标签，使用BIO标注方案"""
    tags = ['O'] * len(text_tokens)
    for var_name, value in variables.items():
        value_tokens = tokenize(value)
        # 查找所有匹配位置
        for i in range(len(text_tokens) - len(value_tokens) + 1):
            if text_tokens[i:i + len(value_tokens)] == value_tokens:
                tags[i] = f"B-{var_name}"  # 开始标签
                for j in range(1, len(value_tokens)):
                    tags[i + j] = f"I-{var_name}"  # 中间标签
    return tags


def process_entry(entry, split_key, use_train_templates=True, train_templates=None):
    """处理单个条目

    Args:
        entry: 数据条目
        split_key: 数据集划分的键名
        use_train_templates: 是否只使用训练集中的模板
        train_templates: 训练集中的模板集合，用于过滤

    Returns:
        两个列表：用于分类的数据和用于生成的数据
    """
    results_cls = []
    results_gen = []

    sql_list = entry['sql']
    template_sql = normalize_sql(choose_shortest_sql(sql_list))

    # 确定当前条目的split
    entry_level_split = entry.get(split_key, 'train')

    # 如果是训练集，且需要过滤低频模板
    if entry_level_split == 'train' and use_train_templates and train_templates is not None:
        if template_sql not in train_templates:
            return [], []  # 过滤训练集中的低频模板

    # 为模板分配ID
    if template_sql not in template2id:
        template_id = len(template2id)
        template2id[template_sql] = template_id
    else:
        template_id = template2id[template_sql]

    for sentence in entry['sentences']:
        # 确定当前句子的split
        if split_key == 'question-split' and 'question-split' in sentence:
            split = sentence['question-split']
        else:
            split = entry_level_split

        text = sentence['text']
        variables = sentence['variables']

        # 替换变量，创建实际文本和SQL
        real_text = replace_variables(text, variables)
        # 为SQL中的变量添加引号，数字除外
        real_sql = replace_variables(template_sql, {
            k: (f'"{v}"' if not v.isdigit() else v)
            for k, v in variables.items()
        })

        tokens = tokenize(real_text)
        tags = get_variable_tags(tokens, variables)

        results_cls.append((split, {
            'text': real_text,
            'tokens': tokens,
            'template_id': template_id,
            'tags': tags
        }))

        results_gen.append((split, {
            'text': real_text,
            'sql': real_sql
        }))

    return results_cls, results_gen


def save_jsonl(data, out_path):
    """保存数据为JSONL格式"""
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')


if __name__ == '__main__':
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载数据集
    print(f"📂 正在加载数据集: {INPUT_PATH}")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"✅ 加载完成，共 {len(dataset)} 条数据")

    template2id = {}
    splits = ['train', 'dev', 'test']

    for mode in ['question_split', 'query_split']:
        print(f'\n🔄 正在处理模式: {mode} ...')
        split_key = 'question-split' if mode == 'question_split' else 'query-split'

        # 统计每个split中的样本数量
        split_counts = defaultdict(int)
        for entry in dataset:
            entry_split = entry.get(split_key, 'train')
            for sentence in entry['sentences']:
                # 确定当前句子的split
                if split_key == 'question-split' and 'question-split' in sentence:
                    current_split = sentence['question-split']
                else:
                    current_split = entry_split
                split_counts[current_split] += 1

        print(f"📊 各分割样本数量: {dict(split_counts)}")

        # 检查不同split中有多少唯一模板
        split_templates = defaultdict(set)
        for entry in dataset:
            entry_split = entry.get(split_key, 'train')
            template_sql = normalize_sql(choose_shortest_sql(entry['sql']))

            for sentence in entry['sentences']:
                # 确定当前句子的split
                if split_key == 'question-split' and 'question-split' in sentence:
                    current_split = sentence['question-split']
                else:
                    current_split = entry_split
                split_templates[current_split].add(template_sql)

        print(f"📊 各分割唯一模板数量: {dict((k, len(v)) for k, v in split_templates.items())}")

        # 计算模板重叠情况
        train_templates_set = split_templates['train']
        for split in ['dev', 'test']:
            if split in split_templates:
                overlap = len(train_templates_set.intersection(split_templates[split]))
                total = len(split_templates[split])
                if total > 0:
                    print(f"📊 训练集与{split}集模板重叠: {overlap}/{total} ({overlap / total * 100:.2f}%)")

        # Step 1：统计训练集中SQL模板的频率
        raw_template_counts = defaultdict(int)
        print("📊 统计训练集模板频率...")
        for entry in dataset:
            entry_split = entry.get(split_key, 'train')
            for sentence in entry['sentences']:
                # 确定当前句子的split
                if split_key == 'question-split' and 'question-split' in sentence:
                    current_split = sentence['question-split']
                else:
                    current_split = entry_split

                if current_split == 'train':
                    sql_list = entry['sql']
                    template_sql = normalize_sql(choose_shortest_sql(sql_list))
                    raw_template_counts[template_sql] += 1

        # Step 2：筛选频率高于阈值的模板
        min_template_count = 5  # 最小模板频率阈值
        train_templates = {sql for sql, count in raw_template_counts.items()
                           if count >= min_template_count}

        print(f"✅ 保留模板数: {len(train_templates)}/{len(raw_template_counts)}")

        # Step 3：处理数据
        classification_data = defaultdict(list)
        generation_data = defaultdict(list)

        # 决定处理策略
        if mode == 'query_split':
            # 对query_split模式，只过滤训练集，保留dev和test所有数据
            use_train_templates = False
            print("🔧 策略: 只过滤训练集中的低频模板，保留验证集和测试集所有数据")
        else:
            # 对question_split模式，过滤所有数据集中不在训练集频繁模板里的数据
            use_train_templates = True
            print("🔧 策略: 只保留在训练集中高频出现的模板")

        print("🔄 处理数据集...")
        for entry in tqdm(dataset):
            entry_split = entry.get(split_key, 'train')

            # 处理条目，根据策略决定是否过滤
            if use_train_templates:
                # 对所有分割都使用相同的模板过滤
                cls_list, gen_list = process_entry(entry, split_key, True, train_templates)
            else:
                # 只对训练集进行模板过滤
                if entry_split == 'train':
                    cls_list, gen_list = process_entry(entry, split_key, True, train_templates)
                else:
                    cls_list, gen_list = process_entry(entry, split_key, False, None)

            for split, item in cls_list:
                classification_data[split].append(item)
            for split, item in gen_list:
                generation_data[split].append(item)

        # Step 4：保存数据
        mode_dir = os.path.join(OUTPUT_DIR, mode)
        os.makedirs(mode_dir, exist_ok=True)

        for split in splits:
            cls_path = os.path.join(mode_dir, f'classification_{split}.jsonl')
            gen_path = os.path.join(mode_dir, f'generation_{split}.jsonl')

            save_jsonl(classification_data[split], cls_path)
            save_jsonl(generation_data[split], gen_path)

            print(
                f"💾 保存 {split} 数据: {len(classification_data[split])} 条分类样本, {len(generation_data[split])} 条生成样本")

    # 保存模板映射
    with open(os.path.join(OUTPUT_DIR, 'templates.json'), 'w', encoding='utf-8') as f:
        json.dump(template2id, f, indent=2)

    print("\n✅ 数据预处理完成！所有数据已保存至 processed_data/")