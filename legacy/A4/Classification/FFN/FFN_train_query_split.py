import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import re
from tqdm import tqdm
from collections import defaultdict
from A4.Classification.dataset import ClassificationDataset
from A4.Classification.utils import accuracy
from FFN import FeedForwardClassifier


# ===== 工具函数 =====
def normalize_sql(sql):
    """标准化SQL语句，移除多余空格"""
    return re.sub(r'\s+', ' ', sql.strip())


def load_templates(template_path):
    """加载SQL模板映射"""
    with open(template_path, 'r', encoding='utf-8') as f:
        templates = json.load(f)
    # 创建模板ID到模板的映射
    id2template = {int(v): k for k, v in templates.items()}
    return id2template


def load_test_data(test_data_path):
    """加载测试数据，包含原始文本和目标SQL"""
    texts = []
    target_sqls = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            texts.append(item['text'])
            if 'sql' in item:
                target_sqls.append(item['sql'])
    return texts, target_sqls


# ===== 训练函数 =====
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_acc = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_acc += accuracy(outputs, labels)
    return total_acc / len(dataloader)


# ===== 基础评估函数 =====
def eval_model(model, dataloader, device):
    model.eval()
    total_acc = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            total_acc += accuracy(outputs, labels)
    return total_acc / len(dataloader)


# ===== 改进的评估函数 =====
def eval_model_with_sql(model, dataloader, device, id2template, test_data_path=None):
    """评估模型，包括模板分类准确率和SQL生成准确率"""
    model.eval()
    template_acc = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            template_acc += torch.sum(preds == labels).item() / len(labels)

    template_acc_final = template_acc / len(dataloader)

    # 如果提供了测试数据路径，计算SQL准确率
    if test_data_path and id2template:
        # 加载测试数据
        texts, target_sqls = load_test_data(test_data_path)

        # 检查数据长度是否匹配
        if len(all_preds) == len(texts) and len(texts) == len(target_sqls):
            # 生成预测的SQL
            pred_sqls = []
            for i, pred_id in enumerate(all_preds):
                if pred_id in id2template:
                    # 这里简化了SQL生成过程，实际中需要处理变量
                    template = id2template[pred_id]
                    # 在实际使用中，这里需要使用变量标注模型来填充变量
                    pred_sqls.append(normalize_sql(template))
                else:
                    # 模板ID不存在，视为错误
                    pred_sqls.append("")

            # 计算SQL准确率
            sql_correct = 0
            for pred, target in zip(pred_sqls, target_sqls):
                # 如果目标SQL是列表，检查预测是否匹配任何一个
                if isinstance(target, list):
                    if any(normalize_sql(t) == pred for t in target):
                        sql_correct += 1
                else:
                    if normalize_sql(target) == pred:
                        sql_correct += 1

            sql_acc = sql_correct / len(pred_sqls)
            return template_acc_final, sql_acc, all_preds, all_labels
        else:
            print(f"Warning: 数据长度不匹配! preds={len(all_preds)}, texts={len(texts)}, targets={len(target_sqls)}")

    return template_acc_final, 0.0, all_preds, all_labels


def analyze_errors(predictions, labels, dataset, id2template=None):
    """分析错误预测，生成错误分析报告"""
    error_indices = [i for i, (p, l) in enumerate(zip(predictions, labels)) if p != l]
    print(f"\n错误分析: 总共发现 {len(error_indices)} 个错误预测")

    # 统计每个类别的错误
    error_by_class = defaultdict(int)
    confusion_matrix = defaultdict(int)

    for idx in error_indices:
        pred = predictions[idx]
        true = labels[idx]
        error_by_class[true] += 1
        confusion_matrix[(true, pred)] += 1

    # 输出错误最多的类别
    print("\n错误最多的类别:")
    for cls, count in sorted(error_by_class.items(), key=lambda x: x[1], reverse=True)[:5]:
        template_name = f"Template {cls}"
        if id2template and cls in id2template:
            template_short = id2template[cls][:50] + "..." if len(id2template[cls]) > 50 else id2template[cls]
            template_name = f"{template_name} ({template_short})"
        print(f"  类别 {cls}: {count} 个错误")

    # 输出最常见的混淆对
    print("\n最常见的混淆对:")
    for (true, pred), count in sorted(confusion_matrix.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  真实: {true}, 预测: {pred}, 数量: {count}")

    # 如果提供了数据集，输出一些错误样例
    if dataset and len(error_indices) > 0:
        print("\n错误样例:")
        for i in range(min(5, len(error_indices))):
            idx = error_indices[i]
            # 尝试获取原始文本，如果没有属性则使用索引
            if hasattr(dataset, 'original_texts') and idx < len(dataset.original_texts):
                text = dataset.original_texts[idx]
            elif hasattr(dataset, 'data') and idx < len(dataset.data):
                text = dataset.data[idx]
            else:
                text = f"Example {idx}"

            pred = predictions[idx]
            true = labels[idx]

            print(f"\n样例 {i + 1}:")
            print(f"  文本: {text}")
            print(f"  真实类别: {true}")
            print(f"  预测类别: {pred}")

            if id2template:
                if true in id2template:
                    true_template = id2template[true][:100] + "..." if len(id2template[true]) > 100 else id2template[
                        true]
                    print(f"  真实模板: {true_template}")
                if pred in id2template:
                    pred_template = id2template[pred][:100] + "..." if len(id2template[pred]) > 100 else id2template[
                        pred]
                    print(f"  预测模板: {pred_template}")

    return error_indices, error_by_class, confusion_matrix


# ===== 完整的评估流程 =====
def full_evaluation(model, test_loader, test_set, device, template_path, test_data_path=None):
    """执行完整的模型评估"""
    print("\n🔍 开始模型评估...")

    # 加载模板映射
    id2template = None
    if os.path.exists(template_path):
        id2template = load_templates(template_path)
        print(f"✅ 已加载 {len(id2template)} 个模板")
    else:
        print(f"❌ 模板文件不存在: {template_path}")

    # 评估模型
    template_acc, sql_acc, all_preds, all_labels = eval_model_with_sql(
        model, test_loader, device, id2template, test_data_path
    )

    print(f"\n📊 评估结果:")
    print(f"  模板分类准确率: {template_acc:.4f}")
    if sql_acc > 0:
        print(f"  SQL生成准确率: {sql_acc:.4f}")

    # 计算每个类别的准确率
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for pred, label in zip(all_preds, all_labels):
        if pred == label:
            class_correct[label] += 1
        class_total[label] += 1

    print("\n📊 各类别准确率:")
    for label in sorted(class_total.keys())[:10]:  # 只显示前10个类别
        if class_total[label] > 0:
            acc = class_correct[label] / class_total[label]
            print(f"  类别 {label}: {acc:.4f} ({class_correct[label]}/{class_total[label]})")

    # 分析错误
    error_indices, error_by_class, confusion_matrix = analyze_errors(
        all_preds, all_labels, test_set, id2template
    )

    return {
        'template_accuracy': template_acc,
        'sql_accuracy': sql_acc,
        'predictions': all_preds,
        'labels': all_labels,
        'error_indices': error_indices,
        'error_by_class': error_by_class,
        'confusion_matrix': confusion_matrix
    }


# ===== 主函数 =====
def main():
    data_path = '../../processed_data/query_split'
    template_path = '../../processed_data/templates.json'  # 模板文件路径

    # 加载数据
    train_set = ClassificationDataset(os.path.join(data_path, 'classification_train.jsonl'))
    dev_set = ClassificationDataset(os.path.join(data_path, 'classification_dev.jsonl'), word2id=train_set.word2id)
    test_set = ClassificationDataset(os.path.join(data_path, 'classification_test.jsonl'), word2id=train_set.word2id)

    word2id = train_set.word2id
    num_classes = len(set([label for _, label in train_set]))

    # 超参数配置
    embed_dim = 128
    batch_size = 64
    epochs = 100
    learning_rate = 5e-4

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FeedForwardClassifier(len(word2id), embed_dim, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🧠 正在训练 FFN 模型（query_split）")
    print(f"Embedding dim: {embed_dim} | Batch size: {batch_size} | Epochs: {epochs} | LR: {learning_rate}\n")

    best_dev_acc = 0
    patience_counter = 0
    patience = 5  # 早停耐心值

    for epoch in range(1, epochs + 1):
        train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        dev_acc = eval_model(model, dev_loader, device)
        print(f"Epoch {epoch}: Train Acc = {train_acc:.4f}, Dev Acc = {dev_acc:.4f}")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            patience_counter = 0
            # 保存更完整的checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'word2id': word2id,
                'best_acc': best_dev_acc
            }, "best_ffn_model_query.pt")
            print(f"✅ 新的最佳模型已保存到 best_ffn_model_query.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⚠️ {patience}轮内验证集准确率无提升，提前结束训练")
                break

    # fallback：如果没有任何模型被保存，则保存最后一轮
    if not os.path.exists("best_ffn_model_query.pt"):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'word2id': word2id
        }, "best_ffn_model_query.pt")
        print("⚠️ Dev Acc 始终为 0，保存最后一轮模型作为 fallback。")

    print(f"\n🎯 最佳 Dev Accuracy: {best_dev_acc:.4f}")

    # 在测试集上评估最终模型
    print("\n开始在测试集上进行评估...")
    # 尝试加载检查点，处理不同格式
    checkpoint = torch.load("best_ffn_model_query.pt")
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 新格式（字典）
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 旧格式（直接state_dict）
        model.load_state_dict(checkpoint)

    # 使用简单评估获取基础准确率
    test_acc = eval_model(model, test_loader, device)
    print(f"基础测试准确率: {test_acc:.4f}")

    # 使用改进的完整评估功能
    test_data_path = os.path.join(data_path, 'generation_test.jsonl')  # 包含完整SQL的测试数据

    # 检查文件是否存在
    if os.path.exists(test_data_path):
        print(f"找到测试数据: {test_data_path}")
    else:
        print(f"警告: 测试数据文件不存在 {test_data_path}")
        test_data_path = None

    # 执行完整评估
    results = full_evaluation(
        model, test_loader, test_set, device,
        template_path, test_data_path
    )

    print(f"\n🏁 最终在 query_split 测试集上的评估结果:")
    print(f"  模板分类准确率: {results['template_accuracy']:.4f}")
    if 'sql_accuracy' in results and results['sql_accuracy'] > 0:
        print(f"  SQL生成准确率: {results['sql_accuracy']:.4f}")

    # 保存评估结果
    output_file = "ffn_evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            'template_accuracy': float(results['template_accuracy']),
            'sql_accuracy': float(results.get('sql_accuracy', 0.0)),
            'best_dev_accuracy': float(best_dev_acc)
        }, f, indent=2)

    print(f"\n✅ 评估结果已保存到 {output_file}")


if __name__ == '__main__':
    main()