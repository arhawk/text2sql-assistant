import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from A4.Classification.dataset import ClassificationDataset
from A4.Classification.utils import accuracy
from FFN import FeedForwardClassifier


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


def main():
    data_path = '../../processed_data/question_split'

    # === 加载数据集 ===
    train_set = ClassificationDataset(f'{data_path}/classification_train.jsonl')
    dev_set = ClassificationDataset(f'{data_path}/classification_dev.jsonl', word2id=train_set.word2id)
    test_set = ClassificationDataset(f'{data_path}/classification_test.jsonl', word2id=train_set.word2id)

    word2id = train_set.word2id
    num_classes = len(set([label for _, label in train_set]))

    # === 读取 template2id 文件（用于预测阶段还原 SQL）===
    templates_path = f'{data_path}/../templates.json'
    with open(templates_path) as f:
        template2id = json.load(f)

    embed_dim = 128
    batch_size = 64

    epochs = 195
    learning_rate = 5e-4

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FeedForwardClassifier(len(word2id), embed_dim, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🧠 正在训练 FFN 模型：")
    print(f"Embedding dim: {embed_dim} | Batch size: {batch_size} | Epochs: {epochs} | LR: {learning_rate}\n")

    best_dev_acc = 0.0
    model_save_path = "ffn_model_question.pt"

    for epoch in range(1, epochs + 1):
        train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        dev_acc = eval_model(model, dev_loader, device)
        print(f"Epoch {epoch}: Train Acc = {train_acc:.4f}, Dev Acc = {dev_acc:.4f}")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'word2id': word2id,
                'template2id': template2id
            }, model_save_path)
            print("✅ 已保存新最佳模型")
            # 保存词表
            with open("word2id.json", "w", encoding="utf-8") as f:
                json.dump(word2id, f, indent=2, ensure_ascii=False)

            # 保存模板映射
            with open("template2id.json", "w", encoding="utf-8") as f:
                json.dump(template2id, f, indent=2, ensure_ascii=False)

    # 若从未保存过最佳模型（极端情况）
    if not os.path.exists(model_save_path):
        torch.save({
            'model_state_dict': model.state_dict(),
            'word2id': word2id,
            'template2id': template2id
        }, model_save_path)
        print("⚠️ 未获得任何提升，保存最后一轮模型")
        # 保存词表
        with open("word2id.json", "w", encoding="utf-8") as f:
            json.dump(word2id, f, indent=2, ensure_ascii=False)

        # 保存模板映射
        with open("template2id.json", "w", encoding="utf-8") as f:
            json.dump(template2id, f, indent=2, ensure_ascii=False)

    print(f"\n🎯 最佳 Dev Accuracy: {best_dev_acc:.4f}")

    # === 加载最佳模型，在 test 集上评估 ===
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_acc = eval_model(model, test_loader, device)
    print(f"🏁 最终在 Geography 测试集（question_split）上的准确率: {test_acc:.4f}")


if __name__ == '__main__':
    main()
