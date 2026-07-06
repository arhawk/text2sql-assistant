import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from A4.Classification.dataset import ClassificationDataset
from A4.Classification.utils import accuracy
from transformer import TransformerClassifier


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
    # 超参数配置
    embed_dim = 128
    batch_size = 64
    epochs = 150
    learning_rate = 1e-4

    # 使用 query_split 的数据路径
    data_path = '../../processed_data/query_split'
    train_set = ClassificationDataset(f'{data_path}/classification_train.jsonl')
    dev_set = ClassificationDataset(f'{data_path}/classification_dev.jsonl', word2id=train_set.word2id)
    test_set = ClassificationDataset(f'{data_path}/classification_test.jsonl', word2id=train_set.word2id)

    word2id = train_set.word2id
    num_classes = len(set([label for _, label in train_set]))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化 Transformer 模型
    model = TransformerClassifier(
        vocab_size=len(word2id),
        embed_dim=embed_dim,
        num_classes=num_classes,
        num_heads=4,
        num_layers=2,
        dropout=0.3,
        max_len=100
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🧠 正在训练 Transformer 模型（query_split）")
    print(f"Embedding dim: {embed_dim} | Batch size: {batch_size} | Epochs: {epochs} | LR: {learning_rate}\n")

    best_dev_acc = 0.0
    patience = 10
    patience_counter = 0
    best_model_path = './Transformer_best_query_model.pt'

    for epoch in range(1, epochs + 1):
        train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        dev_acc = eval_model(model, dev_loader, device)
        print(f'Epoch {epoch}: Train Acc = {train_acc:.4f}, Dev Acc = {dev_acc:.4f}')

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ 新的最佳模型已保存 (Dev Acc: {dev_acc:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"⏹️ Early stopping at epoch {epoch}. Best Dev Acc = {best_dev_acc:.4f}")
            break

    # fallback：防止 dev acc 始终为 0 时模型缺失
    if not os.path.exists(best_model_path):
        torch.save(model.state_dict(), best_model_path)
        print("⚠️ Dev Acc 恒为 0，保存最后一轮模型作为 fallback。")

    print(f"\n🎯 最佳 Dev Accuracy: {best_dev_acc:.4f}")

    # 最终测试集评估
    model.load_state_dict(torch.load(best_model_path))
    test_acc = eval_model(model, test_loader, device)
    print(f"🏁 最终在 query_split 测试集上的准确率: {test_acc:.4f}")


if __name__ == '__main__':
    main()
