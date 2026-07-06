import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from A4.Classification.dataset import ClassificationDataset
from A4.Classification.utils import accuracy
from lstm import LSTMClassifier
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_epoch(model, dataloader, optimizer, criterion_cls, criterion_tag, device, tag_weight=0.3):
    model.train()
    total_acc = 0
    total_loss = 0
    total_samples = 0

    for inputs, labels in dataloader:
        batch_size = inputs.size(0)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # dummy tags
        seq_length = inputs.size(1)
        dummy_tags = torch.zeros(batch_size, seq_length, dtype=torch.long).to(device)

        optimizer.zero_grad()
        template_logits, tag_logits = model(inputs)

        cls_loss = criterion_cls(template_logits, labels)
        tag_loss = criterion_tag(tag_logits.view(-1, tag_logits.size(-1)), dummy_tags.view(-1))
        loss = cls_loss + tag_weight * tag_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch_size
        total_acc += accuracy(template_logits, labels) * batch_size
        total_samples += batch_size

    return total_acc / total_samples, total_loss / total_samples


def eval_model(model, dataloader, criterion_cls, device):
    model.eval()
    total_acc = 0
    total_loss = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            batch_size = inputs.size(0)
            inputs = inputs.to(device)
            labels = labels.to(device)

            template_logits, _ = model(inputs)
            loss = criterion_cls(template_logits, labels)

            total_loss += loss.item() * batch_size
            total_acc += accuracy(template_logits, labels) * batch_size
            total_samples += batch_size

    return total_acc / total_samples, total_loss / total_samples


def main():
    # 固定随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    # 数据加载
    data_path = '../../processed_data/query_split'
    train_set = ClassificationDataset(f'{data_path}/classification_train.jsonl')
    dev_set = ClassificationDataset(f'{data_path}/classification_dev.jsonl', word2id=train_set.word2id)
    test_set = ClassificationDataset(f'{data_path}/classification_test.jsonl', word2id=train_set.word2id)

    word2id = train_set.word2id
    num_classes = len(set([label for _, label in train_set]))

    embed_dim = 256
    hidden_dim = 256
    batch_size = 32
    epochs = 100
    learning_rate = 2e-3
    dropout = 0.3
    tag_weight = 0.2

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_tags = len(train_set.all_tags) if hasattr(train_set, 'all_tags') else 5

    model = LSTMClassifier(len(word2id), embed_dim, hidden_dim, num_classes, num_tags, dropout=dropout).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_tag = nn.CrossEntropyLoss(ignore_index=-1)

    print(f"\n🧠 正在训练 LSTM 模型（query_split）:")
    print(f"Embedding dim: {embed_dim} | Hidden dim: {hidden_dim} | Dropout: {dropout}")
    print(f"Batch size: {batch_size} | Epochs: {epochs} | LR: {learning_rate}\n")

    best_dev_acc = 0
    best_epoch = 0
    patience = 10
    no_improve = 0
    model_path = './LSTM_best_query_model.pt'

    for epoch in range(1, epochs + 1):
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, criterion_cls, criterion_tag, device, tag_weight)
        dev_acc, dev_loss = eval_model(model, dev_loader, criterion_cls, device)

        scheduler.step(dev_acc)

        print(f"Epoch {epoch}: Train Acc = {train_acc:.4f}, Loss = {train_loss:.4f} | Dev Acc = {dev_acc:.4f}, Loss = {dev_loss:.4f}")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
            no_improve = 0
            print(f"✅ 新的最佳模型已保存 (Dev Acc: {dev_acc:.4f})")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"🛑 {patience} 个 epoch 没有提升，提前停止训练")
            break

    # fallback 保存（防止模型从未保存）
    if not os.path.exists(model_path):
        torch.save(model.state_dict(), model_path)
        print("⚠️ Dev Acc 始终为 0，已保存最后一轮模型作为 fallback。")

    print(f"\n训练完成！最佳模型在 epoch {best_epoch}，Dev Acc = {best_dev_acc:.4f}")

    # 在测试集上评估
    model.load_state_dict(torch.load(model_path))
    test_acc, _ = eval_model(model, test_loader, criterion_cls, device)
    print(f"🏁 最终在 query_split 测试集上的准确率: {test_acc:.4f}")


if __name__ == '__main__':
    main()
