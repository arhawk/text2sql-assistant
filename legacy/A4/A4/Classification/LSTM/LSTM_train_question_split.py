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

        # 创建一个dummy标签序列用于序列标注任务
        # 在实际应用中，你应该使用真实的序列标签
        seq_length = inputs.size(1)
        dummy_tags = torch.zeros(batch_size, seq_length, dtype=torch.long).to(device)

        optimizer.zero_grad()

        # 获取模型的两个输出
        template_logits, tag_logits = model(inputs)

        # 计算分类损失
        cls_loss = criterion_cls(template_logits, labels)

        # 计算序列标注损失 (使用dummy标签，实际应用中应该用真实标签)
        tag_loss = criterion_tag(tag_logits.view(-1, tag_logits.size(-1)), dummy_tags.view(-1))

        # 组合两个损失
        loss = cls_loss + tag_weight * tag_loss

        loss.backward()
        # 梯度裁剪，防止梯度爆炸
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

            # 只关注分类输出
            template_logits, _ = model(inputs)

            loss = criterion_cls(template_logits, labels)

            total_loss += loss.item() * batch_size
            total_acc += accuracy(template_logits, labels) * batch_size
            total_samples += batch_size

    return total_acc / total_samples, total_loss / total_samples


def main():
    # 设置随机种子以确保可复现性
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    data_path = '../../processed_data/question_split'
    train_set = ClassificationDataset(f'{data_path}/classification_train.jsonl')
    dev_set = ClassificationDataset(f'{data_path}/classification_dev.jsonl', word2id=train_set.word2id)

    word2id = train_set.word2id
    num_classes = len(set([label for _, label in train_set]))

    # 超参数优化
    embed_dim = 256  # 增加嵌入维度
    hidden_dim = 256  # 增加隐藏层维度
    batch_size = 32  # 使用较小的batch size可能有更好的泛化性能
    epochs = 100
    learning_rate = 2e-3
    dropout = 0.3  # 增加dropout防止过拟合
    tag_weight = 0.2  # 序列标注任务的权重

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_tags = len(train_set.all_tags) if hasattr(train_set, 'all_tags') else 5  # 如果没有标签，默认使用5个标签

    model = LSTMClassifier(len(word2id), embed_dim, hidden_dim, num_classes, num_tags, dropout=dropout).to(device)

    # 使用权重衰减减轻过拟合
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # 添加学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_tag = nn.CrossEntropyLoss(ignore_index=-1)  # -1 通常用于忽略填充标记

    print(f"\n🧠 正在训练 LSTM 模型：")
    print(f"Embedding dim: {embed_dim} | Hidden dim: {hidden_dim} | Dropout: {dropout}")
    print(f"Batch size: {batch_size} | Epochs: {epochs} | LR: {learning_rate}\n")

    best_dev_acc = 0
    best_epoch = 0
    patience = 10  # 早停的patience
    no_improve = 0

    for epoch in range(1, epochs + 1):
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, criterion_cls, criterion_tag, device,
                                            tag_weight)
        dev_acc, dev_loss = eval_model(model, dev_loader, criterion_cls, device)

        # 更新学习率
        scheduler.step(dev_acc)

        print(
            f"Epoch {epoch}: Train Acc = {train_acc:.4f}, Loss = {train_loss:.4f} | Dev Acc = {dev_acc:.4f}, Loss = {dev_loss:.4f}")

        # 保存最佳模型
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_epoch = epoch
            torch.save(model.state_dict(), './lstm_model_question.pt')
            no_improve = 0
            print(f"✅ 新的最佳模型已保存 (Dev Acc: {dev_acc:.4f})")
        else:
            no_improve += 1

        # 早停
        if no_improve >= patience:
            print(f"🛑 {patience} 个epoch没有改善，提前停止训练")
            break

    print(f"\n训练完成! 最佳模型在epoch {best_epoch}，开发集准确率: {best_dev_acc:.4f}")


    # 加载测试集
    test_set = ClassificationDataset(f'{data_path}/classification_test.jsonl', word2id=word2id)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # 使用保存的最佳模型评估测试集
    model.load_state_dict(torch.load('./lstm_model_question.pt'))
    test_acc, _ = eval_model(model, test_loader, criterion_cls, device)
    print(f"🏁 最终在 question_split 测试集上的准确率: {test_acc:.4f}")


if __name__ == '__main__':
    main()