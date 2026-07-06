import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from A4.Classification.dataset import ClassificationDataset
from transformer import TransformerClassifier
from A4.Classification.utils import accuracy

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

    # 数据路径
    data_path = '../../processed_data/question_split'
    train_set = ClassificationDataset(f'{data_path}/classification_train.jsonl')
    dev_set = ClassificationDataset(f'{data_path}/classification_dev.jsonl', word2id=train_set.word2id)
    word2id = train_set.word2id

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(set([label for _, label in train_set]))

    # 初始化 Transformer 模型
    model = TransformerClassifier(
        vocab_size=len(word2id),
        embed_dim=embed_dim,
        num_classes=num_classes,
        num_heads=4,
        num_layers=2,
        dropout=0.3,
        max_len=100
    )
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🧠 正在训练 transformer 模型：")
    print(f"Embedding dim: {embed_dim} | Batch size: {batch_size} | Epochs: {epochs} | LR: {learning_rate}\n")

    # Early stopping 配置
    best_dev_acc = 0.0
    patience = 10
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        dev_acc = eval_model(model, dev_loader, device)
        print(f'Epoch {epoch}: Train Acc = {train_acc:.4f}, Dev Acc = {dev_acc:.4f}')

        # 保存最优模型
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            patience_counter = 0
            torch.save(model.state_dict(), './Transformer_best_question_model.pt')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"⏹️ Early stopping at epoch {epoch}. Best Dev Acc = {best_dev_acc:.4f}")
            break
    # 测试集评估
    test_set = ClassificationDataset(f'{data_path}/classification_test.jsonl', word2id=word2id)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    model.load_state_dict(torch.load('./Transformer_best_question_model.pt'))
    test_acc = eval_model(model, test_loader, device)
    print(f"🏁 最终在 question_split 测试集上的准确率: {test_acc:.4f}")


if __name__ == '__main__':
    main()
