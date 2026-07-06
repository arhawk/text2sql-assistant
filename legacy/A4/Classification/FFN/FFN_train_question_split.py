# FFN/FFN_train_question_split.py
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
    train_set = ClassificationDataset(f'{data_path}/classification_train.jsonl')
    dev_set = ClassificationDataset(f'{data_path}/classification_dev.jsonl', word2id=train_set.word2id)

    word2id = train_set.word2id
    num_classes = len(set([label for _, label in train_set]))

    embed_dim = 128
    batch_size = 64
    epochs = 100
    learning_rate = 5e-4

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FeedForwardClassifier(len(word2id), embed_dim, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🧠 正在训练 FFN 模型：")
    print(f"Embedding dim: {embed_dim} | Batch size: {batch_size} | Epochs: {epochs} | LR: {learning_rate}\n")

    for epoch in range(1, epochs + 1):
        train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        dev_acc = eval_model(model, dev_loader, device)
        print(f"Epoch {epoch}: Train Acc = {train_acc:.4f}, Dev Acc = {dev_acc:.4f}")


if __name__ == '__main__':
    main()