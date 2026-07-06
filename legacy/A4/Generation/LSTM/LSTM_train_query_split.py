import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from A4.Generation.utils import accuracy

from A4.Generation.LSTM.lstm_seq2seq import EncoderLSTM, DecoderLSTM, Seq2SeqLSTM


class QueryGenerationDataset(torch.utils.data.Dataset):
    def __init__(self, path, word2id=None, target2id=None, max_len=100):
        self.data = []
        self.word2id = word2id or {}
        self.target2id = target2id or {}
        self.max_len = max_len

        self.UNK = '<unk>'
        self.PAD = '<pad>'
        self.SOS = '<sos>'
        self.EOS = '<eos>'

        self.special_tokens = [self.PAD, self.UNK, self.SOS, self.EOS]

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append((item['text'], item['sql']))

        if word2id is None or target2id is None:
            self.build_vocab()

        self.pad_id = self.word2id[self.PAD]
        self.sos_id = self.target2id[self.SOS]
        self.eos_id = self.target2id[self.EOS]

    def build_vocab(self):
        word_counter = Counter()
        target_counter = Counter()

        for text, sql in self.data:
            word_counter.update(text.strip().split())
            target_counter.update(sql.strip().split())

        self.word2id = {token: idx for idx, token in enumerate(self.special_tokens)}
        for word in word_counter:
            if word not in self.word2id:
                self.word2id[word] = len(self.word2id)

        self.target2id = {token: idx for idx, token in enumerate(self.special_tokens)}
        for word in target_counter:
            if word not in self.target2id:
                self.target2id[word] = len(self.target2id)

    def encode(self, tokens, vocab, add_sos_eos=False):
        ids = [vocab.get(token, vocab[self.UNK]) for token in tokens[:self.max_len - 2]]
        if add_sos_eos:
            return [vocab[self.SOS]] + ids + [vocab[self.EOS]]
        return ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, sql = self.data[idx]
        src_ids = self.encode(text.strip().split(), self.word2id)
        tgt_ids = self.encode(sql.strip().split(), self.target2id, add_sos_eos=True)

        src_tensor = torch.LongTensor(src_ids)
        tgt_tensor = torch.LongTensor(tgt_ids)

        return src_tensor, tgt_tensor


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_batch, tgt_batch


def train(model, dataloader, optimizer, criterion, device):
    model.train()
    correct = 0
    total = 0

    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        pred = output.argmax(-1)
        correct += (pred == tgt[:, 1:]).sum().item()
        total += tgt[:, 1:].numel()

    return correct / total


def evaluate(model, dataloader, criterion, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:, :-1])
            pred = output.argmax(-1)
            correct += (pred == tgt[:, 1:]).sum().item()
            total += tgt[:, 1:].numel()

    return correct / total


def main():
    data_path = '../../processed_data/query_split'
    batch_size = 32
    embed_dim = 256
    hidden_dim = 256
    lr = 1e-3
    epochs = 30
    max_len = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = QueryGenerationDataset(os.path.join(data_path, 'generation_train.jsonl'))
    dev_set = QueryGenerationDataset(os.path.join(data_path, 'generation_dev.jsonl'),
                                     word2id=train_set.word2id,
                                     target2id=train_set.target2id)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_set, batch_size=batch_size, collate_fn=collate_fn)

    encoder = EncoderLSTM(len(train_set.word2id), embed_dim, hidden_dim)
    decoder = DecoderLSTM(len(train_set.target2id), embed_dim, hidden_dim)
    model = Seq2SeqLSTM(encoder, decoder, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=train_set.pad_id)

    best_dev_acc = 0
    model_path = './LSTM_best_query_model.pt'

    print(f"\n🧠 正在训练 LSTM 生成模型（query split）")
    print(f"Embedding dim: {embed_dim} | Hidden dim: {hidden_dim} | Epochs: {epochs} | LR: {lr}\n")

    for epoch in range(1, epochs + 1):
        train_acc = train(model, train_loader, optimizer, criterion, device)
        dev_acc = evaluate(model, dev_loader, criterion, device)

        print(f"Epoch {epoch}: 🏋️‍♀️ Train Acc = {train_acc:.4f} | 🎯 Dev Acc = {dev_acc:.4f}")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), model_path)
            print(f"✅ 新的最佳模型已保存 (Dev Acc: {dev_acc:.4f})")

    print("\n📦 训练结束，加载最佳模型...")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("🧪 准备进行测试或生成评估...（待补充）")


if __name__ == '__main__':
    main()
