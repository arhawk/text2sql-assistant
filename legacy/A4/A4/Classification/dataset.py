import json
import torch
from torch.utils.data import Dataset

class ClassificationDataset(Dataset):
    def __init__(self, filepath, word2id=None, max_len=30):
        self.samples = []
        self.word2id = word2id
        self.max_len = max_len

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                tokens = item['tokens']
                label = item['template_id']
                self.samples.append((tokens, label))

        if self.word2id is None:
            self.word2id = self.build_vocab()

    def build_vocab(self):
        word2id = {'<PAD>': 0, '<UNK>': 1}
        for tokens, _ in self.samples:
            for token in tokens:
                if token not in word2id:
                    word2id[token] = len(word2id)
        return word2id

    def encode_tokens(self, tokens):
        ids = [self.word2id.get(tok, self.word2id['<UNK>']) for tok in tokens]
        if len(ids) < self.max_len:
            ids += [self.word2id['<PAD>']] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, label = self.samples[idx]
        input_ids = self.encode_tokens(tokens)
        return torch.tensor(input_ids), torch.tensor(label)
