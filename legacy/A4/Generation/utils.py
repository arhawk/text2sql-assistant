import torch
import numpy as np

def accuracy(preds, labels):
    pred_classes = torch.argmax(preds, dim=1)
    correct = (pred_classes == labels).sum().item()
    total = labels.size(0)
    return correct / total

def load_glove(path, word2id, embed_dim=100):
    embeddings = np.random.uniform(-0.1, 0.1, (len(word2id), embed_dim)).astype(np.float32)
    found = 0

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            if word in word2id:
                embeddings[word2id[word]] = vec
                found += 1

    print(f"🔍 找到了 {found}/{len(word2id)} 个词的预训练词向量")
    return embeddings
