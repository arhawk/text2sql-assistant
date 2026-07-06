import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 简单池化聚合 token 向量
        )
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        pooled = embedded.transpose(1, 2)  # [batch_size, embed_dim, seq_len]
        pooled = self.encoder(pooled).squeeze(-1)  # [batch_size, embed_dim]
        logits = self.classifier(pooled)  # [batch_size, num_classes]
        return logits
