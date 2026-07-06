import torch
import torch.nn as nn

class FeedForwardClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        emb = self.embedding(x)        # (batch, seq_len, embed_dim)
        emb = emb.mean(dim=1)          # (batch, embed_dim)
        emb = self.dropout(emb)
        out = self.fc(emb)             # (batch, num_classes)
        return out
