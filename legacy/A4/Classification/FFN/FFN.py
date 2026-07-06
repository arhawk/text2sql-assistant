import torch
import torch.nn as nn

class FeedForwardClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        emb = self.embedding(x)             # (batch, seq_len, embed_dim)
        emb = emb.mean(dim=1)              # (batch, embed_dim)
        out = self.fc(emb)                 # (batch, num_classes)
        return out
