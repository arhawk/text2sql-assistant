import torch
import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        embedded = self.embedding(x)                  # [B, L, D]
        conv_input = embedded.transpose(1, 2)         # [B, D, L]
        pooled = self.encoder(conv_input).squeeze(-1) # [B, D]
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)              # [B, C]
        return logits
