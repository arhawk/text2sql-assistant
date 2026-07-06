import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=256, num_heads=4, num_layers=3, dropout=0.1, max_len=100):
        super().__init__()
        self.embed_dim = embed_dim

        self.src_embedding = nn.Embedding(src_vocab_size, embed_dim)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
        self.pos_decoder = PositionalEncoding(embed_dim, max_len)

        self.transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )

        self.generator = nn.Linear(embed_dim, tgt_vocab_size)

    def forward(self, src, tgt):
        src_mask = self._generate_src_mask(src)
        tgt_mask = self._generate_tgt_mask(tgt.size(1)).to(tgt.device)

        src_emb = self.pos_encoder(self.src_embedding(src))
        tgt_emb = self.pos_decoder(self.tgt_embedding(tgt))

        out = self.transformer(
            src_emb,
            tgt_emb,
            src_key_padding_mask=src_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=self._generate_src_mask(tgt)
        )
        return self.generator(out)

    def _generate_src_mask(self, seq):
        return (seq == 0)

    def _generate_tgt_mask(self, size):
        mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
