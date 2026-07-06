import torch
import torch.nn as nn


class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, padding_idx=0):
        super(EncoderLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        # src: (batch_size, src_len)
        embedded = self.embedding(src)  # (batch_size, src_len, embed_dim)
        outputs, (hidden, cell) = self.lstm(embedded)  # outputs: (batch_size, src_len, hidden_dim)
        return hidden, cell


class DecoderLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, padding_idx=0):
        super(DecoderLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, tgt, hidden, cell):
        # tgt: (batch_size, tgt_len)
        embedded = self.embedding(tgt)  # (batch_size, tgt_len, embed_dim)
        outputs, _ = self.lstm(embedded, (hidden, cell))  # (batch_size, tgt_len, hidden_dim)
        predictions = self.fc_out(outputs)  # (batch_size, tgt_len, vocab_size)
        return predictions


class Seq2SeqLSTM(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2SeqLSTM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt):
        # src: (batch_size, src_len)
        # tgt: (batch_size, tgt_len)
        hidden, cell = self.encoder(src)
        output = self.decoder(tgt, hidden, cell)
        return output  # (batch_size, tgt_len, vocab_size)

    def generate(self, src, max_len, start_token_id):
        # src: (batch_size, src_len)
        hidden, cell = self.encoder(src)
        batch_size = src.size(0)
        inputs = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=self.device)  # [B, 1]
        outputs = []

        for _ in range(max_len):
            logits = self.decoder(inputs, hidden, cell)  # logits: (B, step, V)
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)  # (B, 1)
            outputs.append(next_token)
            inputs = next_token

        return torch.cat(outputs, dim=1)  # (B, max_len)
