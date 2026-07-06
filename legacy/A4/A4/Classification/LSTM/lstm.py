import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_tags, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)

        # 分类器部分
        self.template_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # 序列标注部分
        self.tag_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tags)
        )

    def forward(self, x, mask=None):
        emb = self.embedding(x)  # (B, T, E)
        lstm_out, (h_n, c_n) = self.lstm(emb)  # lstm_out: (B, T, H*2)

        # 模板分类使用最后一个隐状态（双向拼接）
        final_state = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, H*2)
        template_logits = self.template_fc(final_state)

        # 标签预测
        tag_logits = self.tag_fc(lstm_out)  # (B, T, tag_num)

        return template_logits, tag_logits
