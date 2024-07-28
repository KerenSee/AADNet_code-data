import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import copy


class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        # self.posEmbedding = PositionalEncoding(config.embed, config.seq_len, config.dropout, config.device)
        self.encoder = Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)
        ])

        self.fc1 = nn.Linear(config.seq_len * config.dim_model, 64)
        self.fc2 = nn.Linear(64, config.num_classes)
        # self.sm = nn.Softmax(dim=0)

    def forward(self, x):
        x = x.to(torch.float32)
        out = x
        # out = self.posEmbedding(x)
        for encoder in self.encoders:
            out = encoder(out)
        out1 = self.fc1(out.view(out.size(0), -1))
        out = self.fc2(out1)
        # out = self.sm(out)
        return out1, out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(dim_model, num_head, dropout)
        self.feed_forward = PositionWiseFeedForward(dim_model, hidden, dropout)


    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, embed, seq_len, dropout, device):
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([
            [pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(seq_len)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    @staticmethod
    def forward(query, key, value, scale=None):
        attention = torch.matmul(query, key.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = f.softmax(attention, dim=-1)
        context = torch.matmul(attention, value)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)

        self.attention = ScaledDotProductAttention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        query = self.fc_Q(x)
        key = self.fc_K(x)
        value = self.fc_V(x)

        q = query.view(batch_size * self.num_head, -1, self.dim_head)
        k = key.view(batch_size * self.num_head, -1, self.dim_head)
        v = value.view(batch_size * self.num_head, -1, self.dim_head)

        scale = k.size(-1) ** -0.5  # 缩放因子
        context = self.attention(q, k, v, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        # 残差连接
        out = out + x
        out = self.layer_norm(out)
        return out


class PositionWiseFeedForward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):   
        out = self.fc1(x)
        out = f.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out
