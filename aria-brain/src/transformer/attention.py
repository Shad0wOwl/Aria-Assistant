
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, embedding_size=64, num_heads=4):
        super().__init__()

        if embedding_size % num_heads != 0:
            raise ValueError(
                "Embedding size must be divisible by num_heads"
            )

        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads

        self.query = nn.Linear(embedding_size, embedding_size)
        self.key = nn.Linear(embedding_size, embedding_size)
        self.value = nn.Linear(embedding_size, embedding_size)

        self.output = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / (self.head_dim ** 0.5)

        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )

        scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)

        attended = weights @ v

        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, embed_dim)

        return self.output(attended)
