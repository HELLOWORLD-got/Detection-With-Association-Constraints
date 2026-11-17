import torch
from torch import nn


class CrossAttentionLayer(nn.Module):
    """Paper CA: cross-attention + residual/post-norm, then position-wise FFN."""

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, query, memory):
        x = self.norm1(
            query + self.dropout(self.attention(query, memory, memory, need_weights=False)[0])
        )
        return self.norm2(x + self.dropout(self.ffn(x)))


class MultiViewFeatureFusion(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, num_layers=3, num_views=2):
        super().__init__()
        self.num_views = num_views
        self.ca_layers = nn.ModuleList(
            [CrossAttentionLayer(hidden_dim, num_heads) for _ in range(num_layers)]
        )
        # Paper specifies channel concatenation but omits the reduction operator.
        self.projection = nn.Conv2d(hidden_dim * num_views, hidden_dim, 1)
        self.sa_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    hidden_dim, num_heads, 4 * hidden_dim, dropout=0.1, batch_first=True
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, features):
        if len(features) != self.num_views or any(f.shape != features[0].shape for f in features):
            raise ValueError("Invalid view count or unequal feature shapes")
        b, c, h, w = features[0].shape
        tokens = [f.flatten(2).transpose(1, 2) for f in features]
        fused = []
        for i, x in enumerate(tokens):
            memory = (
                torch.cat([t for j, t in enumerate(tokens) if j != i], 1) if len(tokens) > 1 else x
            )
            for layer in self.ca_layers:
                x = layer(x, memory)
            fused.append(x.transpose(1, 2).reshape(b, c, h, w))
        x = self.projection(torch.cat(fused, 1)).flatten(2).transpose(1, 2)
        for layer in self.sa_layers:
            x = layer(x)
        return x.transpose(1, 2).reshape(b, c, h, w)
