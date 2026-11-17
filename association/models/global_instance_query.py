import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GlobalInstanceQuery(nn.Module):
    def __init__(self, hidden_dim=256, num_queries=25, num_classes=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes

        # Learnable instance queries
        self.instance_queries = nn.Parameter(torch.randn(num_queries, hidden_dim))

        # Position encoding
        self.register_buffer("position_encoding", self._create_position_encoding(hidden_dim))

        # Attention layer for instance feature extraction
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)

        # Feed-forward network for category prediction
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, num_classes + 1),  # +1 for no-object class
        )

    def _create_position_encoding(self, d_model, max_h=100, max_w=100):
        if d_model % 4:
            raise ValueError("2D positional channels must be divisible by four")
        half = d_model // 2
        frequencies = torch.exp(torch.arange(0, half, 2).float() * (-math.log(10000.0) / half))
        y = torch.arange(max_h, dtype=torch.float32)[:, None] * frequencies
        x = torch.arange(max_w, dtype=torch.float32)[:, None] * frequencies
        py = torch.stack([y.sin(), y.cos()], -1).flatten(1)
        px = torch.stack([x.sin(), x.cos()], -1).flatten(1)
        pe = torch.cat(
            [py[:, None, :].expand(max_h, max_w, half), px[None, :, :].expand(max_h, max_w, half)],
            -1,
        )
        return pe.unsqueeze(0)

    def forward(self, global_feature, return_logits=False):
        """
        global_feature: (B, C, H, W)
        """
        B, C, H, W = global_feature.shape

        # Add position encoding
        if H <= self.position_encoding.shape[1] and W <= self.position_encoding.shape[2]:
            pos_enc = self.position_encoding[:, :H, :W].permute(0, 3, 1, 2).to(global_feature)
        else:
            pos_enc = F.interpolate(
                self.position_encoding.permute(0, 3, 1, 2),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            ).to(global_feature)
        global_feature = global_feature + pos_enc

        # Reshape for attention
        global_feature_flat = global_feature.flatten(2).transpose(1, 2)  # (B, H*W, C)

        # Expand instance queries to batch size
        queries = self.instance_queries.unsqueeze(0).repeat(B, 1, 1)  # (B, N, C)

        # Extract instance features using attention
        instance_features, _ = self.attention(
            queries, global_feature_flat, global_feature_flat, need_weights=False
        )  # (B, N, C)

        # Predict categories
        class_logits = self.classification_head(instance_features)  # (B, N, C+1)
        class_probs = F.softmax(class_logits, dim=-1)

        return instance_features, class_logits if return_logits else class_probs
