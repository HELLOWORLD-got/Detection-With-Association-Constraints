import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiViewObjectQuery(nn.Module):
    """Predict one box and visibility score per instance in each view."""

    def __init__(self, hidden_dim=256, num_queries=25, num_views=2, mode="legacy"):
        super().__init__()
        if mode not in ("author", "legacy", "triplet"):
            raise ValueError(f"Unknown MVOQ mode: {mode}")
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_heads = 8
        if mode != "author":
            self.object_queries = nn.Parameter(
                torch.randn(num_views, num_queries, hidden_dim) * 0.02
            )

        if mode == "triplet":
            if hidden_dim % self.num_heads:
                raise ValueError("hidden_dim must be divisible by the number of attention heads")
            self.query_projection = nn.Linear(hidden_dim, hidden_dim)
            self.instance_projection = nn.Linear(hidden_dim, hidden_dim)
            self.view_projection = nn.Linear(hidden_dim, 2 * hidden_dim)
            self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=self.num_heads, batch_first=True
            )

        # Feed-forward networks for box prediction and confidence
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 4),  # (x, y, w, h)
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(), nn.Linear(hidden_dim * 2, 1)
        )

    def _triplet_attention(self, object_queries, instance_features, view_tokens):
        query = self.query_projection(object_queries)
        gate = 1 + self.instance_projection(instance_features).tanh()
        key, value = self.view_projection(view_tokens).chunk(2, dim=-1)

        # q_i · (gate_i * k_s) = (q_i * gate_i) · k_s.
        # Applying the gate on this side avoids a B × N × HW × C key tensor.
        query = (query * gate).unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        key = key.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        value = value.unflatten(-1, (self.num_heads, -1)).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value)
        return self.output_projection(attended.transpose(1, 2).flatten(-2))

    def forward(self, instance_features, single_view_features, return_logits=False):
        """
        instance_features: (B, N, C)
        single_view_features: list of tensors, each of shape (B, C, H, W) for each view
        """
        all_boxes = []
        all_confidences = []

        for k, view_feature in enumerate(single_view_features):
            view_feature_flat = view_feature.flatten(2).transpose(1, 2)  # (B, H*W, C)

            if self.mode == "triplet":
                object_features = self._triplet_attention(
                    self.object_queries[k], instance_features, view_feature_flat
                )
            else:
                # Uniform repetitions in the author code leave the weighted sum unchanged.
                queries = instance_features
                if self.mode == "legacy":
                    queries = queries + self.object_queries[k].unsqueeze(0)
                object_features, _ = self.attention(
                    queries, view_feature_flat, view_feature_flat, need_weights=False
                )

            # Predict boxes and confidences
            boxes = self.box_head(object_features)  # (B, N, 4)
            boxes = torch.sigmoid(boxes)  # Normalize to [0, 1]

            confidences = self.confidence_head(object_features)  # (B, N, 1)
            confidences = (confidences if return_logits else torch.sigmoid(confidences)).squeeze(
                -1
            )  # (B, N)

            all_boxes.append(boxes)
            all_confidences.append(confidences)

        return all_boxes, all_confidences
