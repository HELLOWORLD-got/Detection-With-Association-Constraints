import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiViewObjectQuery(nn.Module):
    def __init__(self, hidden_dim=256, num_queries=25, num_views=2):
        super().__init__()
        self.object_queries = nn.Parameter(torch.randn(num_views, num_queries, hidden_dim) * 0.02)
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries

        # Attention layer for triplet interaction
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)

        # Feed-forward networks for box prediction and confidence
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 4),  # (x, y, w, h)
        )

        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(), nn.Linear(hidden_dim * 2, 1)
        )

    def forward(self, instance_features, single_view_features, return_logits=False):
        """
        instance_features: (B, N, C)
        single_view_features: list of tensors, each of shape (B, C, H, W) for each view
        """
        B, N, C = instance_features.shape
        K = len(single_view_features)

        all_boxes = []
        all_confidences = []

        for k in range(K):
            # Get single view feature
            view_feature = single_view_features[k]
            Bv, Cv, Hv, Wv = view_feature.shape

            # Reshape view feature
            view_feature_flat = view_feature.flatten(2).transpose(1, 2)  # (B, H*W, C)

            # Paper Eq. 11 does not specify compatible key/value lengths.
            # Condition each learned view query on its bound global instance,
            # then attend to spatial tokens without repeating them N times.
            queries = instance_features + self.object_queries[k].unsqueeze(0)
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
