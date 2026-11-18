import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiViewObjectQuery(nn.Module):
    def __init__(self, hidden_dim=256, num_queries=25):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        
        # Attention layer for triplet interaction
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Feed-forward networks for box prediction and confidence
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 4)  # (x, y, w, h)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, 1)
        )
    
    def forward(self, instance_features, single_view_features):
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
            
            # Expand instance features for each view
            expanded_instance = instance_features.unsqueeze(1).repeat(1, Hv*Wv, 1, 1)  # (B, H*W, N, C)
            expanded_instance = expanded_instance.reshape(B, Hv*Wv*N, C)  # (B, H*W*N, C)
            
            # Use view feature as keys and values
            keys = view_feature_flat.unsqueeze(1).repeat(1, N, 1, 1).reshape(B, Hv*Wv*N, C)
            values = view_feature_flat.unsqueeze(1).repeat(1, N, 1, 1).reshape(B, Hv*Wv*N, C)
            
            # Apply attention
            object_features, _ = self.attention(
                expanded_instance,
                keys,
                values
            )  # (B, H*W*N, C)
            
            # Reshape back to (B, N, H*W, C)
            object_features = object_features.reshape(B, N, Hv*Wv, C)
            
            # Average over spatial dimensions
            object_features = object_features.mean(dim=2)  # (B, N, C)
            
            # Predict boxes and confidences
            boxes = self.box_head(object_features)  # (B, N, 4)
            boxes = torch.sigmoid(boxes)  # Normalize to [0, 1]
            
            confidences = self.confidence_head(object_features)  # (B, N, 1)
            confidences = torch.sigmoid(confidences).squeeze(-1)  # (B, N)
            
            all_boxes.append(boxes)
            all_confidences.append(confidences)
        
        return all_boxes, all_confidences
