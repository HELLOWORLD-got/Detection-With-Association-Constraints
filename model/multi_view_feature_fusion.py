import torch
import torch.nn as nn

class MultiViewFeatureFusion(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Cross-View Attention (CA) layers
        self.ca_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            self.ca_layers.append(layer)
        
        # Global Self-Attention (SA) layers
        self.sa_layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            self.sa_layers.append(layer)
    
    def forward(self, features):
        """
        features: list of tensors, each of shape (B, C, H, W) for each view
        """
        B, C, H, W = features[0].shape
        K = len(features)
        
        # Reshape features to (B, H*W, C)
        reshaped_features = []
        for feat in features:
            reshaped = feat.flatten(2).transpose(1, 2)  # (B, H*W, C)
            reshaped_features.append(reshaped)
        
        # Cross-View Attention
        cross_view_features = []
        for i in range(K):
            # Query: current view
            query = reshaped_features[i]
            
            # Keys and values: all other views
            other_views = []
            for j in range(K):
                if j != i:
                    other_views.append(reshaped_features[j])
            if other_views:
                kv = torch.cat(other_views, dim=1)
            else:
                kv = query
            
            # Apply CA layers
            x = query
            for layer in self.ca_layers:
                x = layer(x, kv, kv)
            
            cross_view_features.append(x)
        
        # Concatenate cross-view features
        concatenated = torch.cat(cross_view_features, dim=1)  # (B, K*H*W, C)
        
        # Global Self-Attention
        x = concatenated
        for layer in self.sa_layers:
            x = layer(x)
        
        # Reshape back to feature map
        global_feature = x.transpose(1, 2).reshape(B, C, H, W)  # (B, C, H, W)
        
        return global_feature
