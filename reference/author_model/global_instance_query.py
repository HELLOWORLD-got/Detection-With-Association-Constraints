import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GlobalInstanceQuery(nn.Module):
    def __init__(self, hidden_dim=256, num_queries=25, num_classes=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        # Learnable instance queries
        self.instance_queries = nn.Parameter(torch.randn(num_queries, hidden_dim))
        
        # Position encoding
        self.position_encoding = self._create_position_encoding(hidden_dim)
        
        # Attention layer for instance feature extraction
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Feed-forward network for category prediction
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, num_classes + 1)  # +1 for no-object class
        )
    
    def _create_position_encoding(self, d_model, max_h=100, max_w=100):
        pe = torch.zeros(max_h, max_w, d_model)
        position_h = torch.arange(0, max_h, dtype=torch.float32).unsqueeze(1)
        position_w = torch.arange(0, max_w, dtype=torch.float32).unsqueeze(0)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, :, 0::2] = torch.sin(position_h * div_term)
        pe[:, :, 1::2] = torch.cos(position_h * div_term)
        pe[:, :, 0::2] += torch.sin(position_w * div_term)
        pe[:, :, 1::2] += torch.cos(position_w * div_term)
        
        return pe.unsqueeze(0)  # (1, H, W, C)
    
    def forward(self, global_feature):
        """
        global_feature: (B, C, H, W)
        """
        B, C, H, W = global_feature.shape
        
        # Add position encoding
        pos_enc = self.position_encoding[:, :H, :W, :].to(global_feature.device)
        pos_enc = pos_enc.permute(0, 3, 1, 2)  # (1, C, H, W)
        global_feature = global_feature + pos_enc
        
        # Reshape for attention
        global_feature_flat = global_feature.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Expand instance queries to batch size
        queries = self.instance_queries.unsqueeze(0).repeat(B, 1, 1)  # (B, N, C)
        
        # Extract instance features using attention
        instance_features, _ = self.attention(
            queries,
            global_feature_flat,
            global_feature_flat
        )  # (B, N, C)
        
        # Predict categories
        class_logits = self.classification_head(instance_features)  # (B, N, C+1)
        class_probs = F.softmax(class_logits, dim=-1)
        
        return instance_features, class_probs
