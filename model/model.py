import torch
import torch.nn as nn
from .multi_view_feature_fusion import MultiViewFeatureFusion
from .global_instance_query import GlobalInstanceQuery
from .multi_view_object_query import MultiViewObjectQuery

class InstanceBasedAssociation(nn.Module):
    def __init__(self, backbone, hidden_dim=256, num_queries=25, num_classes=1):
        super().__init__()
        self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        # Multi-View Feature Fusion Module
        self.mvff = MultiViewFeatureFusion(hidden_dim=hidden_dim)
        
        # Global Instance Query Module
        self.giq = GlobalInstanceQuery(
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_classes=num_classes
        )
        
        # Multi-View Object Query Module
        self.mvoq = MultiViewObjectQuery(
            hidden_dim=hidden_dim,
            num_queries=num_queries
        )
    
    def forward(self, images):
        """
        images: list of tensors, each of shape (B, 3, H, W) for each view
        """
        # Extract single-view features using backbone
        single_view_features = []
        for img in images:
            feat = self.backbone(img)
            single_view_features.append(feat)
        
        # Multi-View Feature Fusion
        global_feature = self.mvff(single_view_features)
        
        # Global Instance Query
        instance_features, class_probs = self.giq(global_feature)
        
        # Multi-View Object Query
        boxes, confidences = self.mvoq(instance_features, single_view_features)
        
        return class_probs, boxes, confidences
