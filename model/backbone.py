import torch
import torch.nn as nn
from torchvision.models import resnet50, vit_b_16

def get_resnet_backbone(pretrained=True):
    """
    Get ResNet-50 backbone with projection layer
    """
    backbone = resnet50(pretrained=pretrained)
    # Remove the classification head
    backbone = nn.Sequential(*list(backbone.children())[:-2])
    # Add a projection layer to match hidden dimension
    backbone.add_module('projection', nn.Conv2d(2048, 256, kernel_size=1))
    return backbone

def get_vit_backbone(pretrained=True):
    """
    Get ViT backbone with projection layer
    """
    backbone = vit_b_16(pretrained=pretrained)
    # Remove the classification head
    backbone = nn.Sequential(*list(backbone.children())[:-1])
    # Add a projection layer to match hidden dimension
    backbone.add_module('projection', nn.Linear(768, 256))
    return backbone
