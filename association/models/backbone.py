from torch import nn
import torch
from torch.nn import functional as F
from torchvision.models import resnet50, vit_b_16, ResNet50_Weights, ViT_B_16_Weights


def get_resnet_backbone(pretrained=True, hidden_dim=256, weights_path=None):
    model = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained and not weights_path else None)
    if pretrained and weights_path:
        model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return nn.Sequential(*list(model.children())[:-2], nn.Conv2d(2048, hidden_dim, 1))


class ViTBackbone(nn.Module):
    def __init__(self, pretrained, hidden_dim, weights_path=None):
        super().__init__()
        self.vit = vit_b_16(
            weights=ViT_B_16_Weights.DEFAULT if pretrained and not weights_path else None
        )
        if pretrained and weights_path:
            self.vit.load_state_dict(
                torch.load(weights_path, map_location="cpu", weights_only=True)
            )
        # Classifier is not part of spatial feature extraction.
        self.vit.heads = nn.Identity()
        self.projection = nn.Conv2d(768, hidden_dim, 1)

    def forward(self, images):
        # Rectangular inputs: pad right/bottom to patch size, interpolate the
        # pretrained 2D positional grid; preserve the class-token position.
        patch = self.vit.patch_size
        h, w = images.shape[-2:]
        images = F.pad(images, (0, (-w) % patch, 0, (-h) % patch))
        feature = self.vit.conv_proj(images)
        b, c, hp, wp = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)
        tokens = torch.cat([self.vit.class_token.expand(b, -1, -1), tokens], 1)
        pe = self.vit.encoder.pos_embedding
        side = int((pe.shape[1] - 1) ** 0.5)
        spatial = pe[:, 1:].transpose(1, 2).reshape(1, c, side, side)
        spatial = (
            F.interpolate(spatial, size=(hp, wp), mode="bicubic", align_corners=False)
            .flatten(2)
            .transpose(1, 2)
        )
        pe = torch.cat([pe[:, :1], spatial], 1)
        x = self.vit.encoder.dropout(tokens + pe)
        x = self.vit.encoder.ln(self.vit.encoder.layers(x))[:, 1:]
        return self.projection(x.transpose(1, 2).reshape(b, c, hp, wp))


def get_vit_backbone(pretrained=True, hidden_dim=256, weights_path=None):
    return ViTBackbone(pretrained, hidden_dim, weights_path)
