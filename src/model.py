import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ConvNeXt_Tiny_Weights

class GeMPooling(nn.Module):
    """
    Generalized Mean Pooling (GeM).
    Computes (1/|Ω| * sum(x^p))^(1/p).
    When p=1 -> Average Pooling; when p->inf -> Max Pooling.
    Learns to focus on salient motif and textile weave structures.
    """
    def __init__(self, p=3.0, eps=1e-6, learnable=True):
        super(GeMPooling, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p) if learnable else torch.tensor(p)
        self.eps = eps

    def forward(self, x):
        # x shape: (B, C, H, W)
        p = torch.clamp(self.p, min=1.0)
        x_clamped = torch.clamp(x, min=self.eps)
        # Spatial average of x^p
        pooled = F.adaptive_avg_pool2d(x_clamped ** p, (1, 1))
        return pooled ** (1.0 / p)

class MotifWeaveModel(nn.Module):
    """
    Modular Textile Embedding Architecture:
    Image -> Backbone -> Global Pooling -> Projection Head -> L2 Normalization -> Embedding
    """
    def __init__(self, 
                 backbone_name='convnext_tiny', 
                 pretrained=True, 
                 embedding_dim=256, 
                 pooling_type='gem', 
                 gem_p=3.0,
                 head_type='mlp',
                 freeze_backbone=False,
                 dropout=0.1):
        super(MotifWeaveModel, self).__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.pooling_type = pooling_type
        self.freeze_backbone_mode = freeze_backbone
        
        # 1. Backbone Initialization
        if backbone_name == 'convnext_tiny':
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            base_model = models.convnext_tiny(weights=weights)
            self.features = base_model.features
            in_features = 768
        elif backbone_name == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_model = models.resnet50(weights=weights)
            self.features = nn.Sequential(*list(base_model.children())[:-2])
            in_features = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        # 2. Configurable Freezing for Small-Data Transfer Learning
        if freeze_backbone is True or freeze_backbone == 'full':
            for param in self.features.parameters():
                param.requires_grad = False
        elif freeze_backbone == 'early_stages':
            # Freeze stem, stage 1, stage 2, downsamples; keep stage 3 & 4 trainable
            for i in range(len(self.features) - 3):
                for param in self.features[i].parameters():
                    param.requires_grad = False
        elif freeze_backbone == 'stage4_only':
            # Freeze everything except the final stage 4 block
            for i in range(len(self.features) - 1):
                for param in self.features[i].parameters():
                    param.requires_grad = False
                    
        # 3. Global Pooling Layer
        if pooling_type == 'gem':
            self.pool = GeMPooling(p=gem_p, learnable=True)
        elif pooling_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise ValueError(f"Unsupported pooling: {pooling_type}")
            
        # 4. Projection Head
        if head_type == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, embedding_dim)
            )
        elif head_type == 'linear':
            self.head = nn.Linear(in_features, embedding_dim)
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")
            
    def extract_features(self, x):
        """
        Extract unnormalized pooled visual features prior to projection.
        """
        if self.freeze_backbone_mode in [True, 'full']:
            with torch.no_grad():
                feat_map = self.features(x)
        else:
            feat_map = self.features(x)
            
        pooled = self.pool(feat_map)
        return torch.flatten(pooled, 1)

    def forward(self, x, return_unnormalized=False):
        """
        Forward pass producing normalized design embedding.
        """
        pooled_feat = self.extract_features(x)
        raw_embed = self.head(pooled_feat)
        norm_embed = F.normalize(raw_embed, p=2, dim=1)
        
        if return_unnormalized:
            return norm_embed, raw_embed
        return norm_embed

def build_model(config):
    """
    Factory method to construct MotifWeaveModel from dictionary/config.
    """
    return MotifWeaveModel(
        backbone_name=config.get('backbone', 'convnext_tiny'),
        pretrained=config.get('pretrained', True),
        embedding_dim=config.get('embedding_dim', 256),
        pooling_type=config.get('pooling', 'gem'),
        gem_p=config.get('gem_p', 3.0),
        head_type=config.get('head_type', 'mlp'),
        freeze_backbone=config.get('freeze_backbone', False),
        dropout=config.get('dropout', 0.1)
    )
