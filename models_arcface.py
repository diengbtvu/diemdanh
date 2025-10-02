"""
ArcFace Model Definition
Metric learning approach cho face recognition với generalization tốt nhất
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import math


class ArcMarginProduct(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss
    Paper: https://arxiv.org/abs/1801.07698
    
    ArcFace adds angular margin to make features more discriminative
    """
    
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50, easy_margin=False):
        """
        Args:
            in_features: Size of input features (embedding dimension)
            out_features: Number of classes
            scale (s): Normalization hypersphere radius (default: 30.0)
            margin (m): Angular margin penalty (default: 0.50 radians ~= 28.6 degrees)
            easy_margin: Use easy margin or not
        """
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        
        # Weight matrix W: (out_features, in_features)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # For numerical stability
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, input, label):
        """
        Args:
            input: Features (batch_size, in_features) - embeddings
            label: Ground truth labels (batch_size,)
        
        Returns:
            Cosine scores with margin (batch_size, out_features)
        """
        # Normalize features và weights
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # Calculate sin(theta)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # Calculate cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot encode labels
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # Apply margin chỉ cho ground truth class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Scale
        output *= self.scale
        
        return output


class ArcFaceResNet50(nn.Module):
    """
    ResNet50 backbone + ArcFace head
    Tốt nhất cho face recognition với generalization
    """
    
    def __init__(self, num_classes, embedding_size=512, pretrained=True):
        """
        Args:
            num_classes: Số người trong database
            embedding_size: Dimension của embedding vector (512 or 256)
            pretrained: Sử dụng pretrained ImageNet weights
        """
        super(ArcFaceResNet50, self).__init__()
        
        # Backbone: ResNet50
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet50(weights=None)
        
        # Remove FC layer
        self.backbone.fc = nn.Identity()
        
        # Feature dimension từ ResNet50
        self.feature_dim = 2048
        
        # Bottleneck: Giảm dimension 2048 → embedding_size
        self.bottleneck = nn.Sequential(
            nn.Linear(self.feature_dim, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )
        
        # ArcFace head
        self.arcface = ArcMarginProduct(
            in_features=embedding_size,
            out_features=num_classes,
            scale=30.0,      # s parameter
            margin=0.50,     # m parameter (0.50 radians ~= 28.6 degrees)
            easy_margin=False
        )
        
        self.embedding_size = embedding_size
        
    def forward(self, x, labels=None):
        """
        Args:
            x: Input images (batch_size, 3, 224, 224)
            labels: Ground truth labels (batch_size,) - chỉ cần khi training
        
        Returns:
            - Training mode: ArcFace logits (batch_size, num_classes)
            - Inference mode: Embeddings (batch_size, embedding_size)
        """
        # Extract features từ backbone
        features = self.backbone(x)  # (batch_size, 2048)
        
        # Bottleneck to embedding
        embeddings = self.bottleneck(features)  # (batch_size, embedding_size)
        
        # Training mode: return ArcFace logits
        if self.training and labels is not None:
            arcface_logits = self.arcface(embeddings, labels)
            return arcface_logits, embeddings
        
        # Inference mode: return normalized embeddings
        else:
            return F.normalize(embeddings)


class ArcFaceVanillaCNN(nn.Module):
    """
    Lightweight CNN backbone + ArcFace head
    Nhanh hơn ResNet50 nhưng vẫn hiệu quả
    """
    
    def __init__(self, num_classes, embedding_size=256):
        super(ArcFaceVanillaCNN, self).__init__()
        
        # Feature extraction
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )
        
        # ArcFace head
        self.arcface = ArcMarginProduct(
            in_features=embedding_size,
            out_features=num_classes,
            scale=30.0,
            margin=0.50
        )
        
        self.embedding_size = embedding_size
    
    def forward(self, x, labels=None):
        # Extract features
        x = self.features(x)
        x = self.global_pool(x)
        embeddings = self.bottleneck(x)
        
        # Training mode
        if self.training and labels is not None:
            arcface_logits = self.arcface(embeddings, labels)
            return arcface_logits, embeddings
        
        # Inference mode
        else:
            return F.normalize(embeddings)


if __name__ == "__main__":
    # Test models
    print("Testing ArcFace Models...")
    print("="*60)
    
    num_classes = 294
    batch_size = 4
    
    # Test input
    x = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # Test ArcFace ResNet50
    print("\n1. ArcFace ResNet50:")
    model1 = ArcFaceResNet50(num_classes, embedding_size=512, pretrained=False)
    model1.train()
    logits, embeddings = model1(x, labels)
    print(f"   Training output - Logits shape: {logits.shape}, Embeddings: {embeddings.shape}")
    
    model1.eval()
    embeddings_inf = model1(x)
    print(f"   Inference output - Embeddings: {embeddings_inf.shape}")
    print(f"   Embeddings normalized: {torch.norm(embeddings_inf[0]):.4f} (should be ~1.0)")
    
    # Test ArcFace Vanilla CNN
    print("\n2. ArcFace Vanilla CNN:")
    model2 = ArcFaceVanillaCNN(num_classes, embedding_size=256)
    model2.train()
    logits, embeddings = model2(x, labels)
    print(f"   Training output - Logits shape: {logits.shape}, Embeddings: {embeddings.shape}")
    
    model2.eval()
    embeddings_inf = model2(x)
    print(f"   Inference output - Embeddings: {embeddings_inf.shape}")
    print(f"   Embeddings normalized: {torch.norm(embeddings_inf[0]):.4f}")
    
    # Calculate model sizes
    def get_model_size(model):
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / 1024**2
    
    print(f"\n3. Model Sizes:")
    print(f"   ArcFace ResNet50: {get_model_size(model1):.2f} MB")
    print(f"   ArcFace Vanilla CNN: {get_model_size(model2):.2f} MB")
    
    print("\n" + "="*60)
    print(" All ArcFace models working correctly!")
    print("="*60)

