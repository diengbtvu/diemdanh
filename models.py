"""
Model Definitions
Các model definitions: Vanilla CNN, ResNet50, Attention CNN, AdaBoost
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import cv2
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# ===== 1. VANILLA CNN MODEL =====
class VanillaCNN(nn.Module):
    """Custom lightweight CNN architecture"""
    
    def __init__(self, num_classes):
        super(VanillaCNN, self).__init__()
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Second block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Third block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ===== 2. RESNET50-BASED MODEL =====
class ResNet50Face(nn.Module):
    """ResNet50 with transfer learning"""
    
    def __init__(self, num_classes):
        super(ResNet50Face, self).__init__()
        # Use pre-trained weights
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# ===== 3. ATTENTION CNN MODEL =====
class AttentionCNN(nn.Module):
    """Improved CNN with Attention Mechanism"""
    
    def __init__(self, num_classes):
        super(AttentionCNN, self).__init__()
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
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        output = self.classifier(attended_features)
        return output


# ===== 4. ADABOOST CLASSIFIER =====
class AdaBoostFaceClassifier:
    """AdaBoost classifier with hand-crafted features"""
    
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.model = None
        self.training_time = 0

    def extract_features(self, images):
        """Extract features from images for AdaBoost"""
        features = []

        for img in images:
            # Convert tensor to numpy if needed
            if torch.is_tensor(img):
                img = img.permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)

            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img

            # Resize to fixed size
            gray = cv2.resize(gray, (64, 64))

            # Extract multiple types of features
            feature_vector = []

            # 1. Raw pixel values (subsampled)
            pixel_features = gray[::4, ::4].flatten()
            feature_vector.extend(pixel_features)

            # 2. Histogram features
            hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
            feature_vector.extend(hist.flatten())

            # 3. LBP-like features (simplified)
            lbp_features = []
            for i in range(1, gray.shape[0]-1):
                for j in range(1, gray.shape[1]-1):
                    center = gray[i, j]
                    pattern = 0
                    pattern += (gray[i-1, j-1] > center) * 1
                    pattern += (gray[i-1, j] > center) * 2
                    pattern += (gray[i-1, j+1] > center) * 4
                    pattern += (gray[i, j+1] > center) * 8
                    lbp_features.append(pattern)

            # Sample LBP features
            if len(lbp_features) > 100:
                lbp_features = lbp_features[::len(lbp_features)//100][:100]
            feature_vector.extend(lbp_features)

            features.append(feature_vector)

        return np.array(features)

    def fit(self, train_loader, val_loader=None, max_samples=None):
        """
        Train AdaBoost classifier
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (not used)
            max_samples: Maximum số samples để train (None = train hết)
                        Khuyến khích: 5000-10000 cho AdaBoost (classical ML)
        """
        import time
        
        print("Extracting features for AdaBoost training...")
        
        # Mặc định: train 5000 samples cho AdaBoost
        # Nếu max_samples=None: TRAIN HẾT TẤT CẢ DATA
        if max_samples is None:
            max_samples = float('inf')  # Không giới hạn - train hết
            print(f"[INFO] AdaBoost will train on ALL available samples (best accuracy mode)")

        # Extract features from training data
        X_train, y_train = [], []
        for batch_idx, (images, labels) in enumerate(train_loader):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx}/{len(train_loader)} | Samples: {len(X_train)}/{max_samples}")

            # Denormalize images
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            images = images * std + mean

            batch_features = self.extract_features(images)
            X_train.extend(batch_features)
            y_train.extend(labels.numpy())

            # Limit data (AdaBoost classical ML không cần quá nhiều data)
            if len(X_train) >= max_samples:
                print(f"[INFO] Reached {len(X_train)} samples, stopping feature extraction")
                break

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        print(f"Training AdaBoost with {X_train.shape[0]} samples, {X_train.shape[1]} features")

        # Train AdaBoost
        self.model = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=self.n_estimators,
            random_state=42
        )

        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time

        print(f"AdaBoost training completed in {self.training_time:.2f} seconds")

        return self.training_time

    def predict(self, test_loader):
        """Make predictions on test data"""
        import time
        
        print("Extracting features for AdaBoost prediction...")

        X_test, y_test = [], []
        for batch_idx, (images, labels) in enumerate(test_loader):
            # Denormalize images
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            images = images * std + mean

            batch_features = self.extract_features(images)
            X_test.extend(batch_features)
            y_test.extend(labels.numpy())

        X_test = np.array(X_test)
        y_test = np.array(y_test)

        start_time = time.time()
        predictions = self.model.predict(X_test)
        inference_time = time.time() - start_time

        return predictions, y_test, inference_time


def calculate_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


if __name__ == "__main__":
    # Test models
    print("Testing model definitions...")
    
    num_classes = 294
    batch_size = 4
    
    # Test input
    x = torch.randn(batch_size, 3, 224, 224)
    
    # Test Vanilla CNN
    model1 = VanillaCNN(num_classes)
    out1 = model1(x)
    print(f"Vanilla CNN output shape: {out1.shape}")
    print(f"Vanilla CNN size: {calculate_model_size(model1):.2f} MB")
    
    # Test ResNet50
    model2 = ResNet50Face(num_classes)
    out2 = model2(x)
    print(f"ResNet50 output shape: {out2.shape}")
    print(f"ResNet50 size: {calculate_model_size(model2):.2f} MB")
    
    # Test Attention CNN
    model3 = AttentionCNN(num_classes)
    out3 = model3(x)
    print(f"Attention CNN output shape: {out3.shape}")
    print(f"Attention CNN size: {calculate_model_size(model3):.2f} MB")
    
    print("\nAll models working correctly!")

