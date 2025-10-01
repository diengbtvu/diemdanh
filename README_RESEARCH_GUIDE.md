# Hướng Dẫn Sử Dụng Code Cho Nghiên Cứu Khoa Học
## Hệ Thống Điểm Danh Sinh Viên Bằng Nhận Diện Khuôn Mặt

---

## 📋 MỤC LỤC

1. [Tổng Quan](#tổng-quan)
2. [Cấu Hình Training](#cấu-hình-training)
3. [Các Model Được So Sánh](#các-model-được-so-sánh)
4. [Cách Chạy Code](#cách-chạy-code)
5. [Kết Quả & Biểu Đồ](#kết-quả--biểu-đồ)
6. [Viết Cơ Sở Lý Thuyết](#viết-cơ-sở-lý-thuyết)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 TỔNG QUAN

### Mục Tiêu Nghiên Cứu
So sánh hiệu năng của 4 phương pháp nhận diện khuôn mặt khác nhau cho hệ thống điểm danh sinh viên tự động.

### Dataset
- **Số lượng sinh viên**: 294 người
- **Số ảnh/sinh viên**: ~20 ảnh
- **Tổng số ảnh**: 5,880 ảnh
- **Chia dữ liệu**: 70% Train / 15% Validation / 15% Test

### Metrics Đánh Giá
1. **Accuracy**: Độ chính xác tổng thể
2. **Precision**: Độ chính xác của dự đoán positive
3. **Recall**: Khả năng phát hiện đúng
4. **F1-Score**: Trung bình điều hòa của Precision & Recall
5. **Training Time**: Thời gian training
6. **Inference Time**: Thời gian dự đoán
7. **Model Size**: Kích thước model (MB)

---

## ⚙️ CÁC THÔNG SỐ TRAINING

### Early Stopping Configuration
```python
patience = 15  # Dừng sau 15 epochs không cải thiện
min_delta = 0.001  # Cải thiện tối thiểu 0.1%
```

### Learning Rate Schedule
```python
optimizer = Adam(lr=0.001)
scheduler = ReduceLROnPlateau(
    mode='max',        # Maximize validation accuracy
    factor=0.5,        # Giảm LR xuống 50%
    patience=7,        # Sau 7 epochs không cải thiện
    verbose=True
)
```

### Training Parameters
- **Max Epochs**: 200
- **Batch Size**: 32
- **Loss Function**: CrossEntropyLoss
- **Data Augmentation**: 
  - Random Horizontal Flip (p=0.5)
  - Random Rotation (±10°)
  - Color Jitter (brightness, contrast, saturation)

---

## 🤖 CÁC MODEL ĐƯỢC SO SÁNH

### 1. Vanilla CNN (Baseline)
```
Architecture:
- 4 Conv Blocks (32→64→128→256 channels)
- BatchNorm + ReLU + MaxPool
- 3 FC Layers (12544→512→256→294)
- Dropout 0.5

Ưu điểm: Nhẹ, đơn giản
Nhược điểm: Accuracy thấp hơn
```

### 2. ResNet50 (Transfer Learning)
```
Architecture:
- Pre-trained ResNet50 (ImageNet)
- Replace FC layer (2048→294)

Ưu điểm: Accuracy cao, convergence nhanh
Nhược điểm: Model size lớn (~90MB)
```

### 3. Attention CNN (Custom)
```
Architecture:
- 3 Conv Blocks (VGG-style)
- Attention Mechanism (spatial attention)
- 2 FC Layers (256→512→294)

Ưu điểm: Focus vào vùng quan trọng, cân bằng
Nhược điểm: Phức tạp hơn Vanilla CNN
```

### 4. AdaBoost (Classical ML)
```
Features:
- Raw pixels (subsampled 16×16)
- Histogram (16 bins)
- LBP-like features (100 patterns)

Classifier:
- 100 Decision Stumps (max_depth=1)

Ưu điểm: Không cần GPU, interpretable
Nhược điểm: Accuracy thấp, feature engineering thủ công
```

---

## 🚀 CÁCH CHẠY CODE TRÊN GOOGLE COLAB

### Bước 1: Chuẩn Bị Dataset

1. Upload dataset lên Google Drive:
```
/content/drive/MyDrive/aligned_faces/
├── student_001/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── student_002/
└── ...
```

2. Đảm bảo có ít nhất 10-20 ảnh/sinh viên

### Bước 2: Upload Notebook

1. Upload `Face_Detection.ipynb` lên Colab
2. Chọn Runtime → Change runtime type → **GPU** (T4 hoặc L4)

### Bước 3: Chạy Từng Cell

```python
# Cell 1: Install packages (chạy 1 lần)
!pip install -q torch torchvision opencv-python-headless ...

# Cell 2: Mount Drive & Import
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Check GPU
# Xác nhận có GPU NVIDIA

# Cell 5: Load Dataset
# Kiểm tra số lượng folders và samples

# Cell 7: Define Models
# Tạo 3 CNN architectures

# Cell 9: Training Utilities
# Load EarlyStopping & visualization functions

# Cell 11: Setup
# Initialize dictionaries

# Cell 12-15: Train 4 models (chạy tuần tự)
# Mỗi model có thể mất 15-60 phút

# Cell 17: Generate Reports
# Tạo tất cả biểu đồ và reports
```

### Bước 4: Theo Dõi Training

**Chú ý các thông báo:**
```
✓ Validation accuracy improved: 85.50% → 87.20%
EarlyStopping counter: 5/15 (Best: 87.20% at epoch 45)
⚠️ Early Stopping triggered at epoch 60
```

**Learning Rate giảm:**
```
Epoch 00007: reducing learning rate to 0.0005
Epoch 00014: reducing learning rate to 0.00025
```

---

## 📊 KẾT QUẢ & BIỂU ĐỒ

### Files Được Tạo Ra

#### 1. Model Weights
```
vanilla_cnn_best_model.pth      # Best checkpoint của Vanilla CNN
resnet50_best_model.pth         # Best checkpoint của ResNet50
attention_cnn_best_model.pth    # Best checkpoint của Attention CNN
adaboost_best_model.pkl         # AdaBoost model
```

#### 2. Biểu Đồ Chi Tiết Cho Từng Model
```
vanilla_cnn_training_analysis.png
├── Train vs Val Accuracy
├── Train vs Val Loss
├── Learning Rate Schedule
└── Overfitting Analysis (Train-Val Gap)

resnet50_training_analysis.png
attention_cnn_training_analysis.png
```

#### 3. Biểu Đồ So Sánh Tổng Quan
```
model_comparison_summary.png
├── Test Accuracy Comparison
├── Training Time Comparison
├── Model Size Comparison
├── Precision/Recall/F1 Comparison
├── Inference Time Comparison
├── Total Epochs Trained
└── Learning Curves (3 models)
```

#### 4. Analysis Reports
```
SUMMARY_REPORT.md              # Markdown report
model_comparison_results.csv   # Bảng kết quả
complete_analysis.json         # Full data
detailed_analysis.png          # Confusion matrix + metrics
```

---

## 📖 VIẾT CƠ SỞ LÝ THUYẾT

### 1. ABSTRACT (Tóm Tắt)

**Dữ liệu từ:** `model_comparison_results.csv`

```
Nghiên cứu này so sánh 4 phương pháp nhận diện khuôn mặt
cho hệ thống điểm danh sinh viên tự động trên dataset 
294 sinh viên (5,880 ảnh). 

Kết quả: ResNet50 đạt accuracy cao nhất (XX.XX%), 
Vanilla CNN có inference time nhanh nhất (X.XXXs),
AdaBoost có model size nhỏ nhất (XX MB).

Early stopping được áp dụng với patience=15 để tối ưu
thời gian training và tránh overfitting.
```

### 2. RELATED WORK (Nghiên Cứu Liên Quan)

**Trích dẫn:**
- VGGFace, FaceNet cho face recognition
- ResNet cho transfer learning
- Attention mechanism trong computer vision
- AdaBoost cho classical ML

### 3. METHODOLOGY (Phương Pháp)

#### 3.1 Dataset
```
- Nguồn: [Mô tả nguồn dataset]
- Preprocessing: Aligned faces, resize 224×224
- Augmentation: Horizontal flip, rotation, color jitter
- Split: 70-15-15 (Train-Val-Test)
```

#### 3.2 Model Architectures

**Mô tả chi tiết từng model:**

**Vanilla CNN:**
```python
Input (224×224×3)
→ Conv(3→32) → BN → ReLU → MaxPool  # 224→112
→ Conv(32→64) → BN → ReLU → MaxPool  # 112→56
→ Conv(64→128) → BN → ReLU → MaxPool # 56→28
→ Conv(128→256) → BN → ReLU → MaxPool # 28→14
→ AdaptiveAvgPool(7×7) → Flatten
→ FC(12544→512) → ReLU → Dropout(0.5)
→ FC(512→256) → ReLU → Dropout(0.5)
→ FC(256→294)
```

**ResNet50:**
```python
Pre-trained ResNet50 (ImageNet)
→ Replace FC layer: 2048 → 294 classes
→ Fine-tune entire network
```

**Attention CNN:**
```python
Features:
  Conv Blocks (3→64→128→256)
Attention:
  Conv(256→128→1) → Sigmoid → Spatial weights
Classifier:
  GlobalAvgPool → FC(256→512→294)
```

**AdaBoost:**
```python
Feature Extraction:
  - Pixel values: 16×16 = 256 features
  - Histogram: 16 bins
  - LBP patterns: 100 features
Classifier:
  100 Decision Stumps (max_depth=1)
```

#### 3.3 Training Configuration

```
Optimizer: Adam (lr=0.001)
Loss: CrossEntropyLoss
Batch Size: 32
Max Epochs: 200

Early Stopping:
  - Patience: 15 epochs
  - Min delta: 0.001 (0.1%)

LR Scheduler: ReduceLROnPlateau
  - Factor: 0.5
  - Patience: 7 epochs
```

#### 3.4 Evaluation Metrics

```
- Accuracy: (TP + TN) / Total
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: 2 × (Precision × Recall) / (Precision + Recall)
- Inference Time: Thời gian predict trên test set
```

### 4. RESULTS (Kết Quả)

**Sử dụng biểu đồ:**

#### Table 1: Overall Performance
*Dữ liệu từ `model_comparison_results.csv`*

| Model | Test Acc | Precision | Recall | F1 | Inference |
|-------|----------|-----------|--------|----|-----------| 
| Vanilla CNN | XX.XX% | XX.XX% | XX.XX% | XX.XX% | X.XXXs |
| ResNet50 | XX.XX% | XX.XX% | XX.XX% | XX.XX% | X.XXXs |
| Attention CNN | XX.XX% | XX.XX% | XX.XX% | XX.XX% | X.XXXs |
| AdaBoost | XX.XX% | XX.XX% | XX.XX% | XX.XX% | X.XXXs |

#### Figure 1: Training Curves
*Từ `model_comparison_summary.png`*

Mô tả:
- ResNet50 converge nhanh nhất (epoch XX)
- Vanilla CNN có overfitting từ epoch XX
- Attention CNN ổn định, learning rate decay hiệu quả

#### Figure 2: Individual Training Analysis
*Từ `resnet50_training_analysis.png`*

Phân tích:
- Train-Val gap: Đo lường overfitting
- Learning rate schedule: Giảm dần khi plateau
- Early stopping triggered tại epoch XX

#### Figure 3: Model Comparison
*Từ `model_comparison_summary.png`*

So sánh:
- ResNet50: Accuracy cao nhất nhưng inference chậm
- Vanilla CNN: Balance tốt
- AdaBoost: Thấp nhất nhưng explainable

### 5. DISCUSSION (Thảo Luận)

#### 5.1 Performance Analysis

**ResNet50 (Best Accuracy):**
- Pre-trained features từ ImageNet giúp generalize tốt
- Transfer learning hiệu quả cho face recognition
- Trade-off: Model size lớn (90MB), inference chậm

**Vanilla CNN:**
- Đơn giản, dễ deploy
- Có hiện tượng overfitting từ epoch XX
- Early stopping giúp tránh train quá lâu

**Attention CNN:**
- Spatial attention giúp focus vào facial features
- Cân bằng giữa accuracy và efficiency
- Phù hợp cho production

**AdaBoost:**
- Accuracy thấp do hand-crafted features
- Không tận dụng được spatial information
- Ưu điểm: Interpretable, không cần GPU

#### 5.2 Early Stopping Analysis

*Dữ liệu từ `complete_analysis.json`*

```
- Vanilla CNN stopped at epoch XX (patience triggered)
- ResNet50 stopped at epoch XX
- Attention CNN stopped at epoch XX

→ Tiết kiệm XX% thời gian so với train full 200 epochs
→ Best val accuracy được save, tránh overfitting
```

#### 5.3 Learning Rate Schedule

*Từ training curves*

```
All models benefit from ReduceLROnPlateau:
- LR giảm khi val accuracy plateau
- Fine-tuning tốt hơn ở late epochs
- Convergence ổn định
```

### 6. CONCLUSION (Kết Luận)

#### Recommendations:

**1. Hệ thống cần accuracy cao:**
→ Sử dụng **ResNet50**
- Accuracy: XX.XX%
- Phù hợp: Server-based system, GPU available

**2. Real-time attendance:**
→ Sử dụng **Vanilla CNN** hoặc **Attention CNN**
- Inference < X.XXs
- Phù hợp: Live camera feed, multiple students

**3. Mobile/Edge deployment:**
→ Sử dụng **Vanilla CNN**
- Model size: XX MB
- Phù hợp: Smartphone, Raspberry Pi

**4. Explainability required:**
→ Sử dụng **AdaBoost**
- Feature importance analysis
- Phù hợp: Research, debugging

#### Future Work:
1. Thử nghiệm với MobileNet, EfficientNet
2. Áp dụng Face Verification (Siamese Network)
3. Test trên dataset lớn hơn (>1000 students)
4. Real-time deployment và performance testing

---

## 🛠️ TROUBLESHOOTING

### Lỗi Thường Gặp

#### 1. Out of Memory (OOM)
```python
# Giảm batch size
batch_size = 16  # thay vì 32

# Hoặc clear cache thường xuyên
torch.cuda.empty_cache()
```

#### 2. Dataset không tìm thấy
```python
# Kiểm tra path
!ls /content/drive/MyDrive/
# Update DATASET_DIR_DRIVE nếu cần
```

#### 3. Training quá chậm
```python
# Đảm bảo đang dùng GPU
print(torch.cuda.is_available())  # Should be True
```

#### 4. Early Stopping trigger quá sớm
```python
# Tăng patience
patience = 20  # thay vì 15

# Hoặc giảm min_delta
min_delta = 0.0001  # thay vì 0.001
```

---

## 📚 REFERENCES

### Papers:
1. He et al. (2016) - Deep Residual Learning for Image Recognition
2. Vaswani et al. (2017) - Attention Is All You Need
3. Freund & Schapire (1997) - A Decision-Theoretic Generalization of On-Line Learning

### Datasets:
- ImageNet (pre-training)
- Your custom student face dataset

### Frameworks:
- PyTorch 2.8.0
- OpenCV 4.12.0
- scikit-learn

---

## 💡 TIPS CHO NGHIÊN CỨU KHOA HỌC

### 1. Reproducibility
```python
# Set seed để reproduce kết quả
torch.manual_seed(42)
np.random.seed(42)
```

### 2. Ablation Study
- Thử nghiệm với/không có data augmentation
- So sánh different patience values
- Test với different LR schedules

### 3. Statistical Significance
- Chạy multiple runs (3-5 lần)
- Báo cáo mean ± std
- Confidence intervals

### 4. Visualization
- Use high-DPI images (300 dpi)
- Clear labels and legends
- Consistent color schemes

---

## 📞 CONTACT & SUPPORT

Nếu có vấn đề:
1. Kiểm tra [Troubleshooting](#troubleshooting)
2. Review error messages
3. Check Google Colab resources (RAM/GPU usage)

**Good luck with your research! 🎓**
