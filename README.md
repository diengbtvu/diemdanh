# Face Detection Model Comparison

Dự án so sánh các model Object Detection cho dataset nhận dạng khuôn mặt sinh viên.

## 📋 Mô tả

Project này so sánh 4 models khác nhau cho bài toán nhận dạng khuôn mặt:

1. **Vanilla CNN** - Custom lightweight CNN architecture
2. **ResNet50** - Pre-trained ResNet50 với transfer learning
3. **Attention CNN** - CNN với attention mechanism
4. **AdaBoost** - Classical machine learning với hand-crafted features

## 🎯 Tính năng

- ✅ Training tự động với Early Stopping
- ✅ Learning Rate Scheduler (ReduceLROnPlateau)
- ✅ Đầy đủ metrics: Accuracy, Precision, Recall, F1-Score
- ✅ Visualization chi tiết cho từng model
- ✅ So sánh tổng quan giữa các models
- ✅ Export kết quả: CSV, JSON, Markdown reports
- ✅ Lưu best model checkpoints
- ✅ Chạy một lần duy nhất - có kết quả đầy đủ

## 📦 Cài đặt

### 1. Clone hoặc tải project

```bash
cd nhan-dang-khuon-mat
```

### 2. Cài đặt Python packages

**Windows:**
```bash
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
pip3 install -r requirements.txt
```

### 3. Cài đặt PyTorch với GPU support (Optional nhưng khuyến khích)

Truy cập https://pytorch.org/get-started/locally/ và chọn cấu hình phù hợp.

Ví dụ với CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📁 Cấu trúc Dataset

Chuẩn bị dataset theo cấu trúc sau:

```
aligned_faces/
├── student_001/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── student_002/
│   ├── image1.jpg
│   └── ...
├── student_003/
│   └── ...
└── ... (tối đa 300 folders)
```

**Yêu cầu:**
- Mỗi folder là một sinh viên (một class)
- Tên folder tùy ý (nên đặt có ý nghĩa)
- Ảnh format: `.jpg`, `.jpeg`, `.png`
- Nên có ít nhất 5-10 ảnh mỗi người

## 🚀 Chạy chương trình

### Cách 1: Chạy toàn bộ pipeline (Khuyến khích)

```bash
python main.py
```

Chương trình sẽ tự động:
1. ✅ Kiểm tra hệ thống và GPU
2. ✅ Load dataset
3. ✅ Train 4 models (Vanilla CNN, ResNet50, Attention CNN, AdaBoost)
4. ✅ Evaluate trên test set
5. ✅ Tạo visualizations
6. ✅ Export reports và recommendations

**Thời gian chạy:** 
- Với GPU: ~30-60 phút (tùy dataset size)
- Với CPU: ~2-4 giờ

### Cách 2: Test models riêng lẻ

```bash
# Test model definitions
python models.py

# Test dataset loading
python -c "from dataset import create_dataloaders; create_dataloaders()"
```

## ⚙️ Cấu hình

Chỉnh sửa `config.py` để thay đổi:

```python
# Dataset path
DATASET_DIR = "aligned_faces"  # Thay đổi nếu dataset ở chỗ khác

# Training parameters
MAX_EPOCHS = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Early stopping
EARLY_STOP_PATIENCE = 15  # Dừng sau 15 epochs không cải thiện
EARLY_STOP_MIN_DELTA = 0.001  # Ngưỡng cải thiện tối thiểu 0.1%
```

## 📊 Kết quả

Sau khi chạy xong, các file kết quả được lưu trong folder `face_detection_results/`:

### Model Checkpoints
- `vanilla_cnn_best_model.pth`
- `resnet50_best_model.pth`
- `attention_cnn_best_model.pth`
- `adaboost_best_model.pkl`

### Reports
- `SUMMARY_REPORT.md` - Tổng quan kết quả (cho research paper)
- `model_comparison_results.csv` - Bảng so sánh chi tiết
- `complete_analysis.json` - Toàn bộ dữ liệu (cho phân tích sau)

### Visualizations
- `model_comparison_summary.png` - So sánh tổng quan
- `vanilla_cnn_training_analysis.png` - Chi tiết Vanilla CNN
- `resnet50_training_analysis.png` - Chi tiết ResNet50
- `attention_cnn_training_analysis.png` - Chi tiết Attention CNN

## 📈 Metrics được đo

Cho mỗi model:
- **Test Accuracy** - Độ chính xác trên test set
- **Precision, Recall, F1-Score** - Metrics chi tiết
- **Training Time** - Thời gian training
- **Inference Time** - Thời gian predict
- **Model Size** - Kích thước model (MB)
- **Training Curves** - Accuracy và Loss curves
- **Learning Rate Schedule** - Thay đổi learning rate
- **Overfitting Analysis** - Phân tích train-val gap

## 🎓 Sử dụng cho Research Paper

Chương trình tạo đầy đủ materials cho nghiên cứu khoa học:

### 1. Abstract/Introduction
- Sử dụng số liệu từ `SUMMARY_REPORT.md`
- Trích dẫn số lượng classes, samples

### 2. Methodology
- Mô tả 4 models trong report
- Hyperparameters trong `config.py`
- Early stopping strategy

### 3. Results
- Bảng so sánh từ CSV
- Biểu đồ từ PNG files
- Training curves cho mỗi model

### 4. Discussion
- Phân tích overfitting từ training analysis
- So sánh trade-offs: accuracy vs speed vs size
- Best epoch analysis

### 5. Conclusion
- Recommendations từ report
- Deployment scenarios

## 🐛 Troubleshooting

### Lỗi: Dataset not found
```
[WARNING] Dataset directory not found: aligned_faces
```
**Giải pháp:** Kiểm tra đường dẫn trong `config.py`, đảm bảo folder `aligned_faces` tồn tại.

### Lỗi: Out of memory (GPU)
```
RuntimeError: CUDA out of memory
```
**Giải pháp:** Giảm `BATCH_SIZE` trong `config.py` (thử 16 hoặc 8).

### Lỗi: Multiprocessing (Windows)
```
RuntimeError: DataLoader worker ... exited unexpectedly
```
**Giải pháp:** Đặt `NUM_WORKERS = 0` trong `config.py`.

### Lỗi: No GPU detected
```
[WARNING] No GPU detected. Training will be slower on CPU.
```
**Giải pháp:** 
- Cài đặt CUDA toolkit
- Cài PyTorch với GPU support
- Hoặc chấp nhận chạy trên CPU (chậm hơn)

## 📝 Requirements

- Python 3.8+
- GPU với CUDA support (optional nhưng khuyến khích)
- RAM: Tối thiểu 8GB, khuyến khích 16GB+
- Disk space: ~2GB cho models và results

## 🔧 Advanced Usage

### Load model đã train để inference

```python
import torch
from models import VanillaCNN

# Load model
num_classes = 294  # Thay bằng số classes của bạn
model = VanillaCNN(num_classes)
model.load_state_dict(torch.load('face_detection_results/vanilla_cnn_best_model.pth'))
model.eval()

# Inference
# ... your inference code ...
```

### Chỉ train một model cụ thể

```python
from config import *
from dataset import create_dataloaders
from models import VanillaCNN
from train import train_model

# Load data
train_loader, val_loader, test_loader, num_classes, _ = create_dataloaders()

# Train only Vanilla CNN
model = VanillaCNN(num_classes)
results = train_model(model, train_loader, val_loader, 
                     num_epochs=50, model_name="my_model")
```

## 📧 Contact & Support

Nếu có vấn đề hoặc câu hỏi:
1. Kiểm tra [Troubleshooting](#-troubleshooting)
2. Xem log files trong `face_detection_results/`
3. Đảm bảo dataset structure đúng format

## 📄 License

This project is for educational and research purposes.

## 🙏 Acknowledgments

- PyTorch team for deep learning framework
- scikit-learn for machine learning tools
- OpenCV for computer vision utilities

---

**Good luck with your research! 🎓🚀**

