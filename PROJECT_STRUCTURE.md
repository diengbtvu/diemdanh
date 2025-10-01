# 📁 Cấu trúc Project

## 🗂️ Files và chức năng

```
nhan-dang-khuon-mat/
│
├── 📄 config.py                  # Configuration và constants
│   ├── Dataset paths
│   ├── Training hyperparameters
│   ├── Early stopping settings
│   └── Device configuration
│
├── 📄 dataset.py                 # Dataset và DataLoaders
│   ├── FaceDataset class
│   ├── Data transforms (augmentation)
│   └── create_dataloaders()
│
├── 📄 models.py                  # Model Definitions
│   ├── VanillaCNN
│   ├── ResNet50Face
│   ├── AttentionCNN
│   ├── AdaBoostFaceClassifier
│   └── calculate_model_size()
│
├── 📄 train.py                   # Training Utilities
│   ├── EarlyStopping class
│   └── train_model() function
│
├── 📄 evaluate.py                # Evaluation Functions
│   ├── evaluate_model()
│   └── print_results_summary()
│
├── 📄 visualize.py               # Visualization & Analysis
│   ├── create_individual_training_plots()
│   ├── create_comparison_charts()
│   └── create_detailed_analysis()
│
├── 📄 main.py                    # Main Script (chạy file này!)
│   └── Orchestrates toàn bộ pipeline
│
├── 📄 requirements.txt           # Python dependencies
├── 📄 README.md                  # Hướng dẫn đầy đủ
├── 📄 QUICKSTART.md              # Hướng dẫn nhanh
├── 📄 PROJECT_STRUCTURE.md       # File này
│
├── 🔧 run.bat                    # Script chạy cho Windows
├── 🔧 run.sh                     # Script chạy cho Linux/Mac
│
├── 📓 Face_Detection.ipynb       # Notebook gốc (Colab)
│
├── 📂 aligned_faces/             # Dataset folder (tự tạo)
│   ├── student_001/
│   ├── student_002/
│   └── ...
│
└── 📂 face_detection_results/    # Results folder (tự động tạo)
    ├── Models (.pth, .pkl)
    ├── Reports (.md, .csv, .json)
    └── Visualizations (.png)
```

## 🔄 Pipeline Flow

```
main.py
   │
   ├─► [1] config.py
   │       └─► Setup directories, check dataset
   │
   ├─► [2] dataset.py
   │       └─► Load data, create dataloaders
   │
   ├─► [3] models.py + train.py
   │       ├─► Train Vanilla CNN
   │       ├─► Train ResNet50
   │       ├─► Train Attention CNN
   │       └─► Train AdaBoost
   │
   ├─► [4] evaluate.py
   │       └─► Evaluate all models on test set
   │
   └─► [5] visualize.py
           ├─► Create individual training plots
           ├─► Create comparison charts
           └─► Generate detailed analysis reports
```

## 🎯 Quan hệ giữa các modules

```
main.py
    ↓
config ←─────┬─── dataset
             │       ↓
             ├─── models
             │       ↓
             ├─── train ──→ evaluate
             │              ↓
             └─── visualize
```

## 📊 Data Flow

```
1. Dataset (aligned_faces/)
        ↓
2. FaceDataset class (dataset.py)
        ↓
3. Train/Val/Test DataLoaders
        ↓
4. Models (VanillaCNN, ResNet50, etc.)
        ↓
5. Training Loop (train.py)
        ↓
6. Best Model Checkpoints
        ↓
7. Evaluation (evaluate.py)
        ↓
8. Results & Metrics
        ↓
9. Visualizations (visualize.py)
        ↓
10. Reports (CSV, JSON, PNG)
```

## 🔧 Customization Points

### Thay đổi dataset path:
➡️ `config.py` → `DATASET_DIR`

### Thay đổi hyperparameters:
➡️ `config.py` → `MAX_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`

### Thay đổi early stopping:
➡️ `config.py` → `EARLY_STOP_PATIENCE`, `EARLY_STOP_MIN_DELTA`

### Thêm model mới:
1. Thêm class vào `models.py`
2. Thêm training logic vào `main.py`
3. Model tự động integrate với evaluation và visualization

### Thay đổi data augmentation:
➡️ `dataset.py` → `get_transforms()`

### Thay đổi visualization:
➡️ `visualize.py` → Customize các functions

## 📝 Code Style

- **Type hints:** Optional (không bắt buộc)
- **Docstrings:** Có cho tất cả functions chính
- **Comments:** Tiếng Việt cho dễ hiểu
- **Naming:** snake_case cho functions/variables, PascalCase cho classes
- **Imports:** Organized theo standard library → third-party → local

## 🚀 Quick Commands

```bash
# Chạy toàn bộ
python main.py

# Test dataset loading
python -c "from dataset import create_dataloaders; create_dataloaders()"

# Test models
python models.py

# Check config
python -c "from config import print_system_info; print_system_info()"
```

## 💡 Tips

1. **Luôn chạy từ main.py** - Nó orchestrate toàn bộ
2. **Kiểm tra config.py trước** - Đảm bảo paths đúng
3. **Xem logs trong terminal** - Theo dõi progress
4. **Kết quả trong face_detection_results/** - Tất cả outputs ở đây
5. **Đọc SUMMARY_REPORT.md** - Tóm tắt kết quả

---

**Happy coding! 🎓💻**

