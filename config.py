"""
Configuration file for Face Detection Model Comparison
Cấu hình cho so sánh các model nhận dạng khuôn mặt
"""

import os
import torch

# ===== DATASET CONFIGURATION =====
# Đường dẫn đến dataset (thay đổi theo vị trí dataset của bạn)
DATASET_DIR = "aligned_faces"  # Thư mục chứa 300 folders khuôn mặt sinh viên
RESULTS_DIR = "face_detection_results"  # Thư mục lưu kết quả

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ===== TRAINING CONFIGURATION =====
# Training hyperparameters - OPTIMIZED FOR BEST RESULTS
MAX_EPOCHS = 300  # Tăng lên 300 epochs để train kỹ hơn
BATCH_SIZE = 32  # Giảm xuống 16 hoặc 8 nếu out of memory
LEARNING_RATE = 0.0001  # Giảm LR để train stable hơn, converge tốt hơn
NUM_WORKERS = 0  # Đặt 0 cho Windows để tránh lỗi multiprocessing

# Early stopping - CHO PHÉP TRAIN LÂU HƠN
EARLY_STOP_PATIENCE = 30  # Tăng lên 30 epochs (cho phép model học lâu hơn)
EARLY_STOP_MIN_DELTA = 0.0001  # Giảm xuống 0.01% (nhạy hơn với cải thiện nhỏ)

# Learning rate scheduler - FINE-TUNE CHẬM HƠN
LR_SCHEDULER_FACTOR = 0.3  # Giảm LR xuống 30% (chậm hơn, stable hơn)
LR_SCHEDULER_PATIENCE = 10  # Tăng lên 10 epochs (patient hơn)

# Learning rate warmup
USE_WARMUP = True
WARMUP_EPOCHS = 5  # Tăng LR dần trong 5 epochs đầu

# ===== MODEL CONFIGURATION =====
# AdaBoost - TRAIN TRÊN TẤT CẢ DATA
ADABOOST_N_ESTIMATORS = 200  # Tăng lên 200 weak learners
ADABOOST_MAX_SAMPLES = None  # None = TRAIN HẾT TẤT CẢ DATA (best accuracy)

# ===== DEVICE CONFIGURATION =====
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Random seed cho reproducibility
RANDOM_SEED = 42

# ===== IMAGE PREPROCESSING =====
IMAGE_SIZE = (224, 224)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Data augmentation - CÂN BẰNG giữa augmentation và học được features
RANDOM_HORIZONTAL_FLIP_PROB = 0.5
RANDOM_ROTATION_DEGREES = 10  # Giảm về 10 (quá nhiều rotation làm khó học)
COLOR_JITTER_PARAMS = {
    'brightness': 0.2,  # Vừa phải
    'contrast': 0.2,
    'saturation': 0.2,
    'hue': 0.1
}

# Augmentation options - VỪA PHẢI để không làm khó model
USE_RANDOM_ERASING = True
RANDOM_ERASING_PROB = 0.2  # Giảm xuống 20% (vừa đủ)

# Advanced augmentation
USE_MIXUP = False  # Mixup có thể giúp nhưng phức tạp hơn
MIXUP_ALPHA = 0.2

# ===== SYSTEM INFO =====
def print_system_info():
    """In thông tin hệ thống"""
    print("=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"[GPU] Available: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"  - CUDA Version: {torch.version.cuda}")
    else:
        print("[WARNING] No GPU detected. Training will be slower on CPU.")
    
    # PyTorch version
    print(f"\n[INFO] PyTorch Version: {torch.__version__}")
    
    # Device being used
    print(f"[INFO] Using device: {DEVICE}")
    
    print("=" * 50)

# ===== DIRECTORY SETUP =====
def setup_directories():
    """Tạo các thư mục cần thiết"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"[OK] Results directory: {RESULTS_DIR}")
    
    if not os.path.exists(DATASET_DIR):
        print(f"[WARNING] Dataset directory not found: {DATASET_DIR}")
        print("Please ensure your dataset is in the correct location.")
        print("Expected structure:")
        print(f"  {DATASET_DIR}/")
        print("    ├── student_001/")
        print("    │   ├── image1.jpg")
        print("    │   ├── image2.jpg")
        print("    │   └── ...")
        print("    ├── student_002/")
        print("    └── ... (up to 300 folders)")
        return False
    else:
        # Count folders
        try:
            num_folders = len([f for f in os.listdir(DATASET_DIR) 
                             if os.path.isdir(os.path.join(DATASET_DIR, f))])
            print(f"[OK] Dataset found: {DATASET_DIR}")
            print(f"  Number of student folders: {num_folders}")
            return True
        except Exception as e:
            print(f"[ERROR] Error reading dataset: {e}")
            return False

