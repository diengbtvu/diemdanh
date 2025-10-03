"""
Training YOLOv8 Classification Model
Chuyển từ Colab sang Local - Chống overfitting mạnh
"""

import os
import random
import shutil
from pathlib import Path
import torch
import numpy as np
from ultralytics import YOLO

# Import config
from config import DATASET_DIR, RESULTS_DIR, RANDOM_SEED

# ===== CONFIGURATION =====
# Paths
YOLO_DATA_DIR = "yolo_dataset"  # Thư mục tạm cho YOLO format
YOLO_RESULTS_DIR = os.path.join(RESULTS_DIR, "yolo_training")

# YOLO Training Parameters - CHỐNG OVERFITTING MẠNH
# Chọn model version: YOLOv8 hoặc YOLOv12
YOLO_VERSION = 'v12'  # 'v8' hoặc 'v12'
MODEL_SIZE = 'n'      # 'n' (nano), 's' (small), 'm' (medium), 'l' (large)

# Auto select model weights
if YOLO_VERSION == 'v12':
    MODEL_WEIGHTS = f'yolo12{MODEL_SIZE}-cls.pt'  # YOLOv12
else:
    MODEL_WEIGHTS = f'yolov8{MODEL_SIZE}-cls.pt'   # YOLOv8

EPOCHS = 300
IMGSZ = 160
BATCH = 32
PATIENCE = 50           # Early stopping patience
LR0 = 1e-4             # Learning rate rất thấp
MIN_LR = 1e-6          # Minimum learning rate
DROPOUT = 0.5          # Dropout cao
WEIGHT_DECAY = 1e-3    # Weight decay cao

# Set seeds for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# Disable verbose
os.environ['YOLO_VERBOSE'] = 'False'
os.environ['WANDB_DISABLED'] = 'true'


def prepare_yolo_dataset(source_dir, output_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    """
    Chuẩn bị dataset theo format YOLO Classification
    
    Args:
        source_dir: Thư mục dataset gốc (aligned_faces)
        output_dir: Thư mục output (train/val/test splits)
        train_ratio: Tỷ lệ train
        val_ratio: Tỷ lệ validation
        test_ratio: Tỷ lệ test
    """
    print("=" * 80)
    print("PREPARING YOLO DATASET")
    print("=" * 80)
    
    # Import splitfolders
    try:
        import splitfolders
    except ImportError:
        print("[ERROR] Please install splitfolders:")
        print("  pip install split-folders")
        return False
    
    # Xóa thư mục cũ nếu tồn tại
    if os.path.exists(output_dir):
        print(f"[INFO] Removing old dataset: {output_dir}")
        shutil.rmtree(output_dir)
    
    print(f"[INFO] Creating dataset split (train/val/test): {train_ratio}/{val_ratio}/{test_ratio}")
    
    # Split dataset
    splitfolders.ratio(
        source_dir,
        output=output_dir,
        seed=RANDOM_SEED,
        ratio=(train_ratio, val_ratio, test_ratio),
        group_prefix=None,
        move=False
    )
    
    # Thống kê dataset
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_dir, split)
        if os.path.exists(split_dir):
            classes = [p for p in os.listdir(split_dir) 
                      if os.path.isdir(os.path.join(split_dir, p))]
            total_imgs = sum(len(os.listdir(os.path.join(split_dir, cls))) 
                           for cls in classes)
            print(f"{split.upper()}: {len(classes)} classes, {total_imgs:,} images")
    
    print("=" * 80)
    return True


def train_yolo_model():
    """Train YOLO Classification model với chống overfitting mạnh"""
    
    print("\n" + "=" * 80)
    print(f"YOLO {YOLO_VERSION.upper()} CLASSIFICATION TRAINING")
    print("=" * 80)
    print("CONFIGURATION - ANTI-OVERFITTING MODE")
    print("=" * 80)
    print(f"YOLO Version: {YOLO_VERSION.upper()}")
    print(f"Model: {MODEL_WEIGHTS}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image Size: {IMGSZ}")
    print(f"Batch Size: {BATCH}")
    print(f"Learning Rate: {LR0} (very low)")
    print(f"Dropout: {DROPOUT} (high)")
    print(f"Weight Decay: {WEIGHT_DECAY} (high)")
    print(f"Patience: {PATIENCE}")
    print(f"Data Split: 60/20/20")
    print("=" * 80)
    
    # Check dataset exists
    if not os.path.exists(DATASET_DIR):
        print(f"\n[ERROR] Dataset not found: {DATASET_DIR}")
        print("Please ensure dataset is in the correct location")
        return None
    
    # Prepare YOLO dataset
    if not prepare_yolo_dataset(DATASET_DIR, YOLO_DATA_DIR):
        return None
    
    # Initialize model
    print(f"\n[INFO] Initializing YOLO model: {MODEL_WEIGHTS}")
    model = YOLO(MODEL_WEIGHTS)
    
    # Create results directory
    os.makedirs(YOLO_RESULTS_DIR, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print("This will take approximately 3-5 hours with GPU")
    print("Training logs will be saved to:", YOLO_RESULTS_DIR)
    print("=" * 80 + "\n")
    
    try:
        results = model.train(
            data=YOLO_DATA_DIR,
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            patience=PATIENCE,
            
            # Learning rate settings (QUAN TRỌNG)
            lr0=LR0,                    # Learning rate thấp
            lrf=0.01,                   # Final LR = lr0 * lrf
            momentum=0.937,
            weight_decay=WEIGHT_DECAY,   # Regularization mạnh
            dropout=DROPOUT,             # Dropout cao
            
            # Data augmentation MẠNH HƠN
            degrees=20.0,        # Xoay ±20 độ
            translate=0.2,       # Dịch chuyển 20%
            scale=0.8,           # Scale ±80%
            shear=10.0,          # Shear ±10 độ
            perspective=0.0005,  # Perspective transform
            fliplr=0.5,          # Flip ngang 50%
            flipud=0.1,          # Flip dọc 10%
            mosaic=0.0,          # Không dùng mosaic (cho classification)
            mixup=0.3,           # Mixup cao 30%
            copy_paste=0.0,      # Không copy-paste
            erasing=0.5,         # Random erasing 50%
            crop_fraction=1.0,
            
            # Regularization thêm
            hsv_h=0.02,          # Hue variation
            hsv_s=0.8,           # Saturation
            hsv_v=0.5,           # Value/brightness
            
            # Optimizer
            optimizer='AdamW',    # AdamW cho regularization tốt
            cos_lr=True,         # Cosine learning rate decay
            
            # Training settings
            project=YOLO_RESULTS_DIR,
            name='yolo_face_classification',
            seed=RANDOM_SEED,
            pretrained=True,
            plots=True,
            save=True,
            save_period=20,      # Lưu mỗi 20 epochs
            
            # Validation
            val=True,
            fraction=1.0,
            
            # Performance
            device=0 if torch.cuda.is_available() else 'cpu',
            workers=0,           # 0 cho Windows
            verbose=True,
        )
        
        # Tìm model tốt nhất
        run_dir = Path(YOLO_RESULTS_DIR) / 'yolo_face_classification'
        best_weights = run_dir / 'weights' / 'best.pt'
        
        if best_weights.exists():
            print("\n" + "=" * 80)
            print("[SUCCESS] TRAINING COMPLETED!")
            print("=" * 80)
            print(f"Best model saved: {best_weights}")
            
            # Test model
            print("\n[INFO] Loading best model for evaluation...")
            test_model = YOLO(str(best_weights))
            
            # Evaluate trên test set
            print("\n[INFO] Evaluating on test set...")
            test_results = test_model.val(data=f"{YOLO_DATA_DIR}/test", split='test')
            
            print("\n" + "=" * 80)
            print("FINAL RESULTS")
            print("=" * 80)
            print(f"Test Top-1 Accuracy: {test_results.top1:.4f} ({test_results.top1*100:.2f}%)")
            print(f"Test Top-5 Accuracy: {test_results.top5:.4f} ({test_results.top5*100:.2f}%)")
            
            # Kiểm tra overfitting
            print("\n[INFO] Checking for overfitting...")
            train_results = test_model.val(data=f"{YOLO_DATA_DIR}/train", split='train')
            val_results = test_model.val(data=f"{YOLO_DATA_DIR}/val", split='val')
            
            train_acc = train_results.top1
            val_acc = val_results.top1
            test_acc = test_results.top1
            
            print("\n" + "=" * 80)
            print("OVERFITTING ANALYSIS")
            print("=" * 80)
            print(f"Train Accuracy: {train_acc*100:.2f}%")
            print(f"Val Accuracy:   {val_acc*100:.2f}%")
            print(f"Test Accuracy:  {test_acc*100:.2f}%")
            print(f"\nTrain-Val Gap:  {(train_acc - val_acc)*100:.2f}%")
            print(f"Train-Test Gap: {(train_acc - test_acc)*100:.2f}%")
            
            if (train_acc - val_acc) > 0.1 or (train_acc - test_acc) > 0.1:
                print("\n[WARNING] Có dấu hiệu overfitting!")
                print("Khuyến nghị:")
                print("  - Tăng dropout lên 0.7")
                print("  - Tăng weight_decay lên 5e-3")
                print("  - Giảm learning rate xuống 5e-5")
            else:
                print("\n[SUCCESS] Model generalize tốt!")
            
            print("=" * 80)
            
            # Copy best model to results directory
            final_model_name = f"yolo{YOLO_VERSION}_{MODEL_SIZE}_best.pt"
            final_model_path = os.path.join(RESULTS_DIR, final_model_name)
            shutil.copy(str(best_weights), final_model_path)
            print(f"\n[SAVED] Best model copied to: {final_model_path}")
            
            # Save results summary
            summary_file = os.path.join(RESULTS_DIR, f"yolo{YOLO_VERSION}_{MODEL_SIZE}_training_summary.txt")
            with open(summary_file, 'w') as f:
                f.write(f"YOLO {YOLO_VERSION.upper()} CLASSIFICATION TRAINING RESULTS\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"YOLO Version: {YOLO_VERSION.upper()}\n")
                f.write(f"Model: {MODEL_WEIGHTS}\n")
                f.write(f"Image Size: {IMGSZ}\n")
                f.write(f"Epochs: {EPOCHS}\n")
                f.write(f"Patience: {PATIENCE}\n\n")
                f.write("RESULTS:\n")
                f.write(f"  Train Accuracy: {train_acc*100:.2f}%\n")
                f.write(f"  Val Accuracy:   {val_acc*100:.2f}%\n")
                f.write(f"  Test Accuracy:  {test_acc*100:.2f}%\n\n")
                f.write("OVERFITTING:\n")
                f.write(f"  Train-Val Gap:  {(train_acc - val_acc)*100:.2f}%\n")
                f.write(f"  Train-Test Gap: {(train_acc - test_acc)*100:.2f}%\n")
            
            print(f"[SAVED] Summary: {summary_file}")
            
            return {
                'model_path': str(best_weights),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'test_acc': test_acc
            }
        else:
            print(f"\n[ERROR] Best weights not found at: {best_weights}")
            return None
            
    except KeyboardInterrupt:
        print("\n[WARNING] Training interrupted by user")
        return None
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function"""
    
    print("\n" + "=" * 80)
    print(f"YOLO {YOLO_VERSION.upper()} FACE CLASSIFICATION TRAINING")
    print("=" * 80)
    print("Anti-Overfitting Configuration")
    print("=" * 80)
    print(f"Selected: YOLO{YOLO_VERSION.upper()}-{MODEL_SIZE.upper()}")
    print("=" * 80)
    
    # Check YOLO installed
    try:
        from ultralytics import YOLO
        print("[OK] Ultralytics YOLO installed")
    except ImportError:
        print("[ERROR] Ultralytics not installed")
        print("Please install: pip install ultralytics")
        return
    
    # Check splitfolders installed
    try:
        import splitfolders
        print("[OK] splitfolders installed")
    except ImportError:
        print("[ERROR] splitfolders not installed")
        print("Please install: pip install split-folders")
        return
    
    # Train model
    results = train_yolo_model()
    
    if results:
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\nModel: {results['model_path']}")
        print(f"Test Accuracy: {results['test_acc']*100:.2f}%")
        print("\n" + "=" * 80)
        print("NEXT STEPS:")
        print("=" * 80)
        print("1. Model saved in:", RESULTS_DIR)
        print("2. Use this model with server_yolo.py")
        print("3. Update MODEL_PATH in api_server/server_yolo.py")
        print("=" * 80)
    else:
        print("\n[ERROR] Training did not complete successfully")


if __name__ == "__main__":
    main()

