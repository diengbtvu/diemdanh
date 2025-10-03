"""
Train CẢ YOLOv8 VÀ YOLOv12 một lúc
"""

import subprocess
import sys
import time


def train_yolo_version(version, model_size='n'):
    """
    Train một version của YOLO
    
    Args:
        version: 'v8' hoặc 'v12'
        model_size: 'n', 's', 'm', 'l'
    """
    print("\n" + "="*80)
    print(f"TRAINING YOLO{version.upper()}-{model_size.upper()}")
    print("="*80)
    
    # Đọc file train_yolo.py
    with open('train_yolo.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Backup original
    with open('train_yolo_backup.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Update version và size
    content = content.replace(
        f"YOLO_VERSION = 'v8'",
        f"YOLO_VERSION = '{version}'"
    ).replace(
        f"YOLO_VERSION = 'v12'",
        f"YOLO_VERSION = '{version}'"
    ).replace(
        f"MODEL_SIZE = 'n'",
        f"MODEL_SIZE = '{model_size}'"
    ).replace(
        f"MODEL_SIZE = 's'",
        f"MODEL_SIZE = '{model_size}'"
    )
    
    # Write updated file
    with open('train_yolo.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Run training
    print(f"[INFO] Starting YOLO{version.upper()}-{model_size.upper()} training...")
    print(f"[INFO] This will take 3-5 hours...\n")
    
    start_time = time.time()
    
    try:
        subprocess.run([sys.executable, 'train_yolo.py'], check=True)
        elapsed = (time.time() - start_time) / 60
        
        print(f"\n[SUCCESS] YOLO{version.upper()}-{model_size.upper()} completed in {elapsed:.1f} minutes")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Training failed: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n[WARNING] Training interrupted by user")
        return False
    finally:
        # Restore original file
        with open('train_yolo_backup.py', 'r', encoding='utf-8') as f:
            original_content = f.read()
        with open('train_yolo.py', 'w', encoding='utf-8') as f:
            f.write(original_content)


def main():
    """Train tất cả YOLO versions"""
    
    print("\n" + "="*80)
    print("TRAIN ALL YOLO MODELS")
    print("="*80)
    print("This will train:")
    print("  1. YOLOv12-nano")
    print("  2. YOLOv8-nano")
    print("\nTotal time: ~6-10 hours")
    print("="*80)
    
    input("\nPress Enter to continue or Ctrl+C to cancel...")
    
    results = {}
    
    # Train YOLOv12
    print("\n[STEP 1/2] Training YOLOv12...")
    results['v12'] = train_yolo_version('v12', model_size='n')
    
    if results['v12']:
        print("\n✅ YOLOv12 training completed!")
    else:
        print("\n❌ YOLOv12 training failed!")
        return
    
    # Train YOLOv8
    print("\n[STEP 2/2] Training YOLOv8...")
    results['v8'] = train_yolo_version('v8', model_size='n')
    
    if results['v8']:
        print("\n✅ YOLOv8 training completed!")
    else:
        print("\n❌ YOLOv8 training failed!")
        return
    
    # Summary
    print("\n" + "="*80)
    print("ALL TRAININGS COMPLETED!")
    print("="*80)
    print("\nModels saved in face_detection_results/:")
    print("  - yolov12_n_best.pt")
    print("  - yolov8_n_best.pt")
    print("\nYou can now use these models with API servers!")
    print("="*80)


if __name__ == "__main__":
    main()


