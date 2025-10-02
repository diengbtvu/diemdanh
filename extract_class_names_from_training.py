"""
Extract class names từ training session
Chạy script này để lấy class names từ khi train
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import DATASET_DIR
from dataset import FaceDataset
import torchvision.transforms as transforms

print("="*60)
print("EXTRACTING CLASS NAMES FROM TRAINING DATASET")
print("="*60)

# Kiểm tra dataset
if not os.path.exists(DATASET_DIR):
    print(f"\n[ERROR] Dataset not found at: {DATASET_DIR}")
    print("\nYour dataset path (from config.py):")
    print(f"  DATASET_DIR = '{DATASET_DIR}'")
    print("\nPlease:")
    print("1. Update DATASET_DIR in config.py")
    print("2. Or run this on the machine that has the dataset")
    exit(1)

print(f"\n[INFO] Loading dataset from: {DATASET_DIR}")

# Create simple transform
transform = transforms.Compose([transforms.ToTensor()])

# Load dataset (same way as training)
try:
    full_dataset = FaceDataset(DATASET_DIR, transform=transform, max_samples_per_class=None)
    
    # Get class names (EXACTLY as used during training)
    class_names = full_dataset.classes
    
    print(f"\n[SUCCESS] Found {len(class_names)} classes")
    print(f"\nFirst 10 classes:")
    for i, name in enumerate(class_names[:10]):
        print(f"  {i}: {name}")
    
    print(f"\nLast 5 classes:")
    for i, name in enumerate(class_names[-5:], start=len(class_names)-5):
        print(f"  {i}: {name}")
    
    # Save to api_server/class_names.txt
    output_file = os.path.join("api_server", "class_names.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    print(f"\n{'='*60}")
    print(f"[SUCCESS] Saved {len(class_names)} class names to:")
    print(f"  {output_file}")
    print(f"{'='*60}")
    
    print("\n✅ NOW YOU CAN START API SERVERS!")
    print("   cd api_server")
    print("   python server_attention_cnn.py")
    
except Exception as e:
    print(f"\n[ERROR] Failed to load dataset: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n" + "="*60)
    print("ALTERNATIVE SOLUTION:")
    print("="*60)
    print("\nIf you don't have access to the dataset, manually create:")
    print("  api_server/class_names.txt")
    print("\nFormat: One folder name per line, sorted alphabetically")
    print("\nExample:")
    print("  Nguyen_Van_A")
    print("  Tran_Thi_B")
    print("  ...")

