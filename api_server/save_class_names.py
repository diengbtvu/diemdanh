"""
Script để lưu class names từ dataset vào file
Chạy script này MỘT LẦN để tạo file class_names.txt
"""

import os
import json
import sys

# Thêm path để import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import FaceDataset
import torchvision.transforms as transforms

# Path đến dataset
DATASET_DIR = "../aligned_faces"  # Thay đổi nếu cần

# Kiểm tra dataset có tồn tại không
if not os.path.exists(DATASET_DIR):
    print(f"[ERROR] Dataset not found at: {DATASET_DIR}")
    print("\nPlease update DATASET_DIR variable to point to your dataset folder")
    print("Or run this script from the machine that has the dataset")
    exit(1)

print(f"[INFO] Loading dataset from: {DATASET_DIR}")

# Tạo dummy transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load dataset
try:
    dataset = FaceDataset(DATASET_DIR, transform=transform)
    
    # Lấy class names
    class_names = dataset.classes
    
    print(f"\n[INFO] Found {len(class_names)} classes")
    print(f"[INFO] First 10 classes: {class_names[:10]}")
    
    # Save to text file
    output_file = "class_names.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    print(f"\n[SUCCESS] Saved to: {output_file}")
    
    # Save to JSON
    output_json = "class_names.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump({
            "total_classes": len(class_names),
            "class_names": class_names
        }, f, indent=2, ensure_ascii=False)
    
    print(f"[SUCCESS] Saved to: {output_json}")
    
    print("\n[DONE] You can now use these files in API servers!")
    
except Exception as e:
    print(f"\n[ERROR] Failed to load dataset: {e}")
    print("\nAlternative: Manually create class_names.txt with your class names")
    print("Format: One class name per line")

