"""
Utility functions cho API servers
"""

import os
import json
import sys

def load_class_names(class_names_file="class_names.txt", num_classes=294):
    """
    Load class names từ file
    
    Args:
        class_names_file: Path đến file chứa class names
        num_classes: Số classes expected
    
    Returns:
        List of class names
    """
    print(f"[INFO] Loading class names from: {class_names_file}")
    
    # Try loading from txt file
    if os.path.exists(class_names_file):
        try:
            with open(class_names_file, 'r', encoding='utf-8') as f:
                class_names = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            if len(class_names) == num_classes:
                print(f"[OK] Loaded {len(class_names)} class names from {class_names_file}")
                print(f"[OK] First 5: {class_names[:5]}")
                return class_names
            else:
                print(f"[WARNING] Expected {num_classes} classes but got {len(class_names)}")
                if len(class_names) > 0:
                    print(f"[INFO] Using {len(class_names)} classes from file")
                    return class_names
        except Exception as e:
            print(f"[ERROR] Failed to load {class_names_file}: {e}")
    
    # Try loading from JSON
    json_file = class_names_file.replace('.txt', '.json')
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                class_names = data.get('class_names', [])
            
            if len(class_names) > 0:
                print(f"[OK] Loaded {len(class_names)} class names from {json_file}")
                return class_names
        except Exception as e:
            print(f"[ERROR] Failed to load {json_file}: {e}")
    
    # Fallback: Try loading from dataset
    print(f"[WARNING] Class names file not found: {class_names_file}")
    print(f"[INFO] Attempting to load from dataset...")
    
    try:
        # Add parent directory to path
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from dataset import FaceDataset
        import torchvision.transforms as transforms
        
        dataset_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "aligned_faces")
        
        if os.path.exists(dataset_dir):
            transform = transforms.Compose([transforms.ToTensor()])
            dataset = FaceDataset(dataset_dir, transform=transform)
            class_names = dataset.classes
            
            print(f"[OK] Loaded {len(class_names)} class names from dataset")
            
            # Save for future use
            try:
                with open(class_names_file, 'w', encoding='utf-8') as f:
                    for name in class_names:
                        f.write(f"{name}\n")
                print(f"[INFO] Saved class names to {class_names_file} for future use")
            except:
                pass
            
            return class_names
    except Exception as e:
        print(f"[ERROR] Failed to load from dataset: {e}")
    
    # Last resort: Generate dummy class names
    print(f"\n{'='*60}")
    print(f"[WARNING] Could not load actual class names!")
    print(f"[WARNING] Using dummy class names: class_0, class_1, ...")
    print(f"{'='*60}")
    print(f"\nTO FIX THIS:")
    print(f"1. Run: python save_class_names.py (on machine with dataset)")
    print(f"2. Or manually create {class_names_file} with your class names")
    print(f"3. Format: One class name per line\n")
    
    return [f"class_{i}" for i in range(num_classes)]


def get_class_name_from_index(class_names, index):
    """
    Get class name from index
    
    Args:
        class_names: List of class names
        index: Class index
    
    Returns:
        Class name or f"class_{index}" if index out of range
    """
    if 0 <= index < len(class_names):
        return class_names[index]
    else:
        return f"class_{index}"

