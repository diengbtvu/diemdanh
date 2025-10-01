"""
Face Dataset Class và Data Loaders
Dataset class cho face recognition với 300 student folders
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

from config import *


class FaceDataset(Dataset):
    """Dataset for face recognition with student folders"""

    def __init__(self, root_dir, transform=None, max_samples_per_class=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = []
        self.class_to_idx = {}
        self.samples = []

        # Check if root directory exists
        if not os.path.exists(root_dir):
            raise ValueError(f"Dataset directory not found: {root_dir}")

        # Get all student folders
        try:
            folders = [f for f in os.listdir(root_dir) 
                      if os.path.isdir(os.path.join(root_dir, f))]
            folders.sort()
        except Exception as e:
            print(f"Error reading dataset directory: {e}")
            folders = []

        print(f"Found {len(folders)} student folders")

        for idx, folder in enumerate(folders):
            if len(self.classes) >= 300:  # Limit to 300 classes
                break

            folder_path = os.path.join(root_dir, folder)
            
            # Try to read folder with error handling
            try:
                images = [f for f in os.listdir(folder_path)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                # Skip empty folders
                if len(images) == 0:
                    print(f"Skipping empty folder: {folder}")
                    continue
                
                # Add this class
                self.classes.append(folder)
                current_idx = len(self.classes) - 1
                self.class_to_idx[folder] = current_idx

                if max_samples_per_class:
                    images = images[:max_samples_per_class]

                for img_name in images:
                    img_path = os.path.join(folder_path, img_name)
                    self.samples.append((img_path, current_idx))
                    
            except (OSError, PermissionError) as e:
                print(f"Warning: Skipping folder '{folder}' due to error: {e}")
                continue
            except Exception as e:
                print(f"Warning: Unexpected error with folder '{folder}': {e}")
                continue

        print(f"Total samples: {len(self.samples)}")
        print(f"Number of classes: {len(self.classes)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy image
            dummy_image = Image.new('RGB', IMAGE_SIZE, (0, 0, 0))
            if self.transform:
                dummy_image = self.transform(dummy_image)
            return dummy_image, label


def get_transforms():
    """Tạo data transforms cho training và testing"""
    
    # Training transforms với augmentation mạnh hơn
    train_transform_list = [
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(RANDOM_HORIZONTAL_FLIP_PROB),
        transforms.RandomRotation(RANDOM_ROTATION_DEGREES),
        transforms.ColorJitter(**COLOR_JITTER_PARAMS),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ]
    
    # Thêm Random Erasing nếu enabled (tăng robustness, giảm overfitting)
    if USE_RANDOM_ERASING:
        train_transform_list.append(
            transforms.RandomErasing(p=RANDOM_ERASING_PROB, scale=(0.02, 0.15))
        )
    
    train_transform = transforms.Compose(train_transform_list)

    test_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)
    ])
    
    return train_transform, test_transform


def create_dataloaders(dataset_dir=DATASET_DIR, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Tạo train, validation và test dataloaders
    
    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    print("=" * 50)
    print("CREATING DATASETS AND DATALOADERS")
    print("=" * 50)
    
    train_transform, test_transform = get_transforms()
    
    try:
        # Load ALL images from each student folder
        full_dataset = FaceDataset(dataset_dir, transform=train_transform, 
                                   max_samples_per_class=None)

        if len(full_dataset) == 0:
            raise ValueError("Dataset is empty. Please check your dataset directory.")

        # Split dataset
        train_size = int(TRAIN_RATIO * len(full_dataset))
        val_size = int(VAL_RATIO * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(RANDOM_SEED)
        )

        # Update test dataset transform
        test_dataset.dataset.transform = test_transform
        val_dataset.dataset.transform = test_transform

        print(f"[OK] Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                shuffle=False, num_workers=num_workers)

        num_classes = len(full_dataset.classes)
        print(f"[OK] Number of classes: {num_classes}")
        print("=" * 50)

        return train_loader, val_loader, test_loader, num_classes, full_dataset

    except Exception as e:
        print(f"\n[ERROR] Failed to create dataset: {e}")
        print("\nPLEASE CHECK:")
        print(f"1. Dataset path: {dataset_dir}")
        print("2. Dataset structure:")
        print(f"   {dataset_dir}/")
        print("     ├── student_001/")
        print("     │   ├── image1.jpg")
        print("     │   └── ...")
        print("     ├── student_002/")
        print("     └── ...")
        raise

