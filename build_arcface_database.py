"""
Build Embedding Database for ArcFace
Extract embeddings từ dataset để làm database cho inference
"""

import os
import torch
import numpy as np
import pickle
from tqdm import tqdm

from config import *
from dataset import create_dataloaders
from models_arcface import ArcFaceResNet50


def build_embedding_database(model, dataloader, device, class_names, save_path):
    """
    Build database of embeddings cho mỗi class
    
    Args:
        model: ArcFace model (inference mode)
        dataloader: DataLoader (có thể dùng train+val)
        device: cuda or cpu
        class_names: List of class names
        save_path: Path để save database
    
    Returns:
        database: Dict với embeddings và metadata
    """
    model.eval()
    model.to(device)
    
    print("\n" + "="*80)
    print("BUILDING ARCFACE EMBEDDING DATABASE")
    print("="*80)
    
    # Dictionary để store embeddings cho mỗi class
    class_embeddings = {i: [] for i in range(len(class_names))}
    
    print(f"Extracting embeddings from {len(dataloader.dataset)} samples...")
    
    with torch.no_grad():
        for batch_idx, (data, labels) in enumerate(tqdm(dataloader, desc="Processing")):
            data = data.to(device)
            
            # Extract embeddings
            embeddings = model(data)  # (batch_size, embedding_size)
            
            # Store theo class
            for emb, label in zip(embeddings, labels):
                class_embeddings[label.item()].append(emb.cpu().numpy())
    
    # Tính mean embedding cho mỗi class
    print("\nComputing mean embeddings for each class...")
    database = {
        'class_names': class_names,
        'num_classes': len(class_names),
        'embeddings': [],  # Mean embeddings
        'all_embeddings': class_embeddings,  # All embeddings (optional)
        'embedding_size': embeddings.shape[1]
    }
    
    for class_idx in range(len(class_names)):
        if len(class_embeddings[class_idx]) > 0:
            # Mean embedding
            mean_emb = np.mean(class_embeddings[class_idx], axis=0)
            # Normalize
            mean_emb = mean_emb / np.linalg.norm(mean_emb)
            database['embeddings'].append(mean_emb)
            
            print(f"  Class {class_idx} ({class_names[class_idx]}): "
                  f"{len(class_embeddings[class_idx])} samples")
        else:
            print(f"  WARNING: Class {class_idx} has no samples!")
            database['embeddings'].append(np.zeros(embeddings.shape[1]))
    
    database['embeddings'] = np.array(database['embeddings'])
    
    # Save database
    with open(save_path, 'wb') as f:
        pickle.dump(database, f)
    
    print(f"\n{'='*80}")
    print(f"[SUCCESS] Database saved to: {save_path}")
    print(f"{'='*80}")
    print(f"Database info:")
    print(f"  - Classes: {len(class_names)}")
    print(f"  - Embedding size: {database['embedding_size']}")
    print(f"  - Database shape: {database['embeddings'].shape}")
    print(f"{'='*80}\n")
    
    return database


def main():
    """Main function"""
    
    print_system_info()
    
    # Load data
    print("\n[LOADING DATASET]")
    train_loader, val_loader, test_loader, num_classes, full_dataset = create_dataloaders()
    
    # Load trained ArcFace model
    model_path = os.path.join(RESULTS_DIR, "arcface_resnet50_best_model.pth")
    
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model not found: {model_path}")
        print("\nPlease train ArcFace model first:")
        print("  python train_arcface.py")
        return
    
    print(f"\n[LOADING MODEL] {model_path}")
    model = ArcFaceResNet50(num_classes=num_classes, embedding_size=512, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
    model.eval()
    print("[OK] Model loaded")
    
    # Combine train + val để build database lớn hơn
    print("\n[INFO] Using train + val data for database (more samples per person)")
    from torch.utils.data import ConcatDataset, DataLoader
    combined_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Build database
    database_path = os.path.join(RESULTS_DIR, "arcface_embedding_database.pkl")
    database = build_embedding_database(
        model=model,
        dataloader=combined_loader,
        device=DEVICE,
        class_names=full_dataset.classes,
        save_path=database_path
    )
    
    # Also save class names separately for API
    api_class_names = os.path.join("api_server", "class_names.txt")
    if not os.path.exists(api_class_names):
        with open(api_class_names, 'w', encoding='utf-8') as f:
            for name in full_dataset.classes:
                f.write(f"{name}\n")
        print(f"[SAVED] Class names for API: {api_class_names}")
    
    print("\n" + "="*80)
    print("DATABASE BUILD COMPLETE!")
    print("="*80)
    print("\nYou can now:")
    print("1. Test inference: python test_arcface_inference.py")
    print("2. Run API server: python api_server/server_arcface.py")
    print("="*80)


if __name__ == "__main__":
    main()

