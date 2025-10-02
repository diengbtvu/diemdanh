"""
Training Script for ArcFace Models
Sử dụng ArcFace loss cho better generalization
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config import *
from dataset import create_dataloaders
from models_arcface import ArcFaceResNet50, ArcFaceVanillaCNN
from evaluate import print_results_summary


class ArcFaceTrainer:
    """Trainer cho ArcFace models"""
    
    def __init__(self, model, train_loader, val_loader, device, results_dir, model_name="arcface"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.results_dir = results_dir
        self.model_name = model_name
        
        # Setup optimizer và scheduler
        self.optimizer = optim.SGD([
            {'params': model.backbone.parameters(), 'lr': LEARNING_RATE * 0.1},  # Backbone: LR thấp hơn
            {'params': model.bottleneck.parameters(), 'lr': LEARNING_RATE},
            {'params': model.arcface.parameters(), 'lr': LEARNING_RATE}
        ], momentum=0.9, weight_decay=5e-4)
        
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[30, 60, 90], gamma=0.1
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
    def train_epoch(self, epoch):
        """Train một epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward với labels (training mode)
            logits, embeddings = self.model(data, target)
            
            # Loss
            loss = self.criterion(logits, target)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Print progress
            if batch_idx % 20 == 0:
                print(f'  Batch [{batch_idx}/{len(self.train_loader)}] '
                      f'Loss: {loss.item():.4f} '
                      f'Acc: {100.*correct/total:.2f}%')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validation"""
        self.model.eval()
        running_loss = 0.0
        
        # Để evaluate, cần compute embeddings và compare
        all_embeddings = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Get embeddings (inference mode)
                embeddings = self.model(data)
                
                all_embeddings.append(embeddings.cpu())
                all_labels.append(target.cpu())
        
        # Concatenate
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute validation accuracy using nearest neighbor
        # Build gallery (sử dụng chính val set - trong thực tế nên dùng riêng)
        correct = 0
        total = len(all_labels)
        
        for i in range(total):
            query_emb = all_embeddings[i:i+1]
            query_label = all_labels[i]
            
            # Compare với tất cả (except chính nó)
            similarities = torch.mm(query_emb, all_embeddings.t())[0]
            similarities[i] = -1  # Exclude chính nó
            
            # Find most similar
            most_similar_idx = torch.argmax(similarities).item()
            predicted_label = all_labels[most_similar_idx]
            
            if predicted_label == query_label:
                correct += 1
        
        val_acc = 100. * correct / total
        val_loss = 0.0  # Placeholder
        
        return val_loss, val_acc
    
    def train(self, num_epochs=100, patience=50):
        """
        Full training loop
        
        Args:
            num_epochs: Maximum epochs
            patience: Early stopping patience
        """
        print(f"\n{'='*80}")
        print(f"[TRAINING] {self.model_name} with ArcFace Loss")
        print(f"{'='*80}")
        print(f"Max Epochs: {num_epochs} | Patience: {patience}")
        print(f"Device: {self.device}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        patience_counter = 0
        
        # Training history file
        history_file = os.path.join(self.results_dir, f'{self.model_name}_training_history.json')
        log_file = os.path.join(self.results_dir, f'{self.model_name}_training_log.txt')
        
        with open(log_file, 'w') as f:
            f.write(f"Training Log for {self.model_name}\n")
            f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch+1}/{num_epochs}]")
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Validate
            print("  Validating...")
            val_loss, val_acc = self.validate()
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Learning rate scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log
            log_msg = (f'Epoch [{epoch+1:3d}/{num_epochs}] | '
                      f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                      f'Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}')
            print(log_msg)
            
            with open(log_file, 'a') as f:
                f.write(log_msg + '\n')
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                patience_counter = 0
                
                # Save checkpoint
                checkpoint_path = os.path.join(self.results_dir, f'{self.model_name}_best_model.pth')
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f'  ✓ New best model! Val Acc: {val_acc:.2f}% (saved)')
                
                with open(log_file, 'a') as f:
                    f.write(f'  ✓ Best model updated!\n')
            else:
                patience_counter += 1
                print(f'  EarlyStopping: {patience_counter}/{patience} (Best: {self.best_val_acc:.2f}% @epoch {self.best_epoch})')
            
            # Save history
            if (epoch + 1) % 5 == 0:
                history = {
                    'model_name': self.model_name,
                    'current_epoch': epoch + 1,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'train_accs': self.train_accs,
                    'val_accs': self.val_accs,
                    'best_val_acc': self.best_val_acc,
                    'best_epoch': self.best_epoch
                }
                with open(history_file, 'w') as f:
                    json.dump(history, f, indent=2)
            
            # Early stopping
            if patience_counter >= patience:
                print(f'\n[EARLY STOP] Patience reached at epoch {epoch+1}')
                print(f'Best Val Acc: {self.best_val_acc:.2f}% at epoch {self.best_epoch}')
                break
        
        training_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"[COMPLETED] {self.model_name}")
        print(f"{'='*80}")
        print(f"Training Time: {training_time/60:.2f} minutes")
        print(f"Best Val Acc: {self.best_val_acc:.2f}%")
        print(f"Best Epoch: {self.best_epoch}")
        print(f"Total Epochs: {epoch + 1}")
        print(f"{'='*80}\n")
        
        return {
            'training_time': training_time,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'total_epochs': epoch + 1,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }


def main():
    """Main training function cho ArcFace"""
    
    print("\n" + "="*80)
    print("ARCFACE MODEL TRAINING")
    print("="*80)
    print("Metric Learning Approach for Better Generalization")
    print("="*80)
    
    # Setup
    print_system_info()
    setup_directories()
    
    # Load data
    print("\n[LOADING DATA]")
    train_loader, val_loader, test_loader, num_classes, full_dataset = create_dataloaders()
    
    # Train ArcFace ResNet50
    print("\n" + "="*80)
    print("[MODEL] ArcFace ResNet50")
    print("="*80)
    
    arcface_resnet = ArcFaceResNet50(
        num_classes=num_classes,
        embedding_size=512,
        pretrained=True
    )
    
    trainer = ArcFaceTrainer(
        model=arcface_resnet,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        results_dir=RESULTS_DIR,
        model_name="arcface_resnet50"
    )
    
    results = trainer.train(num_epochs=100, patience=50)
    
    print("\n[TRAINING COMPLETE]")
    print(f"Best Validation Accuracy: {results['best_val_acc']:.2f}%")
    print(f"Training Time: {results['training_time']/60:.2f} minutes")
    
    # Save class names for inference
    class_names_file = os.path.join(RESULTS_DIR, "arcface_class_names.txt")
    with open(class_names_file, 'w', encoding='utf-8') as f:
        for name in full_dataset.classes:
            f.write(f"{name}\n")
    print(f"[SAVED] Class names: {class_names_file}")
    
    print("\n" + "="*80)
    print("ARCFACE TRAINING DONE!")
    print("="*80)
    print("\nNEXT STEPS:")
    print("1. Build embedding database: python build_arcface_database.py")
    print("2. Run ArcFace API server: python api_server/server_arcface.py")
    print("="*80)


if __name__ == "__main__":
    main()

