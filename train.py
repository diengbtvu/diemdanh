"""
Training Utilities
Functions cho training models với Early Stopping và Learning Rate Scheduler
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from config import *


class EarlyStopping:
    """Early stopping để ngăn overfitting"""
    
    def __init__(self, patience=15, min_delta=0.001, verbose=True):
        """
        Args:
            patience (int): Số epochs chờ đợi trước khi dừng
            min_delta (float): Mức cải thiện tối thiểu
            verbose (bool): Print thông báo
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, val_acc, epoch):
        score = val_acc
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience} '
                      f'(Best: {self.best_score:.2f}% at epoch {self.best_epoch})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f'[IMPROVED] Validation accuracy improved: '
                      f'{self.best_score:.2f}% -> {score:.2f}%')
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        
        return self.early_stop


def train_model(model, train_loader, val_loader, num_epochs=200, 
                model_name="model", results_dir="./results", patience=15):
    """
    Train a PyTorch model with Early Stopping
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs (int): Maximum số epochs
        model_name (str): Tên model để save
        results_dir (str): Thư mục lưu kết quả
        patience (int): Early stopping patience
        
    Returns:
        dict: Training results và metrics
    """
    device = DEVICE
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ReduceLROnPlateau: giảm LR khi val acc không cải thiện
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', 
        factor=LR_SCHEDULER_FACTOR, 
        patience=LR_SCHEDULER_PATIENCE
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, 
                                   min_delta=EARLY_STOP_MIN_DELTA, 
                                   verbose=True)

    train_losses = []
    val_losses = []
    val_accuracies = []
    train_accuracies = []
    learning_rates = []
    
    best_val_acc = 0.0
    best_model_state = None

    start_time = time.time()
    
    print(f"\n{'='*80}")
    print(f"[TRAINING] {model_name}")
    print(f"{'='*80}")
    print(f"Max Epochs: {num_epochs} | Early Stopping Patience: {patience}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    for epoch in range(num_epochs):
        # ===== TRAINING PHASE =====
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()

        # ===== VALIDATION PHASE =====
        model.eval()
        correct_val = 0
        total_val = 0
        val_running_loss = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_running_loss += loss.item()
                
                _, predicted = torch.max(output.data, 1)
                total_val += target.size(0)
                correct_val += (predicted == target).sum().item()

        # Calculate metrics
        epoch_loss = running_loss / len(train_loader)
        val_loss = val_running_loss / len(val_loader)
        train_acc = 100. * correct_train / total_train
        val_acc = 100. * correct_val / total_val
        current_lr = optimizer.param_groups[0]['lr']

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        learning_rates.append(current_lr)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        # Print progress mỗi 5 epochs hoặc epoch cuối
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            print(f'Epoch [{epoch+1:3d}/{num_epochs}] | '
                  f'Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | '
                  f'LR: {current_lr:.6f}')

        # Learning rate scheduler
        scheduler.step(val_acc)
        
        # Early stopping check
        if early_stopping(val_acc, epoch + 1):
            print(f'\n[EARLY STOP] Triggered at epoch {epoch+1}')
            print(f'Best Val Accuracy: {early_stopping.best_score:.2f}% '
                  f'at epoch {early_stopping.best_epoch}')
            break

    training_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"[COMPLETED] Training {model_name} completed!")
    print(f"{'='*80}")
    print(f"Total Training Time: {training_time/60:.2f} minutes")
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Total Epochs Trained: {len(train_losses)}")
    print(f"{'='*80}\n")

    # Save best model
    best_model_path = os.path.join(results_dir, f'{model_name}_best_model.pth')
    torch.save(best_model_state, best_model_path)
    print(f"[SAVED] Best model saved to: {best_model_path}")
    
    # Load best model for final evaluation
    model.load_state_dict(best_model_state)

    return {
        'model': model,
        'training_time': training_time,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'learning_rates': learning_rates,
        'best_val_acc': best_val_acc,
        'total_epochs': len(train_losses),
        'early_stopped': early_stopping.early_stop,
        'best_epoch': early_stopping.best_epoch
    }

