"""
Training Utilities
Functions cho training models với Early Stopping và Learning Rate Scheduler
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from config import *


def save_val_predictions(model, val_loader, device, epoch, model_name, results_dir, 
                        num_samples=16, class_names=None):
    """
    Lưu visualizations của predictions trên validation set
    
    Args:
        model: Model đang train
        val_loader: Validation data loader
        device: Device (cuda/cpu)
        epoch: Epoch hiện tại
        model_name: Tên model
        results_dir: Thư mục lưu kết quả
        num_samples: Số samples để visualize
        class_names: List tên các classes (optional)
    """
    model.eval()
    
    # Lấy một batch từ validation set
    data_iter = iter(val_loader)
    images, labels = next(data_iter)
    
    # Chỉ lấy num_samples đầu tiên
    images = images[:num_samples].to(device)
    labels = labels[:num_samples].to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Denormalize images để hiển thị
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    
    # Move to CPU
    images = images.cpu()
    labels = labels.cpu()
    predicted = predicted.cpu()
    
    # Create figure
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))
    fig.suptitle(f'{model_name} - Epoch {epoch} - Validation Predictions', fontsize=14, fontweight='bold')
    
    for idx in range(min(num_samples, rows*cols)):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        # Display image
        img = images[idx].permute(1, 2, 0).numpy()
        ax.imshow(img)
        
        # Get labels
        true_label = labels[idx].item()
        pred_label = predicted[idx].item()
        
        # Use class names if provided, otherwise use indices
        if class_names and true_label < len(class_names):
            true_name = class_names[true_label]
            pred_name = class_names[pred_label]
        else:
            true_name = f"Class {true_label}"
            pred_name = f"Class {pred_label}"
        
        # Set title color based on correctness
        is_correct = (true_label == pred_label)
        color = 'green' if is_correct else 'red'
        
        title = f"True: {true_name}\nPred: {pred_name}"
        ax.set_title(title, fontsize=8, color=color, fontweight='bold')
        ax.axis('off')
    
    # Save figure
    save_path = os.path.join(results_dir, f'{model_name}_val_predictions_epoch_{epoch}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


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
                model_name="model", results_dir="./results", patience=15, 
                class_names=None):
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
    # Thêm weight decay (L2 regularization) để giảm overfitting
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # ReduceLROnPlateau: giảm LR khi val acc không cải thiện
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', 
        factor=LR_SCHEDULER_FACTOR, 
        patience=LR_SCHEDULER_PATIENCE
    )
    
    # Warmup scheduler (tăng LR dần trong các epochs đầu)
    if USE_WARMUP:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, 
            total_iters=WARMUP_EPOCHS
        )
        print(f"[INFO] Using warmup: LR will increase from {LEARNING_RATE*0.1:.6f} to {LEARNING_RATE:.6f} in {WARMUP_EPOCHS} epochs")
    else:
        warmup_scheduler = None
    
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
    
    # Tạo file để lưu training history real-time
    history_file = os.path.join(results_dir, f'{model_name}_training_history.json')
    log_file = os.path.join(results_dir, f'{model_name}_training_log.txt')
    
    # Log header
    with open(log_file, 'w') as f:
        f.write(f"Training Log for {model_name}\n")
        f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Max Epochs: {num_epochs} | Patience: {patience}\n")
        f.write("="*80 + "\n\n")

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

        # Print progress mỗi epoch
        progress_msg = (f'Epoch [{epoch+1:3d}/{num_epochs}] | '
                       f'Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                       f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | '
                       f'LR: {current_lr:.6f}')
        
        # Print mỗi 5 epochs hoặc khi có improvement
        if (epoch + 1) % 5 == 0 or epoch == 0 or val_acc > best_val_acc or epoch == num_epochs - 1:
            print(progress_msg)
            if val_acc > best_val_acc:
                print(f'  ✓ New best model! Val Acc improved: {best_val_acc:.2f}% -> {val_acc:.2f}%')
        
        # Lưu log sau MỖI EPOCH (để theo dõi real-time)
        with open(log_file, 'a') as f:
            f.write(progress_msg + '\n')
            if val_acc > best_val_acc:
                f.write(f'  ✓ New best model saved!\n')
        
        # Lưu history JSON sau mỗi 5 epochs (không quá thường xuyên)
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            history_data = {
                'model_name': model_name,
                'current_epoch': epoch + 1,
                'total_epochs': num_epochs,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_accuracies': train_accuracies,
                'val_accuracies': val_accuracies,
                'learning_rates': learning_rates,
                'best_val_acc': best_val_acc,
                'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            print(f'  [SAVED] Training history -> {history_file}')
        
        # Lưu visualization của validation predictions sau mỗi 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0 or val_acc > best_val_acc:
            try:
                vis_path = save_val_predictions(
                    model, val_loader, device, epoch + 1, 
                    model_name, results_dir, num_samples=16,
                    class_names=class_names
                )
                print(f'  [SAVED] Validation predictions -> {vis_path}')
            except Exception as e:
                print(f'  [WARNING] Could not save validation predictions: {e}')

        # Learning rate scheduler
        # Warmup phase: tăng LR dần
        if USE_WARMUP and epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()
        # Sau warmup: giảm LR khi plateau
        else:
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
    
    # Save final training history
    final_history = {
        'model_name': model_name,
        'completed': True,
        'total_epochs': len(train_losses),
        'training_time_minutes': training_time / 60,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'learning_rates': learning_rates,
        'best_val_acc': best_val_acc,
        'best_epoch': early_stopping.best_epoch,
        'early_stopped': early_stopping.early_stop,
        'completed_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(history_file, 'w') as f:
        json.dump(final_history, f, indent=2)
    print(f"[SAVED] Final training history -> {history_file}")
    
    # Final log entry
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total time: {training_time/60:.2f} minutes\n")
        f.write(f"Best Val Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Total Epochs: {len(train_losses)}\n")
        f.write(f"{'='*80}\n")
    
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

