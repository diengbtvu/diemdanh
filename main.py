"""
Main Script - Face Detection Model Comparison
Script chính để chạy toàn bộ pipeline so sánh các models
Chạy một lần duy nhất để train và evaluate tất cả models

Usage:
    python main.py
"""

import os
import pickle
import torch
import warnings
warnings.filterwarnings('ignore')

# Import từ các module
from config import *
from dataset import create_dataloaders
from models import VanillaCNN, ResNet50Face, AttentionCNN, AdaBoostFaceClassifier, calculate_model_size
from train import train_model
from evaluate import evaluate_model, print_results_summary
from visualize import create_individual_training_plots, create_comparison_charts, create_detailed_analysis


def main():
    """Main function to run complete pipeline"""
    
    print("\n" + "="*80)
    print("FACE DETECTION MODEL COMPARISON PIPELINE")
    print("="*80)
    
    # ===== 1. SETUP =====
    print("\n[STEP 1/7] SYSTEM SETUP")
    print("-"*80)
    print_system_info()
    
    if not setup_directories():
        print("\n[ERROR] Dataset not found. Please check your dataset path.")
        print(f"Expected path: {DATASET_DIR}")
        return
    
    # ===== 2. LOAD DATA =====
    print("\n[STEP 2/7] LOADING DATASET")
    print("-"*80)
    
    try:
        train_loader, val_loader, test_loader, num_classes, full_dataset = create_dataloaders()
    except Exception as e:
        print(f"\n[ERROR] Failed to load dataset: {e}")
        return
    
    # Dataset info for analysis
    dataset_info = {
        'dataset_path': DATASET_DIR,
        'num_classes': num_classes,
        'train_samples': len(train_loader.dataset),
        'val_samples': len(val_loader.dataset),
        'test_samples': len(test_loader.dataset),
        'device': 'GPU' if DEVICE.type == 'cuda' else 'CPU'
    }
    
    # Dictionary to store all results
    all_results = {}
    training_curves = {}
    
    print(f"\n[INFO] Total classes: {num_classes}")
    print(f"[INFO] Train samples: {dataset_info['train_samples']}")
    print(f"[INFO] Val samples: {dataset_info['val_samples']}")
    print(f"[INFO] Test samples: {dataset_info['test_samples']}")
    
    # ===== 3. TRAIN VANILLA CNN =====
    print("\n" + "="*80)
    print("[STEP 3/7] MODEL 1/4: VANILLA CNN")
    print("="*80)
    
    vanilla_cnn = VanillaCNN(num_classes)
    vanilla_results = train_model(
        model=vanilla_cnn,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=MAX_EPOCHS,
        model_name="vanilla_cnn",
        results_dir=RESULTS_DIR,
        patience=EARLY_STOP_PATIENCE
    )
    
    print("\n[EVAL] Evaluating Vanilla CNN on test set...")
    vanilla_eval = evaluate_model(vanilla_results['model'], test_loader, "vanilla_cnn")
    
    # Store results
    all_results['Vanilla CNN'] = {
        'training_time': vanilla_results['training_time'],
        'best_val_accuracy': vanilla_results['best_val_acc'],
        'test_accuracy': vanilla_eval['accuracy'],
        'test_precision': vanilla_eval['precision'],
        'test_recall': vanilla_eval['recall'],
        'test_f1': vanilla_eval['f1_score'],
        'inference_time': vanilla_eval['inference_time'],
        'model_size_mb': calculate_model_size(vanilla_results['model']),
        'total_epochs': vanilla_results['total_epochs'],
        'early_stopped': vanilla_results['early_stopped'],
        'best_epoch': vanilla_results['best_epoch']
    }
    
    training_curves['Vanilla CNN'] = {
        'train_losses': vanilla_results['train_losses'],
        'val_losses': vanilla_results['val_losses'],
        'train_accuracies': vanilla_results['train_accuracies'],
        'val_accuracies': vanilla_results['val_accuracies'],
        'learning_rates': vanilla_results['learning_rates'],
        'best_epoch': vanilla_results['best_epoch']
    }
    
    print(f"\n[RESULTS] VANILLA CNN:")
    print(f"  Test Accuracy: {vanilla_eval['accuracy']:.2f}%")
    print(f"  Training Time: {vanilla_results['training_time']/60:.2f} minutes")
    print(f"  Model Size: {calculate_model_size(vanilla_results['model']):.2f} MB")
    
    # Clear GPU memory
    del vanilla_cnn, vanilla_results
    torch.cuda.empty_cache()
    
    # ===== 4. TRAIN RESNET50 =====
    print("\n" + "="*80)
    print("[STEP 4/7] MODEL 2/4: RESNET50 (Transfer Learning)")
    print("="*80)
    
    resnet_model = ResNet50Face(num_classes)
    resnet_results = train_model(
        model=resnet_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=MAX_EPOCHS,
        model_name="resnet50",
        results_dir=RESULTS_DIR,
        patience=EARLY_STOP_PATIENCE
    )
    
    print("\n[EVAL] Evaluating ResNet50 on test set...")
    resnet_eval = evaluate_model(resnet_results['model'], test_loader, "resnet50")
    
    # Store results
    all_results['ResNet50'] = {
        'training_time': resnet_results['training_time'],
        'best_val_accuracy': resnet_results['best_val_acc'],
        'test_accuracy': resnet_eval['accuracy'],
        'test_precision': resnet_eval['precision'],
        'test_recall': resnet_eval['recall'],
        'test_f1': resnet_eval['f1_score'],
        'inference_time': resnet_eval['inference_time'],
        'model_size_mb': calculate_model_size(resnet_results['model']),
        'total_epochs': resnet_results['total_epochs'],
        'early_stopped': resnet_results['early_stopped'],
        'best_epoch': resnet_results['best_epoch']
    }
    
    training_curves['ResNet50'] = {
        'train_losses': resnet_results['train_losses'],
        'val_losses': resnet_results['val_losses'],
        'train_accuracies': resnet_results['train_accuracies'],
        'val_accuracies': resnet_results['val_accuracies'],
        'learning_rates': resnet_results['learning_rates'],
        'best_epoch': resnet_results['best_epoch']
    }
    
    print(f"\n[RESULTS] RESNET50:")
    print(f"  Test Accuracy: {resnet_eval['accuracy']:.2f}%")
    print(f"  Training Time: {resnet_results['training_time']/60:.2f} minutes")
    print(f"  Model Size: {calculate_model_size(resnet_results['model']):.2f} MB")
    
    # Clear GPU memory
    del resnet_model, resnet_results
    torch.cuda.empty_cache()
    
    # ===== 5. TRAIN ATTENTION CNN =====
    print("\n" + "="*80)
    print("[STEP 5/7] MODEL 3/4: ATTENTION CNN")
    print("="*80)
    
    attention_cnn = AttentionCNN(num_classes)
    attention_results = train_model(
        model=attention_cnn,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=MAX_EPOCHS,
        model_name="attention_cnn",
        results_dir=RESULTS_DIR,
        patience=EARLY_STOP_PATIENCE
    )
    
    print("\n[EVAL] Evaluating Attention CNN on test set...")
    attention_eval = evaluate_model(attention_results['model'], test_loader, "attention_cnn")
    
    # Store results
    all_results['Attention CNN'] = {
        'training_time': attention_results['training_time'],
        'best_val_accuracy': attention_results['best_val_acc'],
        'test_accuracy': attention_eval['accuracy'],
        'test_precision': attention_eval['precision'],
        'test_recall': attention_eval['recall'],
        'test_f1': attention_eval['f1_score'],
        'inference_time': attention_eval['inference_time'],
        'model_size_mb': calculate_model_size(attention_results['model']),
        'total_epochs': attention_results['total_epochs'],
        'early_stopped': attention_results['early_stopped'],
        'best_epoch': attention_results['best_epoch']
    }
    
    training_curves['Attention CNN'] = {
        'train_losses': attention_results['train_losses'],
        'val_losses': attention_results['val_losses'],
        'train_accuracies': attention_results['train_accuracies'],
        'val_accuracies': attention_results['val_accuracies'],
        'learning_rates': attention_results['learning_rates'],
        'best_epoch': attention_results['best_epoch']
    }
    
    print(f"\n[RESULTS] ATTENTION CNN:")
    print(f"  Test Accuracy: {attention_eval['accuracy']:.2f}%")
    print(f"  Training Time: {attention_results['training_time']/60:.2f} minutes")
    print(f"  Model Size: {calculate_model_size(attention_results['model']):.2f} MB")
    
    # Clear GPU memory
    del attention_cnn, attention_results
    torch.cuda.empty_cache()
    
    # ===== 6. TRAIN ADABOOST =====
    print("\n" + "="*80)
    print("[STEP 6/7] MODEL 4/4: ADABOOST (Classical ML)")
    print("="*80)
    
    adaboost_model = AdaBoostFaceClassifier(n_estimators=ADABOOST_N_ESTIMATORS)
    ada_training_time = adaboost_model.fit(train_loader, val_loader, 
                                           max_samples=ADABOOST_MAX_SAMPLES)
    
    print("\n[EVAL] Evaluating AdaBoost on test set...")
    ada_predictions, ada_targets, ada_inference_time = adaboost_model.predict(test_loader)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    ada_accuracy = accuracy_score(ada_targets, ada_predictions)
    ada_precision, ada_recall, ada_f1, _ = precision_recall_fscore_support(
        ada_targets, ada_predictions, average='weighted', zero_division=0
    )
    
    # Save AdaBoost model
    adaboost_path = os.path.join(RESULTS_DIR, 'adaboost_best_model.pkl')
    with open(adaboost_path, 'wb') as f:
        pickle.dump(adaboost_model, f)
    
    adaboost_size_mb = os.path.getsize(adaboost_path) / (1024 * 1024)
    
    # Store results
    all_results['AdaBoost'] = {
        'training_time': ada_training_time,
        'best_val_accuracy': 0,
        'test_accuracy': ada_accuracy * 100,
        'test_precision': ada_precision * 100,
        'test_recall': ada_recall * 100,
        'test_f1': ada_f1 * 100,
        'inference_time': ada_inference_time,
        'model_size_mb': adaboost_size_mb,
        'total_epochs': ADABOOST_N_ESTIMATORS,
        'early_stopped': False,
        'best_epoch': ADABOOST_N_ESTIMATORS
    }
    
    print(f"\n[RESULTS] ADABOOST:")
    print(f"  Test Accuracy: {ada_accuracy * 100:.2f}%")
    print(f"  Training Time: {ada_training_time/60:.2f} minutes")
    print(f"  Model Size: {adaboost_size_mb:.2f} MB")
    
    # ===== 7. GENERATE ANALYSIS & VISUALIZATIONS =====
    print("\n" + "="*80)
    print("[STEP 7/7] GENERATING ANALYSIS & VISUALIZATIONS")
    print("="*80)
    
    # Print results summary
    print("\n[1] Results Summary:")
    print("-"*80)
    print_results_summary(all_results)
    
    # Create individual training plots
    print("\n[2] Creating Individual Training Plots...")
    print("-"*80)
    create_individual_training_plots(training_curves, RESULTS_DIR)
    
    # Create comparison charts
    print("\n[3] Creating Model Comparison Charts...")
    print("-"*80)
    chart_path = create_comparison_charts(all_results, training_curves, RESULTS_DIR)
    
    # Create detailed analysis report
    print("\n[4] Generating Detailed Analysis Report...")
    print("-"*80)
    analysis_results = create_detailed_analysis(all_results, training_curves, 
                                               RESULTS_DIR, dataset_info)
    
    # ===== FINAL SUMMARY =====
    print(f"\n{'='*80}")
    print("[COMPLETED] EXPERIMENT COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"\n[FILES] All results saved to: {RESULTS_DIR}")
    print(f"\n[REPORTS]")
    print(f"  - Summary: {analysis_results['summary_path']}")
    print(f"  - CSV: {analysis_results['csv_path']}")
    print(f"  - JSON: {analysis_results['json_path']}")
    
    # Print recommendations
    best_models = analysis_results['best_models']
    print(f"\n{'='*80}")
    print("[RECOMMENDATIONS] FOR STUDENT ATTENDANCE SYSTEM")
    print(f"{'='*80}")
    
    print(f"\n[1] FOR HIGHEST ACCURACY:")
    print(f"   Model: {best_models['accuracy']}")
    print(f"   Accuracy: {all_results[best_models['accuracy']]['test_accuracy']:.2f}%")
    print(f"   Use case: Khi cần độ chính xác cao nhất")
    
    print(f"\n[2] FOR REAL-TIME APPLICATIONS:")
    print(f"   Model: {best_models['speed']}")
    print(f"   Inference Time: {all_results[best_models['speed']]['inference_time']:.3f}s")
    print(f"   Use case: Điểm danh thời gian thực")
    
    print(f"\n[3] FOR MOBILE/EDGE DEPLOYMENT:")
    print(f"   Model: {best_models['efficiency']}")
    print(f"   Model Size: {all_results[best_models['efficiency']]['model_size_mb']:.2f} MB")
    print(f"   Use case: Chạy trên thiết bị di động")
    
    print(f"\n[4] FOR BALANCED PERFORMANCE:")
    print(f"   Model: {best_models['balanced']}")
    print(f"   Use case: Cân bằng accuracy, speed và size")
    
    print(f"\n{'='*80}")
    print("ALL DONE! Ready for research paper writing!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

