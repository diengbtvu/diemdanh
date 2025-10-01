"""
Visualization and Analysis Functions
Functions cho visualization và analysis kết quả
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from config import RESULTS_DIR


def create_individual_training_plots(training_curves, results_dir):
    """Tạo biểu đồ training riêng cho từng model (chi tiết cho nghiên cứu khoa học)"""
    
    cnn_models = [m for m in training_curves.keys() if m != 'AdaBoost']
    
    for model_name in cnn_models:
        if model_name not in training_curves:
            continue
            
        curves = training_curves[model_name]
        epochs = range(1, len(curves['train_accuracies']) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{model_name} - Detailed Training Analysis', 
                     fontsize=16, fontweight='bold')
        
        # 1. Accuracy Curves
        ax = axes[0, 0]
        ax.plot(epochs, curves['train_accuracies'], 'b-', linewidth=2, 
                label='Train Accuracy', marker='o', markersize=3, 
                markevery=max(1, len(epochs)//20))
        ax.plot(epochs, curves['val_accuracies'], 'r-', linewidth=2, 
                label='Validation Accuracy', marker='s', markersize=3, 
                markevery=max(1, len(epochs)//20))
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Training vs Validation Accuracy', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Đánh dấu best epoch
        if 'best_epoch' in curves:
            best_epoch = curves['best_epoch']
            if best_epoch <= len(epochs):
                ax.axvline(x=best_epoch, color='green', linestyle='--', 
                          linewidth=2, alpha=0.7, label=f'Best Epoch: {best_epoch}')
                ax.legend(loc='lower right', fontsize=10)
        
        # 2. Loss Curves
        ax = axes[0, 1]
        ax.plot(epochs, curves['train_losses'], 'b-', linewidth=2, 
                label='Train Loss', marker='o', markersize=3, 
                markevery=max(1, len(epochs)//20))
        if 'val_losses' in curves:
            ax.plot(epochs, curves['val_losses'], 'r-', linewidth=2, 
                    label='Validation Loss', marker='s', markersize=3, 
                    markevery=max(1, len(epochs)//20))
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training vs Validation Loss', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 3. Learning Rate Schedule
        ax = axes[1, 0]
        if 'learning_rates' in curves:
            ax.plot(epochs, curves['learning_rates'], 'g-', linewidth=2, 
                   marker='o', markersize=3, markevery=max(1, len(epochs)//20))
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Learning Rate', fontsize=12)
            ax.set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, linestyle='--')
        else:
            ax.text(0.5, 0.5, 'Learning Rate data not available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        # 4. Overfitting Analysis (Train-Val Gap)
        ax = axes[1, 1]
        gap = np.array(curves['train_accuracies']) - np.array(curves['val_accuracies'])
        ax.plot(epochs, gap, 'purple', linewidth=2, marker='o', 
               markersize=3, markevery=max(1, len(epochs)//20))
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.fill_between(epochs, 0, gap, where=(gap >= 0), 
                        alpha=0.3, color='red', label='Overfitting')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy Gap (Train - Val) %', fontsize=12)
        ax.set_title('Overfitting Analysis', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        save_path = os.path.join(results_dir, 
                                f'{model_name.lower().replace(" ", "_")}_training_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[SAVED] {save_path}")


def create_comparison_charts(all_results, training_curves, results_dir):
    """Tạo biểu đồ so sánh tổng quan giữa các models"""

    plt.style.use('default')
    fig = plt.figure(figsize=(20, 15))

    models = list(all_results.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    # 1. Test Accuracy Comparison
    plt.subplot(3, 3, 1)
    accuracies = [all_results[model]['test_accuracy'] for model in models]
    bars = plt.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    plt.title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.ylim([min(accuracies) - 5, 100])
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.2f}%', ha='center', va='bottom', 
                 fontweight='bold', fontsize=9)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # 2. Training Time Comparison
    plt.subplot(3, 3, 2)
    training_times = [all_results[model]['training_time']/60 for model in models]
    bars = plt.bar(models, training_times, color=colors, edgecolor='black', linewidth=1.5)
    plt.title('Training Time Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Time (minutes)', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}m', ha='center', va='bottom', 
                 fontweight='bold', fontsize=9)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # 3. Model Size Comparison
    plt.subplot(3, 3, 3)
    model_sizes = [all_results[model]['model_size_mb'] for model in models]
    bars = plt.bar(models, model_sizes, color=colors, edgecolor='black', linewidth=1.5)
    plt.title('Model Size Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Size (MB)', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}MB', ha='center', va='bottom', 
                 fontweight='bold', fontsize=9)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # 4. Precision-Recall-F1 Comparison
    plt.subplot(3, 3, 4)
    metrics = ['test_precision', 'test_recall', 'test_f1']
    metric_labels = ['Precision', 'Recall', 'F1-Score']
    x = np.arange(len(models))
    width = 0.25

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [all_results[model][metric] for model in models]
        plt.bar(x + i*width, values, width, label=label, edgecolor='black', linewidth=1)

    plt.title('Precision, Recall, F1-Score Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Score (%)', fontsize=11)
    plt.xlabel('Models', fontsize=11)
    plt.xticks(x + width, models, rotation=45, ha='right')
    plt.legend(fontsize=10)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.ylim([0, 105])

    # 5. Inference Time Comparison
    plt.subplot(3, 3, 5)
    inference_times = [all_results[model]['inference_time'] for model in models]
    bars = plt.bar(models, inference_times, color=colors, edgecolor='black', linewidth=1.5)
    plt.title('Inference Time Comparison', fontsize=14, fontweight='bold')
    plt.ylabel('Time (seconds)', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.3f}s', ha='center', va='bottom', 
                 fontweight='bold', fontsize=9)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # 6. Total Epochs Trained
    plt.subplot(3, 3, 6)
    epochs_trained = [all_results[model].get('total_epochs', 0) for model in models]
    bars = plt.bar(models, epochs_trained, color=colors, edgecolor='black', linewidth=1.5)
    plt.title('Total Epochs Trained', fontsize=14, fontweight='bold')
    plt.ylabel('Epochs', fontsize=11)
    plt.xticks(rotation=45, ha='right')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{int(height)}', ha='center', va='bottom', 
                     fontweight='bold', fontsize=9)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # 7-9. Training Curves for CNN models
    cnn_models = ['Vanilla CNN', 'ResNet50', 'Attention CNN']
    for idx, model_name in enumerate(cnn_models):
        plt.subplot(3, 3, 7 + idx)
        if model_name in training_curves:
            epochs = range(1, len(training_curves[model_name]['train_accuracies']) + 1)
            plt.plot(epochs, training_curves[model_name]['train_accuracies'], 
                    'b-', linewidth=2, label='Train', alpha=0.8)
            plt.plot(epochs, training_curves[model_name]['val_accuracies'], 
                    'r-', linewidth=2, label='Validation', alpha=0.8)
            
            # Mark best epoch
            if 'best_epoch' in training_curves[model_name]:
                best_epoch = training_curves[model_name]['best_epoch']
                if best_epoch <= len(epochs):
                    plt.axvline(x=best_epoch, color='green', 
                               linestyle='--', linewidth=1.5, alpha=0.7)
            
            plt.title(f'{model_name} - Learning Curves', fontsize=12, fontweight='bold')
            plt.xlabel('Epoch', fontsize=10)
            plt.ylabel('Accuracy (%)', fontsize=10)
            plt.legend(fontsize=9, loc='lower right')
            plt.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    chart_path = os.path.join(results_dir, 'model_comparison_summary.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {chart_path}")

    return chart_path


def create_detailed_analysis(all_results, training_curves, results_dir, dataset_info):
    """Create detailed analysis report"""

    # Create results dataframe
    results_df = pd.DataFrame(all_results).T
    results_df = results_df.round(2)

    # Save results to CSV
    csv_path = os.path.join(results_dir, 'model_comparison_results.csv')
    results_df.to_csv(csv_path)

    # Generate recommendations
    best_accuracy_model = max(all_results.keys(), 
                             key=lambda x: all_results[x]['test_accuracy'])
    best_speed_model = min(all_results.keys(), 
                          key=lambda x: all_results[x]['inference_time'])
    most_efficient_model = min(all_results.keys(), 
                              key=lambda x: all_results[x]['model_size_mb'])

    # Calculate balanced score
    balanced_scores = {}
    for model in all_results.keys():
        acc = all_results[model]['test_accuracy']
        size = all_results[model]['model_size_mb']
        time = all_results[model]['inference_time']
        balanced_scores[model] = acc / (size + time * 100)

    best_balanced = max(balanced_scores.keys(), key=lambda x: balanced_scores[x])

    # Save complete analysis as JSON
    analysis_report = {
        'experiment_info': {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_path': dataset_info.get('dataset_path', ''),
            'num_classes': dataset_info.get('num_classes', 0),
            'train_samples': dataset_info.get('train_samples', 0),
            'val_samples': dataset_info.get('val_samples', 0),
            'test_samples': dataset_info.get('test_samples', 0),
            'device': dataset_info.get('device', 'CPU')
        },
        'model_results': all_results,
        'best_models': {
            'accuracy': best_accuracy_model,
            'speed': best_speed_model,
            'efficiency': most_efficient_model,
            'balanced': best_balanced
        }
    }

    json_path = os.path.join(results_dir, 'complete_analysis.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        json.dump(convert_numpy(analysis_report), f, indent=2, ensure_ascii=False)

    # Create summary report
    summary_report = f"""# FACE DETECTION MODEL COMPARISON REPORT
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## EXPERIMENT SETUP
- Dataset: {dataset_info.get('dataset_path', 'N/A')}
- Number of Classes: {dataset_info.get('num_classes', 'N/A')}
- Training Samples: {dataset_info.get('train_samples', 'N/A')}
- Validation Samples: {dataset_info.get('val_samples', 'N/A')}
- Test Samples: {dataset_info.get('test_samples', 'N/A')}
- Device: {dataset_info.get('device', 'N/A')}

## MODELS COMPARED
1. Vanilla CNN - Custom lightweight CNN architecture
2. ResNet50 - Pre-trained ResNet50 with transfer learning
3. Attention CNN - CNN with attention mechanism
4. AdaBoost - Classical machine learning with hand-crafted features

## RESULTS SUMMARY

| Model | Test Accuracy | Training Time | Inference Time | Model Size | F1-Score |
|-------|---------------|---------------|----------------|------------|----------|
"""

    for model_name, results in all_results.items():
        summary_report += (f"| {model_name} | {results['test_accuracy']:.2f}% | "
                          f"{results['training_time']/60:.1f}m | "
                          f"{results['inference_time']:.3f}s | "
                          f"{results['model_size_mb']:.1f}MB | "
                          f"{results['test_f1']:.2f}% |\n")

    summary_report += f"""

## KEY FINDINGS
- Best Accuracy: {best_accuracy_model} ({all_results[best_accuracy_model]['test_accuracy']:.2f}%)
- Fastest Inference: {best_speed_model} ({all_results[best_speed_model]['inference_time']:.3f}s)
- Most Compact: {most_efficient_model} ({all_results[most_efficient_model]['model_size_mb']:.1f}MB)
- Best Balanced: {best_balanced}

## RECOMMENDATIONS
1. For highest accuracy: Use {best_accuracy_model}
2. For real-time applications: Use {best_speed_model}
3. For mobile deployment: Use {most_efficient_model}
4. For balanced performance: Use {best_balanced}

Generated by Face Detection Comparison Pipeline
"""

    # Save summary report
    summary_path = os.path.join(results_dir, 'SUMMARY_REPORT.md')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_report)

    return {
        'summary_path': summary_path,
        'csv_path': csv_path,
        'json_path': json_path,
        'best_models': {
            'accuracy': best_accuracy_model,
            'speed': best_speed_model,
            'efficiency': most_efficient_model,
            'balanced': best_balanced
        }
    }

