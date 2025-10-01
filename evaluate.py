"""
Evaluation Functions
Functions cho evaluation models
"""

import time
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config import DEVICE


def evaluate_model(model, test_loader, model_name="model"):
    """
    Evaluate a PyTorch model trên test set
    
    Args:
        model: PyTorch model
        test_loader: Test data loader
        model_name (str): Tên model
        
    Returns:
        dict: Evaluation metrics
    """
    device = DEVICE
    model = model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []

    start_time = time.time()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    inference_time = time.time() - start_time

    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='weighted', zero_division=0
    )

    return {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1_score': f1 * 100,
        'inference_time': inference_time,
        'predictions': all_predictions,
        'targets': all_targets
    }


def print_results_summary(all_results):
    """Print a formatted summary of results"""
    import pandas as pd
    
    print("\n" + "="*80)
    print("FINAL RESULTS COMPARISON")
    print("="*80)

    # Create and print results dataframe
    results_df = pd.DataFrame(all_results).T
    results_df = results_df.round(2)
    print(results_df)
    print("="*80)

