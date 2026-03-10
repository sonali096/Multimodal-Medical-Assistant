"""
Utility functions for data loading and evaluation metrics
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_prob: np.ndarray = None, average: str = 'weighted') -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional, for AUC-ROC)
        average: Averaging strategy for multi-class
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    # Calculate AUC-ROC if probabilities provided
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
            else:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average=average)
        except ValueError:
            metrics['auc_roc'] = 0.0
    
    return metrics


def print_evaluation_report(y_true: np.ndarray, y_pred: np.ndarray, class_names=None):
    """
    Print comprehensive evaluation report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes (optional)
    """
    print("\n" + "="*70)
    print("EVALUATION REPORT")
    print("="*70)
    
    # Basic metrics
    metrics = calculate_metrics(y_true, y_pred)
    print("\nClassification Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name.upper()}: {value:.4f}")
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # Classification Report
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("="*70 + "\n")
