"""
Evaluation module for assessing model performance
Calculates metrics and generates evaluation reports
"""

import os
import json
import yaml
import torch
import numpy as np
from utils import calculate_metrics, print_evaluation_report


def evaluate_model(config_path='config.yaml'):
    """
    Evaluate the trained model on test data.
    
    Args:
        config_path: Path to configuration file
    """
    print("\n" + "="*70)
    print("📊 MODEL EVALUATION")
    print("="*70)
    
    # Load config
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print("⚠️  Config file not found, using defaults")
        config = {}
    
    # For demonstration, create sample predictions and ground truth
    # In production, you would load actual test data
    print("\n⚠️  Note: Using sample data for demonstration")
    print("In production, replace with your actual test dataset")
    
    # Sample ground truth and predictions
    np.random.seed(42)
    n_samples = 100
    n_classes = 5
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    # Add some errors
    errors_idx = np.random.choice(n_samples, size=20, replace=False)
    y_pred[errors_idx] = np.random.randint(0, n_classes, 20)
    
    y_prob = np.random.rand(n_samples, n_classes)
    # Normalize to sum to 1
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    
    # Calculate metrics
    print("\n📈 Calculating metrics...")
    metrics = calculate_metrics(y_true, y_pred, y_prob)
    
    # Class names
    class_names = ['Normal', 'Pneumonia', 'COVID-19', 'Tuberculosis', 'Other']
    
    # Print detailed report
    print_evaluation_report(y_true, y_pred, class_names)
    
    # Save results
    os.makedirs('./results', exist_ok=True)
    results = {
        'metrics': metrics,
        'class_names': class_names,
        'num_samples': int(n_samples),
        'num_classes': int(n_classes)
    }
    
    output_file = './results/evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Evaluation results saved to: {output_file}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    """
    Direct execution for evaluation.
    """
    import sys
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    evaluate_model(config_file)
