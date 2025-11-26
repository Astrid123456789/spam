"""
Detailed evaluation utilities for Spam Detection ML.

This module contains evaluation functions with visualizations 
specific to classification (Confusion Matrix, ROC Curve, Precision-Recall Curve, etc.).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

from utils.config import TARGET_COL
from utils.utils import setup_plotting, save_plot, print_results_summary, get_logger
from pipeline.evaluator import Evaluator


def evaluate_model_detailed(model, X, y, model_name="Model"):
    """
    Perform detailed evaluation with visualizations for classification.
    
    Args:
        model: Trained model (should have predict and predict_proba).
        X: Feature matrix.
        y: Target variable (encoded as 0/1).
        model_name: Name for plots and reports.
        
    Returns:
        Dictionary with detailed evaluation results.
    """
    logger = get_logger()
    logger.step(f"Detailed evaluation of {model_name}", 4)

    # 1. Make predictions
    y_pred = model.predict(X)
    
    # Probabilities for curves (if the model supports it)
    y_proba = None
    if hasattr(model, 'predict_proba'):
        # Probability of the positive class (1=spam)
        y_proba = model.predict_proba(X)[:, 1] 
    else:
        logger.warning("The model does not support 'predict_proba'. ROC/PR curves will use binary predictions.")
        y_proba = y_pred

    # 2. Calculate metrics
    evaluator = Evaluator()
    # We assume y is the encoded target (0/1) and pos_label=1
    metrics = evaluator.calculate_metrics(y, y_pred, pos_label=1) 
    print_results_summary(metrics)
    
    # 3. Create visualizations
    _create_evaluation_plots(y, y_pred, y_proba, model_name)
    
    return {
        'metrics': metrics,
        'predictions': y_pred
    }

def _create_evaluation_plots(y_true, y_pred, y_proba, model_name):
    """Generate standard classification plots."""
    setup_plotting()
    logger = get_logger()
    
    # --- Plot 1: Confusion Matrix ---
    logger.substep("Generating Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=['Ham (0)', 'Spam (1)'],
        yticklabels=['Ham (0)', 'Spam (1)']
    )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix - {model_name}')
    save_plot(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    
    # --- Plot 2: ROC Curve (if probabilities available) ---
    if y_proba is not None and hasattr(y_proba, '__len__') and y_proba.ndim == 1:
        logger.substep("GGenerating ROC Curve")
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title(f'Receiver Operating Characteristic (ROC) - {model_name}')
        plt.legend(loc="lower right")
        save_plot(f'roc_curve_{model_name.lower().replace(" ", "_")}.png')

    # --- Plot 3: Precision-Recall Curve (if probabilities available) ---
    if y_proba is not None and hasattr(y_proba, '__len__') and y_proba.ndim == 1:
        logger.substep("GGenerating Precision-Recall Curve")
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        
        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, color='green', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.grid(True)
        save_plot(f'pr_curve_{model_name.lower().replace(" ", "_")}.png')

    logger.success("All plots generated successfully")
