"""
Spam ML Package - Restructured

This package has been restructured into two main components:

1. Pipeline: Core machine learning pipeline components
    - DataProcessor: Text data loading, cleaning, and stratified split
    - TextPreprocessor: Text preprocessing and vectorization (e.g., TF-IDF)
    - ModelTrainer: Classification model training and comparison (LogReg, NB, SVC)
    - Evaluator: Core classification model evaluation (Accuracy, Precision, Recall, F1, AUC)

2. Utils: Utilities and configuration
    - config: Configuration constants and settings (e.g., file names, model types, parameters)
    - logger: Professional logging system
    - utils: General utility functions (e.g., path validation, plotting helpers)
    - evaluation_utils: Detailed evaluation functions with visualizations (e.g., Confusion Matrix, ROC Curve)

Usage:
    from pipeline import DataProcessor, TextPreprocessor, ModelTrainer, Evaluator
    from utils.config import *
    from utils.evaluation_utils import evaluate_model_detailed
"""

__version__ = "2.0.0"
__author__ = "Spam ML Workshop - Restructured"

# Import main pipeline classes for backward compatibility
from .pipeline import DataProcessor, TextPreprocessor, ModelTrainer, Evaluator

__all__ = [
    'DataProcessor',
    'TextPreprocessor', 
    'ModelTrainer',
    'Evaluator'
]
