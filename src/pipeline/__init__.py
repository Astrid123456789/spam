"""
Core ML Pipeline Components for Spam Detection.

This package contains the essential components for the machine learning workflow:
- DataProcessor: Handles data loading and splitting.
- TextPreprocessor: Handles text preprocessing and vectorization.
- ModelTrainer: Handles model training, comparison, and persistence.
- Evaluator: Handles core metric calculation and hyperparameter optimization.
"""

# Importation des classes principales pour un accès facile
from .data_processor import DataProcessor
from .text_preprocessor import TextPreprocessor
from .model_trainer import ModelTrainer
from .evaluator import Evaluator

__all__ = [
    'DataProcessor',
    'TextPreprocessor',
    'ModelTrainer',
    'Evaluator'
]
