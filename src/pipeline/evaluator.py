"""
Model evaluation module for the Spam Detection pipeline (Classification).

This module provides evaluation functionalities for classification models.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV, KFold

from utils.config import RANDOM_STATE, POSITIVE_CLASS_LABEL
from utils.logger import get_logger, LogLevel


class Evaluator:
    """
    Evaluator for spam detection models.
    
    Handles the calculation of classification metrics and hyperparameter optimization.
    """
    pass
