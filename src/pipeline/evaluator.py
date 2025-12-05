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
    def __init__(self):
        """Initialize the evaluator."""
        self.logger = get_logger()
    pass
    def evaluate_predictions(self, y_true, y_pred):
        """
        Calculate comprehensive classification metrics.

        Computes Accuracy, Precision, Recall, and F1-Score based on the
        provided ground truth and predictions.

        Args:
            y_true (array-like): Ground truth (correct) labels.
            y_pred (array-like): Predicted labels, as returned by a classifier.

        Returns:
            dict: A dictionary containing the calculated metrics.
        """
        self.logger.substep("Calculating classification metrics")

        # Calculate specific metrics
        # Note: POSITIVE_CLASS_LABEL (usually 1 for Spam) ensures we track the correct target
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=POSITIVE_CLASS_LABEL, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=POSITIVE_CLASS_LABEL, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=POSITIVE_CLASS_LABEL, zero_division=0)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        # Log the results nicely
        with self.logger.indent():
            self.logger.info(f"Accuracy:  {accuracy:.4f}")
            self.logger.info(f"Precision: {precision:.4f}")
            self.logger.info(f"Recall:    {recall:.4f}")
            self.logger.info(f"F1 Score:  {f1:.4f}")

        return metrics

