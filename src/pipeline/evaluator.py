"""
Model evaluation module for the Spam Detection pipeline (Classification).

This module provides evaluation functionalities for classification models.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,confusion_matrix
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
        #POSITIVE_CLASS_LABEL (usually 1 for Spam) ensures we track the correct target
        pos_label = 1 if isinstance(POSITIVE_CLASS_LABEL, str) else POSITIVE_CLASS_LABEL
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)

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

    def generate_report(self, y_true, y_pred):
        """
        Generate a full text classification report.

        Args:
            y_true (array-like): Ground truth labels.
            y_pred (array-like): Predicted labels.

        Returns:
            str: Text summary of the precision, recall, F1 score for each class.
        """
        report = classification_report(
            y_true, 
            y_pred, 
            zero_division=0
        )
        self.logger.info("Classification Report:\n" + report)
        return report
    def get_confusion_matrix(self, y_true, y_pred):
        """
        Compute the confusion matrix.
        
        | TN | FP |
        | FN | TP |

        Args:
            y_true (array-like): Ground truth labels.
            y_pred (array-like): Predicted labels.

        Returns:
            np.ndarray: Confusion matrix.
        """
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        self.logger.info(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        return cm