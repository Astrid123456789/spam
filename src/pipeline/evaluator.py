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
    
    def calculate_metrics(self, y_true, y_pred, positive_label=POSITIVE_CLASS_LABEL):
        """
        Calculate classification metrics.
        Wrapper around evaluate_predictions for compatibility.
        """
        return self.evaluate_predictions(y_true, y_pred)

    def evaluate_model(self, model, X, y, cv=5, mlflow_tracking=False):
        """
        Evaluate model using Cross-Validation and log to MLflow.
        
        Args:
            model: The model to evaluate.
            X: Features.
            y: Target.
            cv: Number of folds.
            mlflow_tracking: Whether to log to MLflow.
            
        Returns:
            Dictionary of mean CV metrics.
        """
        self.logger.substep(f"Running {cv}-Fold Cross-Validation")
        
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
        
        scores = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": []
        }
        
        # Import mlflow inside method to avoid circular imports or issues if not installed
        if mlflow_tracking:
            import mlflow
        
        for fold, (train_idx, val_idx) in enumerate(cv_strategy.split(X, y)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # Clone model to avoid side effects
            from sklearn.base import clone
            fold_model = clone(model)
            fold_model.fit(X_fold_train, y_fold_train)
            
            y_fold_pred = fold_model.predict(X_fold_val)
            
            fold_metrics = self.calculate_metrics(y_fold_val, y_fold_pred)
            
            for k, v in fold_metrics.items():
                if k in scores:
                    scores[k].append(v)
                
        # Calculate means
        mean_scores = {k: np.mean(v) for k, v in scores.items()}
        
        self.logger.results_summary(mean_scores)
        
        if mlflow_tracking and mlflow.active_run():
            # Log CV metrics
            for k, v in mean_scores.items():
                mlflow.log_metric(f"cv_mean_{k}", v)
            
            # Log CV strategy (as param)
            mlflow.log_param("cv_folds", cv)
            
        return mean_scores

    def optimize_hyperparameters(self, model, X, y, param_grid, mlflow_tracking=False):
        """
        Optimize hyperparameters using GridSearchCV.
        """
        self.logger.substep("Starting Hyperparameter Optimization")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='f1', # Optimize for F1   score for spam
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        self.logger.success(f"Best parameters: {best_params}")
        self.logger.info(f"Best CV Score (F1): {best_score:.4f}")
        
        if mlflow_tracking:
            import mlflow
            if mlflow.active_run():
                mlflow.log_params(best_params)
                mlflow.log_metric("best_cv_f1", best_score)
        
        return best_model, best_params, best_score

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