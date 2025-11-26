"""
Model training module for the Spam Detection (Classification) pipeline.

This module handles training, comparison, and evaluation of classification models.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Import tracing and configuration tools
from utils.config import MODEL_TYPES, RANDOM_STATE, TARGET_COL, NB_ITERATIONS
from pipeline.evaluator import Evaluator
from utils.logger import get_logger

# MLflow configuration
try:
    import mlflow
    try:
        import mlflow.sklearn
    except Exception:
        pass
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False


class ModelTrainer:
    """
    Model trainer for spam detection.
    
    Handles training of different classification models,
    comparison, and persistence of models.
    """
    
    def __init__(self):
        """Initialize the model trainer."""
        self.trained_models = {}
        self.evaluator = Evaluator()
        self.best_model = None
        self.best_model_name = None
    
    def create_model(self, model_type, **params):
        """
        Create an instance of the specified model type (Classification).
        
        Args:
            model_type (str): Type of model to create.
            **params: Additional hyperparameters for the model.
            
        Returns:
            scikit-learn model instance.
        """
        model_type = model_type.lower()
        
        if model_type == 'logistic_regression':
            # Use LogisticRegression with max_iter as per the notebook
            model = LogisticRegression(
                random_state=RANDOM_STATE, 
                max_iter=NB_ITERATIONS,
                n_jobs=-1, 
                **params
            )
        elif model_type == 'naive_bayes':
            # MultinomialNB is suitable for word count features (BoW/TF-IDF)
            model = MultinomialNB(**params)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Available: {MODEL_TYPES}")
            
        return model

    def train_single_model(self, X, y, model_type, **model_params):
        """
        Train a single model on the entire training dataset.
        
        Args:
            X: Training feature matrix (vectorization sparse matrix).
            y: Training labels.
            model_type: Type of model to train.
            **model_params: Hyperparameters to pass to the model creation.
            
        Returns:
            Trained model.
        """
        logger = get_logger()
        logger.substep(f"Starting training for {model_type.title()} model")

        # TODO: Log with MLflow if necessary (to adapt from Workshop 4)

        # Create the model
        model = self.create_model(model_type, **model_params)

        # Train the model
        model.fit(X, y)
        
        # Store the trained model
        self.trained_models[model_type] = model

        # Calculate training score (Accuracy for classification)
        train_score = model.score(X, y)

        # Logging
        with logger.indent():
            logger.model_info(f"Training Accuracy score: {train_score:.4f}")
        
        logger.success(f"{model_type.title()} model training completed")

        return model
    
    
    def predict(self, model, X):
        """
        Make predictions using a trained model.
        
        Args:
            model: Trained model.
            X: Feature matrix.
            
        Returns:
            Predictions array.
        """
        logger = get_logger()
        predictions = model.predict(X)
        logger.success(f"Generated {len(predictions)} predictions")
        return predictions
