"""
Feature engineering/Text preprocessing module for the Spam Detection pipeline.

This module handles text preprocessing and feature extraction (vectorization).
"""

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from utils.config import TARGET_COL, MESSAGE_COL, VECTORIZER_TYPE, MAX_FEATURES
from utils.logger import get_logger, LogLevel


class TextPreprocessor:
    """
    Feature engineer for spam detection.
    
    Handles text cleaning and vectorization (BoW/TF-IDF).
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.vectorizer = None
        self.selected_features = None
    
    @staticmethod
    def preprocess_message(message):
        """
        Clean a text message: lowercase and replace numbers.
        
        Args:
            message (str): The original text message.
            
        Returns:
            str: The preprocessed message.
        """
        # Lowercasing
        message = message.lower()
        
        # Replace numbers with a unique token '<NUM>' (according to spam.ipynb)
        message = re.sub(r'\d+', '<num>', message)
        
        # Normalize whitespace and remove leading/trailing spaces
        message = re.sub(r'\s+', ' ', message).strip()
        
        return message

    def fit_vectorizer(self, train_messages):
        """
        Fit the vectorizer (Count or TF-IDF) to the training messages.
        
        Args:
            train_messages (pd.Series): The training messages.
        """
        logger = get_logger()
        logger.substep(f"Fitting {VECTORIZER_TYPE} on training data")
        
        # Choose the vectorizer
        if VECTORIZER_TYPE == "CountVectorizer":
            VectorizerClass = CountVectorizer
        elif VECTORIZER_TYPE == "TfidfVectorizer":
            VectorizerClass = TfidfVectorizer
        else:
            raise ValueError(f"Unknown vectorizer type: {VECTORIZER_TYPE}")
        
        self.vectorizer = VectorizerClass(
            preprocessor=self.preprocess_message,
            max_features=MAX_FEATURES
            # We could add stop_words='english' if needed
        )
        
        # Fit the vectorizer
        self.vectorizer.fit(train_messages)
        self.selected_features = self.vectorizer.get_feature_names_out()
        
        logger.info(f"Vocabulary size: {len(self.selected_features)}")
        logger.success("Vectorizer fit completed")
        
        # Log to MLflow
        try:
            import mlflow
            if mlflow.active_run():
                mlflow.log_param("vectorizer_type", VECTORIZER_TYPE)
                mlflow.log_param("max_features", MAX_FEATURES)
                mlflow.log_param("vocab_size", len(self.selected_features))
        except ImportError:
            pass
        

    def transform_messages(self, messages):
        """
        Transform a series of messages into a numerical feature matrix.
        
        Args:
            messages (pd.Series): The messages to transform.
            
        Returns:
            np.array: The feature matrix (sparse matrix).
        """
        if self.vectorizer is None:
            raise RuntimeError("Vectorizer must be fitted first. Call fit_vectorizer()")
            
        logger = get_logger()
        logger.substep("Transforming messages to feature matrix")
        
        X = self.vectorizer.transform(messages)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.success("Transformation completed")
        
        return X

    def fit_transform(self, train_messages):
        """Combine fit and transform for training data."""
        self.fit_vectorizer(train_messages)
        return self.transform_messages(train_messages)

    def load_and_transform_data(self, train_df, test_df):
        """
        Orchestrate the loading and transformation of train/test data.
        
        Args:
            train_df: Training DataFrame with 'message' and 'label'.
            test_df: Test DataFrame with 'message' and 'label'.
            
        Returns:
            X_train, y_train, X_test, y_test
        """
        from utils.config import MESSAGE_COL, TARGET_COL
        
        # Fit on training data
        X_train = self.fit_transform(train_df[MESSAGE_COL])
        y_train = train_df[TARGET_COL].apply(lambda x: 1 if x == 'spam' else 0).values # Ensure numeric
        
        # Transform test data
        X_test = self.transform_messages(test_df[MESSAGE_COL])
        y_test = test_df[TARGET_COL].apply(lambda x: 1 if x == 'spam' else 0).values
        
        return X_train, y_train, X_test, y_test
