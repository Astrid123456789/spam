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


class FeatureEngineer:
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
        message = re.sub(r'\d+', '<NUM>', message)
        
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
