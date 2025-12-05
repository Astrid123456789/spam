"""
Technical validation tests for the TextPreprocessor class (Spam Detection).

These tests validate the text cleaning and vectorization functionality (TF-IDF/CountVectorizer).
"""

import pandas as pd
import numpy as np
import pytest
from scipy.sparse import issparse, csr_matrix 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from src.pipeline.text_preprocessor import TextPreprocessor
from utils.config import TARGET_COL, MESSAGE_COL, VECTORIZER_TYPE


class TestTextPreprocessor:
    """Series of tests for the technical validation of TextPreprocessor (NLP)."""
    
    def test_preprocess_message_lowercase(self):
        """Test the conversion to lowercase."""
        engineer = TextPreprocessor()
        message = "This is A Test Message With UPPERCASE Letters."
        expected = "this is a test message with uppercase letters."
        assert engineer.preprocess_message(message) == expected
        
    def test_preprocess_message_numbers_replacement(self):
        """Test the replacement of numbers with a <NUM> token."""
        engineer = TextPreprocessor()
        message = "I received 123 SMS and 45 emails in 2024."
        expected = "i received <num> sms and <num> emails in <num>."
        assert engineer.preprocess_message(message) == expected
        
    def test_preprocess_message_whitespace_handling(self):
        """Test the handling of multiple whitespace characters."""
        engineer = TextPreprocessor()
        message = "  message   with\tmultiple\nwhitespaces. "
        expected = "message with multiple whitespaces."
        assert engineer.preprocess_message(message) == expected

    def test_fit_vectorizer(self, sample_text_data):
        """Test the fitting of the vectorizer."""
        engineer = TextPreprocessor()
        _, messages, _ = sample_text_data
        
        engineer.fit_vectorizer(messages)
        
        # The vectorizer should be an instance of TfidfVectorizer or CountVectorizer
        assert engineer.vectorizer is not None
        assert isinstance(engineer.vectorizer, (TfidfVectorizer, CountVectorizer))
        
        # The vocabulary should have been created
        vocab = engineer.vectorizer.vocabulary_
        assert len(vocab) > 0, "The vocabulary should not be empty"
        
        # The preprocessed words should be in the vocabulary
        assert 'euros' in vocab
        assert 'num' in vocab # The token for number replacement
        assert 'great' in vocab
        
    def test_transform_messages(self, sample_text_data):
        """Test the transformation (transform) of messages into feature matrix."""
        engineer = TextPreprocessor()
        _, messages, _ = sample_text_data
        
        # 1. Fit the vectorizer
        engineer.fit_vectorizer(messages)
        
        # 2. Transform the messages
        X_features = engineer.transform_messages(messages)
        
        # Should return a sparse matrix
        assert issparse(X_features), "The output should be a sparse matrix"
        
        # Check the shape (Number of samples x Number of features)
        n_samples = len(messages)
        n_features = len(engineer.selected_features)
        
        assert X_features.shape == (n_samples, n_features), "The shape of the feature matrix is incorrect"
        
    def test_transform_before_fit_raises_error(self, sample_text_data):
        """Test that transforming without fitting raises an error."""
        engineer = TextPreprocessor()
        _, messages, _ = sample_text_data
        
        with pytest.raises(RuntimeError, match="Vectorizer must be fitted first"):
            engineer.transform_messages(messages)
