import pytest
import pandas as pd
import numpy as np
from src.pipeline.feature_engineer import FeatureEngineer

class TestFeatureEngineer:
    
    @pytest.fixture
    def feature_engineer(self):
        return FeatureEngineer(max_features=100)
    
    def test_initialization(self, feature_engineer):
        assert feature_engineer.max_features == 100
        assert feature_engineer.use_tfidf is False
        assert feature_engineer.vectorizer is not None
        
    def test_preprocess_text(self, feature_engineer):
        # Test basic lowercasing
        assert feature_engineer.preprocess_text("HELLO") == "hello"
        
        # Test number replacement
        assert feature_engineer.preprocess_text("Call 123 now") == "call <NUM> now"
        assert feature_engineer.preprocess_text("0123456789") == "<NUM>"
        
        # Test whitespace normalization
        assert feature_engineer.preprocess_text("  hello   world  ") == "hello world"
        
        # Test combined
        assert feature_engineer.preprocess_text("WINNER!! Call 0900 123") == "winner!! call <NUM> <NUM>"
        
        # Test non-string input
        assert feature_engineer.preprocess_text(123) == "123"

    def test_fit_transform(self, feature_engineer):
        train_texts = [
            "Hello world",
            "Hello python",
            "Spam message <NUM>",
            "Another spam <NUM>"
        ]
        
        # Test fit
        feature_engineer.fit(train_texts)
        vocab = feature_engineer.vectorizer.get_feature_names_out()
        assert "hello" in vocab
        assert "num" in vocab  # <NUM> becomes num or <num> depending on token pattern handling
        
        # Test transform
        matrix = feature_engineer.transform(train_texts)
        assert matrix.shape[0] == 4
        assert matrix.shape[1] <= 100
        
        # Test fit_transform
        matrix2 = feature_engineer.fit_transform(train_texts)
        assert matrix2.shape == matrix.shape

    def test_balance_data(self, feature_engineer):
        # Create imbalanced data: 3 'ham' vs 1 'spam'
        messages = pd.Series(["ham1", "ham2", "ham3", "spam1"], name="message")
        labels = pd.Series([0, 0, 0, 1], name="label")
        
        balanced_msgs, balanced_lbls = feature_engineer.balance_data(messages, labels)
        
        # Check if counts are equal
        counts = balanced_lbls.value_counts()
        assert counts[0] == counts[1]
        assert counts[0] == 3  # Should match majority class size
        
        # Check if total size is correct (3 + 3 = 6)
        assert len(balanced_lbls) == 6
        
    def test_balance_data_already_balanced(self, feature_engineer):
        messages = pd.Series(["ham1", "spam1"], name="message")
        labels = pd.Series([0, 1], name="label")
        
        balanced_msgs, balanced_lbls = feature_engineer.balance_data(messages, labels)
        
        assert len(balanced_lbls) == 2
        assert balanced_lbls.value_counts()[0] == 1
