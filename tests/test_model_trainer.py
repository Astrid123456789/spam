"""
Technical validation tests for the ModelTrainer class (Spam Detection).

These tests validate the model training functionality, including
model creation, training, and prediction for classification models.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix # For NLP feature matrices

from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC

from unittest.mock import MagicMock
from src.pipeline.model_trainer import ModelTrainer
from utils.config import MODEL_TYPES


# Mock mlflow to prevent errors during testing
@pytest.fixture(autouse=True)
def mock_mlflow(monkeypatch):
    """Mock mlflow to avoid creating actual runs during tests."""
    mock = MagicMock()
    # Patch the module-level mlflow object in model_trainer
    monkeypatch.setattr("src.pipeline.model_trainer.mlflow", mock)
    return mock


# Fixture for classification data (Simulated sparse matrix)
@pytest.fixture(scope="session")
def sample_X_y():
    """Create a small sparse feature matrix (X) and a binary target (y)."""
    # X: Sparse matrix suitable for text classification models (e.g., TF-IDF)
    X_data = np.array([
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
        [1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    ])
    X = csr_matrix(X_data)
    
    # y: Binary target for classification (0=ham, 1=spam)
    y = np.array([0, 1, 0, 1, 0])
    
    return X, y


class TestModelTrainer:
    """Suite of tests for the technical validation of ModelTrainer."""
    
    def test_create_model_logistic_regression(self):
        """Test the creation of the Logistic Regression model."""
        trainer = ModelTrainer()
        model = trainer.create_model('logistic_regression')
        assert isinstance(model, LogisticRegression)
    
    def test_create_model_naive_bayes(self):
        """Test the creation of the Multinomial Naive Bayes model."""
        trainer = ModelTrainer()
        model = trainer.create_model('naive_bayes')
        assert isinstance(model, MultinomialNB)
        
    def test_create_model_bernoulli_nb(self):
        """Test the creation of the Bernoulli Naive Bayes model. (New)"""
        trainer = ModelTrainer()
        model = trainer.create_model('bernoulli_nb')
        assert isinstance(model, BernoulliNB)
        
    def test_create_model_linear_svc(self):
        """Test the creation of the Linear SVC model. (New)"""
        trainer = ModelTrainer()
        model = trainer.create_model('linear_svc')
        assert isinstance(model, LinearSVC)
    
    def test_train_single_model(self, sample_X_y):
        """Test the training of a single model."""
        trainer = ModelTrainer()
        X, y = sample_X_y
        
        # We test with one of the new models to ensure compatibility with sparse matrices
        model = trainer.train_single_model(X, y, model_type='linear_svc') 
        
        assert model is not None
        assert 'linear_svc' in trainer.trained_models
        # For classification models, checking for 'classes_' is essential
        assert hasattr(model, 'classes_'), "The trained model should have classification attributes (classes_)"
        
    def test_predict(self, sample_X_y):
        """Test prediction."""
        trainer = ModelTrainer()
        X, y = sample_X_y
        
        model = trainer.train_single_model(X, y, model_type='naive_bayes')
        predictions = trainer.predict(model, X)
        
        assert predictions.shape == y.shape
        assert np.all(np.isin(predictions, [0, 1])), "Predictions should be binary (0 or 1)"

    def test_train_all_models(self, sample_X_y):
        """Test training of all model types defined in the configuration."""
        trainer = ModelTrainer()
        X, y = sample_X_y
        
        for model_type in MODEL_TYPES:
            trainer.train_single_model(X, y, model_type=model_type)
            assert model_type in trainer.trained_models
