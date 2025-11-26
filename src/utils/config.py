"""
Configuration for Spam ML Pipeline.

This module contains all configuration constants used throughout the pipeline.
"""

from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

# File names (according to spam.ipynb)
SMS_FILE = "sms_spam.csv"
EMAIL_FILE = "email_spam.csv"

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Column names
TARGET_COL = "label"       # Target column ('ham'/'spam')
MESSAGE_COL = "message"    # Text column
POSITIVE_CLASS_LABEL = "spam" # Positive class label

# =============================================================================
# PREPROCESSING CONFIGURATION
# =============================================================================

# Train/Test split size
TRAIN_TEST_SPLIT_SIZE = 0.2

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION (NLP) / TEXT PREPROCESSING
# =============================================================================

# Vectorizer type: 'TfidfVectorizer' or 'CountVectorizer'
VECTORIZER_TYPE = "TfidfVectorizer"
MAX_FEATURES = 5000 # Maximum number of features (words)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Available model types (Classification)
MODEL_TYPES = ["logistic_regression", "naive_bayes", "bernoulli_nb", "linear_svc"] 
MODEL_TYPE_NAMES = {
    "logistic_regression": "LogisticRegression",
    "naive_bayes": "MultinomialNB",
    "bernoulli_nb": "BernoulliNB",
    "linear_svc": "LinearSVC"
}

# Default hyperparameters for optimization
DEFAULT_PARAM_GRIDS = {
    "logistic_regression": {
        'C': [0.1, 1.0, 10.0],
        'solver': ['liblinear'] # Good choice for sparse data
    },
    "naive_bayes": {
        'alpha': [0.01, 0.1, 1.0] # Smoothing parameter
    },
    "bernoulli_nb": {
        'alpha': [0.01, 0.1, 1.0] # Smoothing parameter
    },
    "linear_svc": {
        'C': [0.1, 1.0, 10.0]
    }
}

# Number of iterations for logistic regression (from spam.ipynb)
NB_ITERATIONS = 1000

# Random state for reproducibility
RANDOM_STATE = 3

# =============================================================================
# MLFLOW CONFIGURATION
# =============================================================================
MLFLOW_EXPERIMENT_NAME = "spam_detection_ml"
MLFLOW_TRACKING_URI = "./mlruns"
