"""
Configuration for Spam ML Pipeline.

This module contains all configuration constants used throughout
the pipeline. Students don't need to modify this file.
"""

from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Base project directory (automatically detected)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# File names
# If you end up creating a combined dataset, you can add COMBINED_FILE later.
SMS_TRAIN_FILE = "sms_spam.csv"
EMAIL_TRAIN_FILE = "email_spam.csv"

# Column names
# Classic spam datasets use these names; this matches your notebook:
ID_COL = "id"              # Optional – only if your CSVs have an ID column
TEXT_COL = "message"       # The raw text of the message (SMS or email)
TARGET_COL = "label"       # 0 = ham, 1 = spam (usually)

# If you later add a combined dataset you might also want:
SOURCE_COL = "source"      # e.g. "sms" or "email" (optional)

# =============================================================================
# PREPROCESSING CONFIGURATION
# =============================================================================

# Missing data threshold (drop columns with more than X% missing)
MISSING_THRESHOLD = 0.7

# Cross-validation splits
N_SPLITS = 4

# Train/test split used in the notebook
TRAIN_TEST_SPLIT_SIZE = 0.2

# =============================================================================
# FEATURE ENGINEERING / TEXT CONFIGURATION
# =============================================================================

# Vectorization options – you can use these in your pipeline if you want
USE_TFIDF = True           # True: TfidfVectorizer, False: CountVectorizer
MAX_FEATURES = 10000       # Max vocabulary size (optional)

# Language / text-cleaning options (optional but handy)
STOPWORDS_LANGUAGE = "english"

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Available model types (example – adapt to what you actually implement)
MODEL_TYPES = ["logistic_regression"]

# Default hyperparameter grids for optimization
DEFAULT_PARAM_GRIDS = {
    "logistic_regression": {
        "C": [0.1, 1.0, 10.0],
        "max_iter": [100, 300, 1000]
    },
    # You can add others later, e.g.:
    # "naive_bayes": {...},
    # "xgboost": {...},
}

# Random state for reproducibility
RANDOM_STATE = 3  # matches the notebook

# =============================================================================
# MLFLOW CONFIGURATION
# =============================================================================

MLFLOW_EXPERIMENT_NAME = "spam_detection_ml"
MLFLOW_TRACKING_URI = "./mlruns"

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

# Metrics to calculate for classification instead of regression
# Make sure your evaluator uses these names.
METRICS = ["accuracy", "precision", "recall", "f1"]

def get_data_file_path(filename):
    """Get full path to a data file."""
    return DATA_PATH / filename
