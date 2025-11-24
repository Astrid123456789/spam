"""
Spam ML Utilities Package

This package contains utility functions and configurations:
- config: Configuration constants and settings
- utils: General utility functions
- evaluation_utils: Detailed evaluation functions with visualizations

Usage:
    from utils.config import *
    from utils.utils import print_step_header
    from utils.evaluation_utils import evaluate_model_detailed
"""

__version__ = "1.0.0"

# Import commonly used items
from .config import *
from .utils import print_step_header, print_results_summary

__all__ = [
    # Config constants
    "DATA_PATH",
    "SMS_TRAIN_FILE",
    "EMAIL_TRAIN_FILE",
    "TEXT_COL",
    "TARGET_COL",
    "MODEL_TYPES",
    "DEFAULT_PARAM_GRIDS",
    "N_SPLITS",
    "TRAIN_TEST_SPLIT_SIZE",
    "USE_TFIDF",
    "MAX_FEATURES",
    "RANDOM_STATE",
    "METRICS",
    "MLFLOW_EXPERIMENT_NAME",
    "MLFLOW_TRACKING_URI",
    # Utility functions
    "print_step_header",
    "print_results_summary",
]
