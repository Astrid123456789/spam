"""
Spam ML Utilities Package

This package contains utility functions and configurations:
- config: Configuration constants and settings
- logger: Professional logging system
- utils: General utility functions (e.g., I/O, plotting helpers)
- evaluation_utils: Detailed evaluation functions with visualizations (e.g., ROC, Confusion Matrix)

Usage:
    from utils.config import *
    from utils.utils import print_step_header
    from utils.evaluation_utils import evaluate_model_detailed
"""

__version__ = "1.0.0"

# Import all configuration constants directly (using '*' as requested)
from .config import *

# Import commonly used utility functions
from .utils import print_step_header, print_results_summary, validate_data_files, save_plot

# Import core logger functions for control (essential for any complex ML project)
from .logger import get_logger, set_log_level

# Import specialized evaluation function (for detailed reports/visualizations)
from .evaluation_utils import evaluate_model_detailed


__all__ = [
    # Config constants (Spam Detection Specific)
    "DATA_PATH",
    "SMS_FILE", # Corrected file name
    "EMAIL_FILE", # Corrected file name
    "TARGET_COL",
    "MESSAGE_COL", # Used instead of TEXT_COL for consistency with the text domain
    "POSITIVE_CLASS_LABEL", # Essential for binary classification
    "VECTORIZER_TYPE", # Essential for FeatureEngineer
    "NB_ITERATIONS", # Used for training Logistic Regression/LinearSVC max_iter
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
    "validate_data_files",
    "save_plot",
    
    # Logger functions
    "get_logger",
    "set_log_level",

    # Evaluation functions
    "evaluate_model_detailed"
]
