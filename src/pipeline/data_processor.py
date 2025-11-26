"""
Data processing module for the Spam Detection ML pipeline.

This module handles loading text data (SMS/Email), basic cleaning, and Stratified Train/Test splitting.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# Import Spam-specific configurations
from utils.config import DATA_PATH, SMS_FILE, EMAIL_FILE, TRAIN_TEST_SPLIT_SIZE, RANDOM_STATE, TARGET_COL, MESSAGE_COL, POSITIVE_CLASS_LABEL
from utils.logger import get_logger


class DataProcessor:
    """
    Data processor for spam detection.
    
    Handles loading, cleaning, and splitting text data.
    """
    pass
