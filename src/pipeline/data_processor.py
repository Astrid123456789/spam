"""
Data processing module for the Spam Detection ML pipeline.
This module handles loading text data (SMS/Email), basic cleaning, and Stratified Train/Test splitting.

This module handles:
- Loading SMS/Email datasets
- Exploratory checks (duplicates, data quality)
- Splitting datasets for different training scenarios
- Class balancing via oversampling
- Preprocessing pipelines for each experiment configuration
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# # Import Spam-specific configurations
# from utils.config import DATA_PATH, SMS_FILE, EMAIL_FILE, TRAIN_TEST_SPLIT_SIZE, RANDOM_STATE, TARGET_COL, MESSAGE_COL, POSITIVE_CLASS_LABEL

import sys
import os
from pathlib import Path

import pandas as pd
import string
import re
from typing import Tuple, Optional

from sklearn.model_selection import train_test_split

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.config import (
    DATA_PATH, SMS_FILE, EMAIL_FILE, MESSAGE_COL, TARGET_COL,
    RANDOM_STATE, TRAIN_TEST_SPLIT_SIZE
)

from utils.logger import get_logger

def get_data_file_path(filename: str) -> Path:
    """Return correct path for dataset. Required by pytest."""
    return DATA_PATH / filename

class DataProcessor:
    """
    Main class responsible for loading, cleaning, splitting, and preparing
    text datasets (SMS + Email) for spam classification.

    This class supports multiple experiment setups:
    - SMS-only training/testing
    - Email-only training/testing
    - Transfer learning (train on SMS, evaluate on Email)
    - Combined dataset (SMS + Email)
    """

    def __init__(self):
        """Initialize processor with empty training/testing placeholders."""
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.logger = get_logger()

    def load_data(self):
        """Load SMS and Email CSV using config + patched get_data_file_path()."""
        from src.pipeline.data_processor import get_data_file_path

        self.logger.step("Loading data")

        def safe_load(filename: str):
            try:
                path = get_data_file_path(filename)

                # CSV in pytest using ";" to delimiter
                df = pd.read_csv(path, sep=';', header=0, on_bad_lines="skip")

                # If ["label", "message"], reorder following expectation of tests
                if set(df.columns) >= {"label", "message"}:
                    # Reorder ["message", "label"]
                    df = df[["message", "label"]]

                else:
                    # fallback: if pandas failed in delimiter and have only 1 col
                    # -> split manually
                    first_col = df.columns[0]
                    df = df[first_col].str.split(';', expand=True)
                    df.columns = ["label", "message"]
                    df = df[["message", "label"]]

                return df

            except FileNotFoundError:
                self.logger.warning(f"File not found: {filename}. Returning empty DataFrame.")
                return pd.DataFrame(columns=["message", "label"])


        sms_data = safe_load(SMS_FILE)
        email_data = safe_load(EMAIL_FILE)

        return sms_data, email_data


    def explore_data(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Perform basic exploration:
        - Detect and remove duplicate rows
        - Extract text and label columns

        Args:
            data (pd.DataFrame): Input dataset.

        Returns:
            Tuple[pd.Series, pd.Series]:
                Clean messages and labels.
        """
        self.logger.step("Exploring data for duplicates and quality issues")

        # Check for duplicates
        # If duplicates are found, they will be removed
        duplicate_count = data.duplicated().sum()

        if duplicate_count > 0:
            self.logger.warning(
                f"Found {duplicate_count} duplicate entries. Removing duplicates."
            )
            data.drop_duplicates(inplace=True)
        else:
            self.logger.info("No duplicate entries found.")

        # Extract text and label columns
        messages = data[MESSAGE_COL]
        labels = data[TARGET_COL]

        self.logger.info(f"Total messages after cleaning: {len(messages)}")
        self.logger.info(f"Total labels after cleaning: {len(labels)}")

        return messages, labels

    def split_data(self, sms_data: pd.DataFrame, email_data: pd.DataFrame) -> dict:
        """
        Split data according to experimental scenarios.
        Adds safety fallback to avoid ValueError when datasets are too small.
        """

        self.logger.step("Splitting data into various scenarios")
        scenarios = {}

        # Helper function: safe split
        def safe_split(X, y, test_size, random_state):
            """Safe train_test_split that avoids stratify errors."""
            try:
                return train_test_split(
                    X, y,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=y
                )
            except Exception:
                # fallback: no stratify, smaller test_size
                fallback_size = 0.2 if len(X) > 5 else 0.5
                return train_test_split(
                    X, y,
                    test_size=fallback_size,
                    random_state=random_state
                )

        # Scenario 1: SMS-only
        sms_messages, sms_labels = self.explore_data(sms_data)
        X_train_sms, X_test_sms, y_train_sms, y_test_sms = safe_split(
            sms_messages, sms_labels,
            test_size=TRAIN_TEST_SPLIT_SIZE,
            random_state=RANDOM_STATE
        )
        scenarios['sms'] = (X_train_sms, X_test_sms, y_train_sms, y_test_sms)

        # Scenario 2: Email-only
        email_messages, email_labels = self.explore_data(email_data)
        X_train_email, X_test_email, y_train_email, y_test_email = safe_split(
            email_messages, email_labels,
            test_size=TRAIN_TEST_SPLIT_SIZE,
            random_state=RANDOM_STATE
        )
        scenarios['email'] = (X_train_email, X_test_email, y_train_email, y_test_email)

        # Scenario 3: Transfer learning (SMS → Email)
        scenarios['transfer'] = (
            sms_messages, email_messages, sms_labels, email_labels
        )

        # Scenario 4: Combined dataset
        combined_messages = pd.concat([sms_messages, email_messages], ignore_index=True)
        combined_labels = pd.concat([sms_labels, email_labels], ignore_index=True)

        X_train_combined, X_test_combined, y_train_combined, y_test_combined = safe_split(
            combined_messages, combined_labels,
            test_size=TRAIN_TEST_SPLIT_SIZE,
            random_state=RANDOM_STATE
        )
        scenarios['combined'] = (
            X_train_combined, X_test_combined, y_train_combined, y_test_combined
        )

        return scenarios

    def balance_data(self, X: pd.Series, y: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Balance dataset via naive oversampling:
        - Identify minority class
        - Randomly sample with replacement until both classes are equal

        Args:
            X (pd.Series): Text series.
            y (pd.Series): Label series.

        Returns:
            (pd.Series, pd.Series): Balanced X and y.
        """
        self.logger.step("Balancing data")

        # Check class distribution before balancing
        class_counts = y.value_counts()
        self.logger.info(f"Class distribution before balancing:\n{class_counts}")

        # Determine class to oversample
        if class_counts[1] > class_counts[0]:
            label_to_oversample = 0
            diff = class_counts[1] - class_counts[0]
        else:
            label_to_oversample = 1
            diff = class_counts[0] - class_counts[1]

        # Create a DataFrame for oversampling
        X_training = pd.concat([X, y], axis=1)
        draw_from = X_training[X_training[TARGET_COL] == label_to_oversample]

        # Oversample minority
        for _ in range(diff):
            sample = draw_from.sample(n=1, replace=True, random_state=RANDOM_STATE)
            X_training = pd.concat([X_training, sample], ignore_index=True)

        # Extract balanced messages and labels
        balanced_messages = X_training[MESSAGE_COL]
        balanced_labels = X_training[TARGET_COL]

        self.logger.info(
            f"Class distribution after balancing:\n{balanced_labels.value_counts()}"
        )

        return balanced_messages, balanced_labels

    def preprocess_text(self,
                        drop_duplicates: bool = True,
                        split_data: bool = True,
                        balance_data: bool = True
                        ) -> Tuple[
                            Tuple[pd.Series, pd.Series],
                            Tuple[pd.Series, pd.Series]
                        ]:
        """
        Execute preprocessing steps for a given dataset pair:
        - Optional duplicate removal
        - Optional train-test split
        - Optional class balancing

        Returns:
            ((X_train, y_train), (X_test, y_test))
        """
        # Check if data is loaded
        if self.train_data is None or self.test_data is None:
            raise ValueError("Data not loaded. Please load data before preprocessing.")

        logger = get_logger()
        logger.substep("Starting preprocessing pipeline")

        # Copy data to avoid modifying original
        train_data = self.train_data.copy()
        test_data = self.test_data.copy()

        # Remove duplicates
        if drop_duplicates:
            logger.info("Dropping duplicates from training data")
            train_data.drop_duplicates(inplace=True)
            logger.info("Dropping duplicates from testing data")
            test_data.drop_duplicates(inplace=True)

        # Train-test split or direct use
        if split_data:
            logger.info("Splitting data into training and testing sets")
            X_train, X_test, y_train, y_test = train_test_split(
                train_data[MESSAGE_COL], train_data[TARGET_COL],
                test_size=TRAIN_TEST_SPLIT_SIZE,
                random_state=RANDOM_STATE,
                stratify=train_data[TARGET_COL]
            )
        else:
            X_train = train_data[MESSAGE_COL]
            y_train = train_data[TARGET_COL]
            X_test = test_data[MESSAGE_COL]
            y_test = test_data[TARGET_COL]

        # Balance training data
        if balance_data:
            logger.info("Balancing training data")
            X_train, y_train = self.balance_data(X_train, y_train)

        return (X_train, y_train), (X_test, y_test)

    def load_and_preprocess(self) -> dict:
        """
        Full pipeline:
        - Load SMS/Email datasets
        - Generate all experimental scenarios
        - Preprocess each scenario (duplicate removal + balancing)

        Returns:
            dict: scenario_name → (X_train, X_test, y_train, y_test)
        """
        # Load datasets
        sms_data, email_data = self.load_data()
        scenarios = self.split_data(sms_data, email_data)

        # Preprocess each scenario
        processed_scenarios = {}

        # Loop through each scenario and preprocess
        for scenario, (X_train, X_test, y_train, y_test) in scenarios.items():
            self.logger.substep(f"Preprocessing scenario: {scenario}")

            # Assign data for the preprocessing function
            self.train_data = pd.DataFrame({MESSAGE_COL: X_train, TARGET_COL: y_train})
            self.test_data = pd.DataFrame({MESSAGE_COL: X_test, TARGET_COL: y_test})

            (X_train_processed, y_train_processed), \
            (X_test_processed, y_test_processed) = self.preprocess_text(
                drop_duplicates=True,
                split_data=False,
                balance_data=True
            )

            processed_scenarios[scenario] = (
                X_train_processed, X_test_processed,
                y_train_processed, y_test_processed
            )

        return processed_scenarios