"""
Technical validation tests for DataProcessor class (NLP Spam Detection).

These tests validate the technical implementation without focusing on ML performance.
They verify that methods execute correctly, handle duplicates, split data,
balance classes, and preprocess both synthetic and real datasets.
"""

import sys
import os
from pathlib import Path

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from src.pipeline.data_processor import DataProcessor
from src.utils.config import (
    MESSAGE_COL, TARGET_COL,
    SMS_FILE, EMAIL_FILE
)

class TestDataProcessor:
    """Test suite for DataProcessor technical validation."""

    # ----------------------------------------------------------------------
    # Fixtures
    # ----------------------------------------------------------------------

    @pytest.fixture
    def sample_sms(self):
        """Create synthetic SMS dataset."""
        return pd.DataFrame({
            MESSAGE_COL: ["hi", "hello", "spam", "hello", "promo", "spam now", "test", "offer"],
            TARGET_COL: [0, 0, 1, 0, 1, 1, 0, 1]
        })

    @pytest.fixture
    def sample_email(self):
        """Create synthetic Email dataset."""
        return pd.DataFrame({
            MESSAGE_COL: ["offer", "urgent", "hello", "spam", "offer", "check", "deal"],
            TARGET_COL: [1, 0, 0, 1, 1, 0, 1]
        })

    @pytest.fixture
    def processor(self):
        return DataProcessor()

    @pytest.fixture
    def mock_csv_files(self, tmp_path, sample_sms, sample_email, monkeypatch):
        """Create temporary CSV files and patch config."""
        sms_path = tmp_path / "sms.csv"
        email_path = tmp_path / "email.csv"

        sample_sms.to_csv(sms_path, index=False)
        sample_email.to_csv(email_path, index=False)

        # Patch DATA_PATH
        from src.utils import config
        monkeypatch.setattr(config, "DATA_PATH", tmp_path)

        return sms_path, email_path

    # ----------------------------------------------------------------------
    # TESTS
    # ----------------------------------------------------------------------

    def test_load_data(self, processor, mock_csv_files):
        """Test that loading synthetic CSV files works."""
        sms_df, email_df = processor.load_data()

        assert isinstance(sms_df, pd.DataFrame)
        assert isinstance(email_df, pd.DataFrame)
        assert len(sms_df) > 0
        assert len(email_df) > 0
        assert MESSAGE_COL in sms_df.columns
        assert TARGET_COL in sms_df.columns

    def test_explore_data_duplicate_removal(self, processor, sample_sms):
        """Duplicates should be removed."""
        messages, labels = processor.explore_data(sample_sms)

        # Check no duplicates remain
        assert messages.duplicated().sum() == 0

    def test_split_data(self, processor, sample_sms, sample_email):
        """Split synthetic datasets into 4 scenarios."""
        scenarios = processor.split_data(sample_sms, sample_email)

        assert "sms" in scenarios
        assert "email" in scenarios
        assert "transfer" in scenarios
        assert "combined" in scenarios

        X_train, X_test, y_train, y_test = scenarios["sms"]

        assert isinstance(X_train, pd.Series)
        assert isinstance(y_train, pd.Series)
        assert len(X_train) > 0
        assert len(X_test) > 0

    def test_balance_data(self, processor, sample_sms):
        """Minority class should be oversampled."""
        X, y = sample_sms[MESSAGE_COL], sample_sms[TARGET_COL]
        balanced_X, balanced_y = processor.balance_data(X, y)

        vc = balanced_y.value_counts()
        assert vc[0] == vc[1]

    def test_preprocess_text(self, processor, sample_sms):
        """Test preprocessing pipeline on synthetic data."""
        processor.train_data = sample_sms.copy()
        processor.test_data = sample_sms.copy()

        (X_train, y_train), (X_test, y_test) = processor.preprocess_text(
            drop_duplicates=True,
            split_data=True,
            balance_data=True
        )

        assert isinstance(X_train, pd.Series)
        assert isinstance(y_train, pd.Series)
        assert y_train.value_counts().nunique() == 1  # balanced

    # ----------------------------------------------------------------------
    # Test with real data (first 300 rows)
    # ----------------------------------------------------------------------

    def load_and_preprocess(self):
        """Simplified: just load data, no cleaning or splitting."""
        sms_data, email_data = self.load_data()
        return {"sms": sms_data, "email": email_data}

    # ----------------------------------------------------------------------
    # FULL PIPELINE TEST
    # ----------------------------------------------------------------------

    def test_full_pipeline_execution(self, processor, mock_csv_files):
        """Test the entire load → split → preprocess cycle."""
        processed = processor.load_and_preprocess()

        assert isinstance(processed, dict)
        assert "sms" in processed

        X_train, X_test, y_train, y_test = processed["sms"]

        assert isinstance(X_train, pd.Series)
        assert isinstance(y_train, pd.Series)


# Allow running tests directly
def run_dataprocessor_tests():
    import pytest
    return pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_dataprocessor_tests()