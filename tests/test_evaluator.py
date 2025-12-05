import pytest
import numpy as np
from src.pipeline.evaluator import Evaluator
from src.utils.config import POSITIVE_CLASS_LABEL

class TestEvaluator:
    @pytest.fixture
    def evaluator(self):
        """Fixture to initialize the Evaluator."""
        return Evaluator()
    @pytest.fixture
    def perfect_predictions(self):
        """Case where predictions match truth exactly."""
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1])
        return y_true, y_pred
    
    @pytest.fixture
    def mixed_predictions(self):
        """
        Case with mixed errors to test Precision vs Recall.
        
        y_true: [0, 1, 1, 0]
        y_pred: [0, 1, 0, 1]
        
        Analysis:
        - Index 0: TN (Ham correctly identified)
        - Index 1: TP (Spam correctly identified)
        - Index 2: FN (Spam missed)
        - Index 3: FP (Ham incorrectly flagged as Spam)
        """
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 1])
        return y_true, y_pred