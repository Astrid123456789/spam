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