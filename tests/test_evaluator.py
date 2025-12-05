import pytest
import numpy as np
from src.pipeline.evaluator import Evaluator
from src.utils.config import POSITIVE_CLASS_LABEL

class TestEvaluator:
    @pytest.fixture
    def evaluator(self):
        """Fixture to initialize the Evaluator."""
        return Evaluator()