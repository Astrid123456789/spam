"""
Spam Detection ML Pipeline - Test Suite

This package contains comprehensive tests to validate the technical implementation
of the components of the Spam Detection ML pipeline (Text Classification).

Test Structure:
- test_data_processor.py: Technical validation for DataProcessor (Loading/Stratified Split)
- test_feature_engineer.py: Technical validation for FeatureEngineer (Text Preprocessing/Vectorization)
- test_model_trainer.py: Technical validation for ModelTrainer (Training Classification Models)
- test_integration.py: End-to-end integration tests (to be added)

Usage (via a script like run_tests.py or directly with pytest):
    # Run all tests
    # pytest tests/
    
    # Run a specific test module
    # pytest tests/test_model_trainer.py
"""

__version__ = "1.0.0"
__author__ = "Spam Detection ML Workshop - Test Suite"

# --- Import of Test Modules ---

# Import classes and helper functions for DataProcessor
try:
    from .test_data_processor import TestDataProcessor, run_dataprocessor_tests
    DATAPROCESSOR_AVAILABLE = True
except ImportError:
    DATAPROCESSOR_AVAILABLE = False

# Import classes and helper functions for TextPreprocessor
try:
    from .test_text_preprocessor import TestTextPreprocessor, run_feature_engineer_tests
    TEXT_PREPROCESSOR_AVAILABLE = True
except ImportError:
    TEXT_PREPROCESSOR_AVAILABLE = False

# Import classes and helper functions for ModelTrainer
try:
    from .test_model_trainer import TestModelTrainer, run_model_trainer_tests
    MODEL_TRAINER_AVAILABLE = True
except ImportError:
    MODEL_TRAINER_AVAILABLE = False

# Placeholder for future test imports
# from .test_integration import TestIntegration, run_integration_tests


__all__ = [
    # Test Classes
    'TestDataProcessor',
    'TestTextPreprocessor',
    'TestModelTrainer',
    
    # Functions for programmatic execution
    'run_dataprocessor_tests',
    'run_text_preprocessor_tests',
    'run_model_trainer_tests',
]


def get_available_test_modules():
    """
    Get list of available test modules.
    
    Returns:
        dict: Dictionary mapping module names to their availability status (bool).
    """
    modules = {
        'data_processor': DATAPROCESSOR_AVAILABLE,
        'text_preprocessor': TEXT_PREPROCESSOR_AVAILABLE,
        'model_trainer': MODEL_TRAINER_AVAILABLE,
        'integration': False,
    }
    return modules


def run_all_available_tests():
    """
    Run all available test modules via their dedicated execution functions.
    
    Returns:
        dict: RResults of each test module execution.
    """
    results = {}
    
    # Run DataProcessor tests
    if DATAPROCESSOR_AVAILABLE:
        try:
            results['data_processor'] = run_dataprocessor_tests()
        except Exception as e:
            results['data_processor'] = f"Error: {str(e)}"
    
    # Run TextPreprocessor tests
    if TEXT_PREPROCESSOR_AVAILABLE:
        try:
            results['feature_engineer'] = run_feature_engineer_tests()
        except Exception as e:
            results['feature_engineer'] = f"Error: {str(e)}"

    # Run ModelTrainer tests
    if MODEL_TRAINER_AVAILABLE:
        try:
            results['model_trainer'] = run_model_trainer_tests()
        except Exception as e:
            results['model_trainer'] = f"Error: {str(e)}"

    return results
