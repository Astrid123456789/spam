"""
Simple Spam Detection ML Pipeline with Inline MLflow Integration

This pipeline includes MLflow logging directly in the main workflow, adapting 
the configuration for text classification tasks.
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import pipeline components (Classification)
from pipeline import DataProcessor, TextPreprocessor, ModelTrainer
from pipeline.evaluator import Evaluator
from utils.config import MODEL_TYPES, MODEL_TYPE_NAMES
from utils.config import DEFAULT_PARAM_GRIDS, POSITIVE_CLASS_LABEL

from utils.logger import get_logger, set_log_level, log_level_from_string, LogLevel
from utils.utils import format_time_elapsed
from utils.utils import validate_data_files

# Import MLflow
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False


def run_pipeline(args):
    """
    Runs the complete spam classification pipeline with MLflow integration.
    """
    start_time = time.time()
    logger = get_logger()

    if args.mlflow and not MLFLOW_AVAILABLE:
        logger.warning("MLflow requested but not installed. Install with: pip install mlflow")
        logger.info("Continuing without MLflow tracking...")
        args.mlflow = False

    # MLflow Configuration and Start
    if args.mlflow:
        # Assuming MLFLOW_EXPERIMENT_NAME is available via config import
        # For simplicity, we hardcode here, but ideally, it comes from config
        experiment_name = "spam_detection_ml"
        mlflow.set_experiment(experiment_name)
        run = mlflow.start_run(run_name=f"main_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        logger.info(f"MLflow Run ID: {run.info.run_id}")
    
    try:
        # 1. Data Validation
        validate_data_files()
        
        # 2. Data Loading and Preprocessing
        logger.step(1, "DATA LOADING AND PREPROCESSING", total_steps=4)
        data_processor = DataProcessor()
        # Load and split data for classification
        data_processor.load_data() 
        
        # 3. Feature Engineering (Text Vectorization)
        logger.step(2, "TEXT PREPROCESSING AND VECTORIZATION", total_steps=4)
        text_preprocessor = TextPreprocessor()
        
        # Apply feature transformation (e.g., TF-IDF)
        X_train, y_train, X_test, y_test = text_preprocessor.load_and_transform_data(
            data_processor.train_data, data_processor.test_data
        )

        logger.success(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features (y: {y_train.shape})")
        logger.success(f"Test data: {X_test.shape[0]} samples, {X_test.shape[1]} features (y: {y_test.shape})")

        # 4. Model Training and Optimization
        logger.step(3, "MODEL TRAINING AND EVALUATION", total_steps=4)
        model_trainer = ModelTrainer()
        
        # Note: 'groups' is usually for time-series/geographic data, kept as placeholder 
        # but typically None for standard text classification.
        # groups_train = None 
        
        if args.compare:
            # Model Comparison
            model_trainer.train_multiple_models(X_train, y_train, model_types=MODEL_TYPES, mlflow_tracking=args.mlflow)
            
            # Select best model
            best_model, best_name, _ = model_trainer.compare_models(mlflow_tracking=args.mlflow)
            logger.info(f"Best model selected: {best_name.upper()}")

        elif args.optimize and args.model:
            # Single Model Optimization
            model_type = args.model
            
            if model_type not in MODEL_TYPES:
                 raise ValueError(f"Unsupported model type: {model_type}. Supported: {MODEL_TYPES}")

            param_grid = DEFAULT_PARAM_GRIDS.get(model_type, {})
            
            # The Evaluator must be configured for classification metrics (AUC/F1)
            best_model, best_params, best_score = model_trainer.evaluator.optimize_hyperparameters(
                model_trainer.create_model(model_type),
                X_train,
                y_train,
                param_grid=param_grid,
                # groups=groups_train,
                mlflow_tracking=args.mlflow
            )
            model_trainer.trained_models[model_type] = best_model
            model_trainer.best_model = best_model
            model_trainer.best_model_name = model_type
            
        elif args.model:
            # Single Model Training without optimization
            best_model = model_trainer.train_single_model(X_train, y_train, args.model, mlflow_tracking=args.mlflow)
            model_trainer.best_model = best_model
            model_trainer.best_model_name = args.model
            
        else:
            logger.error("Please specify --model for training or --compare for model comparison.")
            return

        # 5. Final Evaluation
        logger.step(4, "EVALUATION ON TEST SET", total_steps=4)
        
        if model_trainer.best_model is None:
             logger.error("No model was trained or selected for final evaluation.")
             return

        # Predictions on the test set
        y_pred = model_trainer.predict(model_trainer.best_model, X_test)
        
        # Calculate classification metrics
        final_metrics = model_trainer.evaluator.calculate_metrics(y_test, y_pred, positive_label=POSITIVE_CLASS_LABEL)
        
        # Logging final results
        logger.results_summary({"Final Test Metrics": final_metrics})

        if args.mlflow:
            # Log final metrics and model to MLflow
            mlflow.log_metrics({f"final_test_{k}": v for k, v in final_metrics.items()})
            mlflow.sklearn.log_model(model_trainer.best_model, "final_model")
            
    except Exception as e:
        logger.error(f"A critical error occurred: {e}")
        # Terminate MLflow session on failure
        if args.mlflow and mlflow.active_run():
            mlflow.end_run(status="FAILED")
        raise
    
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Close MLflow session on success
        if args.mlflow and mlflow.active_run():
            mlflow.end_run(status="FINISHED")

        logger.final_success(elapsed_time)


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Spam Detection ML Pipeline.")

    # Model Configuration
    parser.add_argument(
        '--model', type=str, default=None,
        choices=MODEL_TYPES,
        help=f"Type of model to train ({', '.join(MODEL_TYPES)})"
    )

    # Pipeline Parameters
    parser.add_argument(
        '--optimize', action='store_true',
        help='Enable hyperparameter optimization (GridSearchCV)'
    )
    
    parser.add_argument(
        '--compare', action='store_true',
        help='Compare multiple models instead of training a single model'
    )
    
    # Logging Configuration
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose output (deprecated, use --log-level verbose)'
    )
    
    parser.add_argument(
        '--log-level', type=str, default='normal',
        choices=['silent', 'normal', 'verbose'],
        help='Logging level: silent (no output), normal (main steps), verbose (all details)'
    )
    
    # MLflow Configuration
    parser.add_argument(
        '--mlflow', action='store_true',
        help='Enable MLflow tracking for the pipeline'
    )
    
    parser.add_argument(
        '--mlflow-experiment', type=str, default="spam_detection_ml",
        help='MLflow experiment name to use'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Configure logging level
        if args.verbose:
            log_level = LogLevel.VERBOSE
        else:
            log_level = log_level_from_string(args.log_level)
        
        set_log_level(log_level)
        
        # Run pipeline
        run_pipeline(args)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        # Error message is printed inside run_pipeline, but ensuring exit code is 1
        # In case of unhandled exception outside run_pipeline:
        if get_logger().level > LogLevel.SILENT:
             print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

