#!/usr/bin/env python3
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
# --- NOUVELLE IMPORTATION ---
from sklearn.metrics import roc_auc_score
# --- FIN NOUVELLE IMPORTATION ---

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
        experiment_name = args.mlflow_experiment
        mlflow.set_experiment(experiment_name)
        # Utiliser un nom de run plus descriptif
        run_name = f"{args.model.upper()}_{datetime.now().strftime('%H%M')}" if args.model else f"main_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run = mlflow.start_run(run_name=run_name)
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        
        # AJOUT MLFLOW : Enregistrement des arguments de la ligne de commande
        mlflow.log_param("model_type", args.model if args.model else "comparison")
        mlflow.log_param("optimization_enabled", args.optimize)
        mlflow.log_param("data_scenario", "combined") # On assume ici 'combined'
        mlflow.log_param("log_level", args.log_level)
    
    try:
        # 1. Data Validation
        validate_data_files()
        
        # 2. Data Loading and Preprocessing
        logger.step("DATA LOADING AND PREPROCESSING", 1, total_steps=4)
        data_processor = DataProcessor()
        # Load and preprocess data (splitting and balancing is handled here)
        scenarios = data_processor.load_and_preprocess()
        
        # Format: (X_train, X_test, y_train, y_test) as pd.Series
        # On utilise le scénario 'combined'
        X_train_txt, X_test_txt, y_train_s, y_test_s = scenarios['combined']
        
        # 3. Feature Engineering (Text Vectorization)
        logger.step("TEXT PREPROCESSING AND VECTORIZATION", 2, total_steps=4)
        text_preprocessor = TextPreprocessor()
        
        # Apply feature transformation (e.g., TF-IDF)
        X_train = text_preprocessor.fit_transform(X_train_txt)
        X_test = text_preprocessor.transform_messages(X_test_txt)
        
        # AJOUT MLFLOW : Enregistrement des paramètres du Vectorizer
        if args.mlflow:
            vectorizer = text_preprocessor.vectorizer
            mlflow.log_param("vectorizer_type", type(vectorizer).__name__)
            mlflow.log_param("vectorizer_max_features", vectorizer.max_features)
            mlflow.log_param("vocabulary_size", X_train.shape[1])
        
        # Convert labels to numeric (spam=1, ham=0) if not already
        if y_train_s.dtype == 'object':
            y_train = y_train_s.apply(lambda x: 1 if x == POSITIVE_CLASS_LABEL else 0).values
            y_test = y_test_s.apply(lambda x: 1 if x == POSITIVE_CLASS_LABEL else 0).values
        else:
            # Already numeric (0/1)
            y_train = y_train_s.values
            y_test = y_test_s.values

        logger.success(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features (y: {y_train.shape})")
        logger.success(f"Test data: {X_test.shape[0]} samples, {X_test.shape[1]} features (y: {y_test.shape})")

        # 4. Model Training and Optimization
        logger.step("MODEL TRAINING AND EVALUATION", 3, total_steps=4)
        model_trainer = ModelTrainer()
        
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
            
            # AJOUT MLFLOW : Enregistrement des hyperparamètres par défaut pour le modèle
            if args.mlflow:
                model_params = model_trainer.best_model.get_params()
                mlflow.log_params({f"model_param_{k}": v for k, v in model_params.items()})

        else:
            logger.error("Please specify --model for training or --compare for model comparison.")
            return

        # 5. Final Evaluation
        logger.step("EVALUATION ON TEST SET", 4, total_steps=4)
        
        if model_trainer.best_model is None:
             logger.error("No model was trained or selected for final evaluation.")
             return

        # Predictions on the test set (classes 0/1)
        y_pred = model_trainer.predict(model_trainer.best_model, X_test)
        
        # --- DÉBUT MODIFICATIONS POUR ROC-AUC ---
        roc_auc = None
        # Vérification si le modèle supporte predict_proba (nécessaire pour ROC-AUC)
        if hasattr(model_trainer.best_model, "predict_proba"):
            # Probabilités pour la classe positive (indice 1)
            y_pred_proba = model_trainer.best_model.predict_proba(X_test)[:, 1]
            
            # Calcul du ROC-AUC
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            logger.info(f"ℹ️  ROC-AUC Score: {roc_auc:.4f}")
            
        else:
            logger.warning("⚠️  Model does not support predict_proba. Skipping ROC-AUC calculation.")

        # Calculate classification metrics
        # Since we converted labels to numeric, positive_label should be 1
        final_metrics = model_trainer.evaluator.calculate_metrics(y_test, y_pred, positive_label=1)
        
        # Ajout du ROC-AUC au dictionnaire des métriques finales
        if roc_auc is not None:
            final_metrics['roc_auc'] = roc_auc
        # --- FIN MODIFICATIONS POUR ROC-AUC ---

        # Logging final results
        logger.results_summary({"Final Test Metrics": final_metrics})

        if args.mlflow:
            # Enregistrement des métriques (y compris le ROC-AUC)
            mlflow.log_metrics({f"test_{k}": v for k, v in final_metrics.items()})
            
            # Enregistrement du modèle
            mlflow.sklearn.log_model(model_trainer.best_model, "final_classifier_model")
            
            # Si vous voulez enregistrer le Vectorizer séparément (optionnel)
            # import joblib
            # vectorizer_path = "artifacts/vectorizer.joblib"
            # joblib.dump(text_preprocessor.vectorizer, vectorizer_path)
            # mlflow.log_artifact(vectorizer_path)


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
