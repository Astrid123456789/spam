# Spam Detection ML Pipeline

This project implements a complete Machine Learning pipeline for Spam Detection (SMS and Email), featuring modular components for data processing, text vectorization, model training, and evaluation. It is fully integrated with **MLflow** for experiment tracking and model management.

## Project Description

The goal of this project is to classify messages. The pipeline handles:

- **Data Loading**: Supports parsing SMS (semicolon-separated) and Email (comma-separated) datasets.
- **Preprocessing**: Cleaning, de-duplication, class balancing (oversampling), and standardizing text (e.g., `<num>` token).
- **Feature Engineering**: TF-IDF vectorization.
- **Modelling**: Supports multiple classifiers including Logistic Regression, Naive Bayes, and SVM.
- **Evaluation**: Comprehensive metrics (Accuracy, Precision, Recall, F1) and confusion matrix analysis.
- **Orchestration**: A unified script `scripts/run_pipeline.py` to run the entire workflow.

## What Has Been Done

- **Modular Architecture**: Code is organized into `src/pipeline` (components) and `src/utils` (helpers).
- **MLflow Integration**:
  - Full tracking of parameters, metrics, and models.
  - Integration in `ModelTrainer`, `Evaluator`, and `TextPreprocessor`.
  - Experiment management in `run_pipeline.py`.
- **Robustness Improvements**:
  - Fixed data parsing for mixed CSV formats.
  - Added comprehensive unit tests (`tests/`).
  - Standardized dependency management with `requirements.txt` and `uv`.

## How to Run the Pipeline

To run the pipeline with MLflow tracking enabled:

```bash
uv run python scripts/run_pipeline.py --mlflow --model logistic_regression
```

Options:

- `--model`: Choose model type (`logistic_regression`, `naive_bayes`, `linear_svc`).
- `--optimize`: Enable hyperparameter optimization.
- `--compare`: Train and compare multiple models.

## How to Access MLflow UI

To view your experiments, metrics, and trained models:

1. **Launch the MLflow server**:
   Run the following command in the project root (using port 5001 to avoid conflicts):

   ```bash
   uv run mlflow ui --port 5001
   ```

2. **Open the Dashboard**:
   Open your browser and navigate to: [http://127.0.0.1:5001](http://127.0.0.1:5001)

3. **Explore**:
   - Select the experiment `spam_detection_ml`.
   - Click on a "Run Name" to see detailed metrics, parameters, and artifacts (saved models).
