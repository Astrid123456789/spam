"""
Configuration file for pytest. Defines fixtures shared across all test modules.
"""

import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

# Assurez-vous que ces imports fonctionnent depuis la racine du projet
from src.utils.logger import get_logger, LogLevel


# --- 1. Fixture d'environnement (Silencing Logger) ---

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """
    Configure l'environnement de test :
    Met le niveau de logging à SILENT pour éviter de polluer la console avec les logs du pipeline.
    """
    logger = get_logger()
    original_level = logger.level
    
    # Définir le niveau de logging à SILENT
    logger.set_level(LogLevel.SILENT)
    
    # Exécuter les tests
    yield
    
    # Teardown: restaurer le niveau de logging après tous les tests
    logger.set_level(original_level)


# --- 2. Fixture de Données de Classification (X, y) ---

@pytest.fixture(scope="session")
def sample_X_y():
    """
    Crée une petite matrice d'attributs sparse (X) et une cible binaire (y) 
    simulées, adaptées aux tests des modèles de classification (ModelTrainer, Evaluator).
    """
    # X: Matrice sparse (10 features, 5 échantillons) simulant le TF-IDF/BoW
    X_data = np.array([
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
        [1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    ])
    X = csr_matrix(X_data)
    
    # y: Cible binaire pour la classification (0=ham, 1=spam)
    y = np.array([0, 1, 0, 1, 0])
    
    return X, y


# --- 3. Fixture de Données Brutes (DataProcessor) ---

@pytest.fixture(scope="session")
def sample_raw_data_df():
    """
    Crée un petit DataFrame brut simulant la structure des données SMS/Email 
    avant le traitement par DataProcessor.
    """
    
    data = {
        'message': [
            "free mobile phone, click link now!", 
            "hey, let's meet tomorrow?", 
            "winner! claim prize. code: XXX", 
            "lunch at the office",
            "Urgent: money transfer needed.",
            "meeting moved to 2pm"
        ],
        'target': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham'],
        # Colonnes pour tester le nettoyage (missing/duplicates)
        'extra_col_1': [1, 2, np.nan, 4, 5, 6],
        'high_missing_col': [np.nan] * 5 + [1]
    }
    df = pd.DataFrame(data)
    
    return df
