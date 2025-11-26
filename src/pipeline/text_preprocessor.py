import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from ..utils.logger import get_logger

class TextPreprocessor:
    """
    Handles text preprocessing, vectorization, and data balancing for spam detection.
    """
    
    def __init__(self, use_tfidf=False, max_features=5000, token_pattern=r"(<NUM>|[a-z]+|[!?]+|[^\w\s])"):
        """
        Initialize the TextPreprocessor.
        
        Args:
            use_tfidf (bool): If True, use TfidfVectorizer; otherwise use CountVectorizer.
            max_features (int): Maximum number of features to extract.
            token_pattern (str): Regex pattern for tokenization.
        """
        self.logger = get_logger()
        self.use_tfidf = use_tfidf
        self.max_features = max_features
        self.token_pattern = token_pattern
        
        # Initialize vectorizer
        VectorizerClass = TfidfVectorizer if use_tfidf else CountVectorizer
        self.vectorizer = VectorizerClass(
            max_features=max_features,
            token_pattern=token_pattern,
            lowercase=True,
            stop_words='english',
            preprocessor=self.preprocess_text
        )
        
    def preprocess_text(self, message):
        """
        Preprocess a text message by replacing numeric sequences with a generic placeholder
        and converting text to lowercase.
        
        Args:
            message (str): The raw text message.
            
        Returns:
            str: The processed message.
        """
        if not isinstance(message, str):
            return str(message)
            
        message = message.lower()
        # Replace numbers with <NUM>
        message = re.sub(r'\d+', '<NUM>', message)
        # Normalize whitespace
        message = re.sub(r'\s+', ' ', message).strip()
        return message

    def fit(self, X, y=None):
        """
        Fit the vectorizer on the training data.
        
        Args:
            X (iterable): Training text data.
            y (iterable, optional): Target labels (unused for vectorizer fitting).
            
        Returns:
            self
        """
        self.logger.info("Fitting vectorizer...")
        self.vectorizer.fit(X)
        self.logger.success(f"Vectorizer fitted with {len(self.vectorizer.get_feature_names_out())} features")
        return self

    def transform(self, X):
        """
        Transform the data using the fitted vectorizer.
        
        Args:
            X (iterable): Text data to transform.
            
        Returns:
            scipy.sparse.csr_matrix: Transformed feature matrix.
        """
        self.logger.info("Transforming data...")
        return self.vectorizer.transform(X)

    def fit_transform(self, X, y=None):
        """
        Fit and transform the data in one step.
        
        Args:
            X (iterable): Training text data.
            y (iterable, optional): Target labels.
            
        Returns:
            scipy.sparse.csr_matrix: Transformed feature matrix.
        """
        return self.fit(X, y).transform(X)
        
    def balance_data(self, messages, labels, random_state=42):
        """
        Balance training data by oversampling the minority class.
        
        Args:
            messages (pd.Series): Text messages.
            labels (pd.Series): Class labels.
            random_state (int): Random seed for reproducibility.
            
        Returns:
            tuple: (balanced_messages, balanced_labels)
        """
        self.logger.info("Checking class balance...")
        
        # Ensure inputs are pandas Series for easier handling
        if not isinstance(messages, pd.Series):
            messages = pd.Series(messages, name="message")
        if not isinstance(labels, pd.Series):
            labels = pd.Series(labels, name="label")
            
        counts = labels.value_counts()
        self.logger.info(f"Original counts:\n{counts}")
        
        if len(counts) < 2:
            self.logger.warning("Only one class present, cannot balance.")
            return messages, labels
            
        majority_class = counts.idxmax()
        minority_class = counts.idxmin()
        
        if counts[majority_class] == counts[minority_class]:
            self.logger.info("Classes are already balanced.")
            return messages, labels
            
        diff = counts[majority_class] - counts[minority_class]
        self.logger.info(f"Balancing data: Oversampling class {minority_class} by {diff} samples")
        
        # Create a DataFrame to keep messages and labels together during sampling
        df = pd.concat([messages, labels], axis=1)
        minority_df = df[df[labels.name] == minority_class]
        
        # Oversample
        oversampled_minority = minority_df.sample(n=diff, replace=True, random_state=random_state)
        balanced_df = pd.concat([df, oversampled_minority])
        
        # Shuffle the result
        balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        
        new_counts = balanced_df[labels.name].value_counts()
        self.logger.success(f"Balanced counts:\n{new_counts}")
        
        return balanced_df[messages.name], balanced_df[labels.name]
