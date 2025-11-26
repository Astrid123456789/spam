"""
Utility functions for Spam ML Pipeline.

This module contains helper functions used across the pipeline,
primarily focused on I/O, configuration validation, and plotting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple

from .logger import get_logger, LogLevel


def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame"):
    """Print useful information about a DataFrame - uses new logger."""
    logger = get_logger()
    logger.dataframe_info(df, name)


def print_step_header(step_number: int, step_name: str):
    """Print a formatted step header - uses new logger."""
    logger = get_logger()
    logger.step(step_name, step_number)


def print_func_header(step_name: str):
    """Print a formatted function header - uses new logger."""
    logger = get_logger()
    logger.substep(step_name)


def print_results_summary(results_dict: Dict[str, Any]):
    """Print a formatted summary of results - uses new logger."""
    logger = get_logger()
    logger.results_summary(results_dict)


def validate_data_files():
    """
    Check if required spam data files exist based on configuration.
    
    Raises:
        FileNotFoundError: If any required file is missing.
    """
    # Import config locally to avoid circular dependencies
    from .config import DATA_PATH, SMS_FILE, EMAIL_FILE 

    logger = get_logger()
    
    # Use the simplified file names (SMS_FILE and EMAIL_FILE) for consistency
    sms_path = DATA_PATH / SMS_FILE
    email_path = DATA_PATH / EMAIL_FILE

    if not sms_path.exists():
        raise FileNotFoundError(f"SMS data file not found: {sms_path}")

    if not email_path.exists():
        raise FileNotFoundError(f"Email data file not found: {email_path}")

    logger.success("Spam data files validated and found")
    return True


def setup_plotting():
    """Configure common plot settings for consistent visualization."""
    sns.set_style("whitegrid")
    # Setting the context and font scale for better readability in reports
    sns.set_context("notebook", font_scale=1.1)


def save_plot(filename: str, path: Path, tight_layout: bool = True):
    """
    Save the current matplotlib figure to a specified path.
    
    Args:
        filename: Name of the file (e.g., 'confusion_matrix.png')
        path: Directory where the plot should be saved
        tight_layout: Whether to adjust plot parameters for tight layout
    """
    logger = get_logger()
    
    # Ensure the directory exists
    path.mkdir(parents=True, exist_ok=True)
    full_path = path / filename
    
    if tight_layout:
        plt.tight_layout()
        
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close() # Close the plot to free memory
    
    logger.info(f"Plot saved to: {full_path}", LogLevel.VERBOSE, icon_key='save')


def format_time_elapsed(start_time: float, end_time: float) -> str:
    """
    Format elapsed time in a readable way (e.g., 5.3s or 2m 15.2s).
    
    Args:
        start_time: Start time (e.g., from time.time())
        end_time: End time (e.g., from time.time())
        
    Returns:
        Formatted time string.
    """
    elapsed = end_time - start_time
    if elapsed < 60:
        return f"{elapsed:.1f} seconds"
    else:
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        return f"{minutes}m {seconds:.1f}s"
