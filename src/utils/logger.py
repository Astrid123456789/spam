"""
Professional logging system for Spam Detection ML Pipeline.

This module provides a modern, hierarchical logging system with multiple verbosity levels,
professional formatting, and centralized configuration.
"""

import sys
import time
from datetime import datetime
from enum import IntEnum
from typing import Optional, Any, Dict
from contextlib import contextmanager
import pandas as pd


class LogLevel(IntEnum):
    """Logging levels enumeration."""
    SILENT = 0
    NORMAL = 1
    VERBOSE = 2


class PipelineLogger:
    """
    Professional logger for ML pipeline operations.
    
    Features:
    - 3 verbosity levels (SILENT, NORMAL, VERBOSE)
    - Hierarchical indentation
    - Professional formatting with icons and timestamps
    - Context management for nested operations
    - Performance timing
    """
    
    def __init__(self, level: LogLevel = LogLevel.NORMAL):
        """
        Initialize the logger.
        
        Args:
            level: Logging level (SILENT, NORMAL, VERBOSE)
        """
        self.level = level
        self._indent_level = 0
        self._start_times = {}
        
        # Icons for different message types
        self.icons = {
            'start': '🚀',
            'step': '📊',
            'substep': '├──',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌',
            'info': 'ℹ️ ',
            'feature': '🔧',
            'model': '🎯',
            'data': '📈',
            'save': '💾',
            'load': '📂',
            'end': '🏁'
        }
        
    def set_level(self, level: LogLevel):
        """Set the current logging level."""
        self.level = level
        
    def _get_timestamp(self) -> str:
        """Get formatted timestamp."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def _get_indent(self) -> str:
        """Get current indentation string."""
        # Use 4 spaces per level
        return "    " * self._indent_level

    def _format_time_elapsed(self, total_time: float) -> str:
        """Format elapsed time in a human-readable format."""
        if total_time < 60:
            return f"{total_time:.2f}s"
        minutes = int(total_time // 60)
        seconds = total_time % 60
        return f"{minutes}m {seconds:.2f}s"

    def _log(self, message: str, level: LogLevel = LogLevel.NORMAL, icon_key: str = 'info'):
        """Core logging function."""
        if self.level >= level:
            icon = self.icons.get(icon_key, ' ')
            indent = self._get_indent()
            # Timestamp is only added for major steps or in verbose mode
            timestamp_str = f"[{self._get_timestamp()}] " if self.level == LogLevel.VERBOSE else ""
            print(f"{timestamp_str}{indent}{icon} {message}")

    # --- Context Managers for Structure and Timing ---

    @contextmanager
    def indent(self):
        """Context manager to increase and decrease indentation level."""
        self._indent_level += 1
        try:
            yield
        finally:
            self._indent_level -= 1

    @contextmanager
    def time_block(self, block_name: str, level: LogLevel = LogLevel.NORMAL):
        """Context manager to time a block of code."""
        if self.level >= level:
            start_time = time.time()
            self.substep(f"Starting {block_name}...", level)
        try:
            yield
        finally:
            if self.level >= level:
                end_time = time.time()
                elapsed_time = end_time - start_time
                time_str = self._format_time_elapsed(elapsed_time)
                self.success(f"{block_name} completed in {time_str}", level)
    
    # --- High-Level Logging Methods ---

    def step(self, step_name: str, step_number: Optional[int] = None, total_steps: Optional[int] = None):
        """Log a major step in the pipeline (always visible)."""
        step_info = f"STEP {step_number}/{total_steps}: " if step_number and total_steps else f"STEP {step_number}: " if step_number else ""
        print(f"\n{self.icons['step']} {'='*10} {step_info}{step_name.upper()} {'='*10}")

    def substep(self, message: str, level: LogLevel = LogLevel.NORMAL):
        """Log a minor step or function call."""
        self._log(message, level, 'substep')

    def info(self, message: str, level: LogLevel = LogLevel.NORMAL):
        """Log general information."""
        self._log(message, level, 'info')

    def warning(self, message: str, level: LogLevel = LogLevel.NORMAL):
        """Log a non-critical warning."""
        self._log(message, level, 'warning')

    def error(self, message: str, level: LogLevel = LogLevel.NORMAL):
        """Log a critical error."""
        self._log(message, level, 'error')

    def success(self, message: str, level: LogLevel = LogLevel.NORMAL):
        """Log a successful completion."""
        self._log(message, level, 'success')

    # --- Specific Logging Methods ---

    def dataframe_info(self, df: pd.DataFrame, name: str):
        """Log useful information about a DataFrame (VERBOSE only)."""
        if self.level >= LogLevel.VERBOSE:
            self.info(f"DataFrame: {name} ({len(df)} rows, {len(df.columns)} cols)", LogLevel.VERBOSE, 'data')
            with self.indent():
                # Display basic info
                df_info = {
                    'Shape': str(df.shape),
                    'Missing Data': f"{df.isnull().sum().sum()} total cells",
                    'Duplicate Rows': df.duplicated().sum(),
                    'Dtypes': ', '.join(df.dtypes.astype(str).unique())
                }
                for key, value in df_info.items():
                    self._log(f"- {key}: {value}", LogLevel.VERBOSE)
            
    def results_summary(self, results_dict: Dict[str, Any]):
        """Log a formatted summary of evaluation results."""
        self.info("Results Summary:")
        with self.indent():
            for key, value in results_dict.items():
                if isinstance(value, float):
                    self._log(f"- {key.title()}: {value:.4f}")
                elif isinstance(value, dict):
                    self._log(f"- {key.title()}:")
                    with self.indent():
                        for sub_key, sub_value in value.items():
                             self._log(f"  {sub_key}: {sub_value:.4f}" if isinstance(sub_value, (float, int)) else f"  {sub_key}: {sub_value}")
                else:
                    self._log(f"- {key.title()}: {value}")

    def feature_info(self, message: str):
        """Log information related to feature engineering (VERBOSE only)."""
        self._log(message, LogLevel.VERBOSE, 'feature')
        
    def model_info(self, message: str):
        """Log information related to model training/configuration."""
        self._log(message, LogLevel.NORMAL, 'model')

    def final_success(self, total_time: Optional[float] = None):
        """Log the final success message with total time."""
        message = "Pipeline completed successfully!"
        if total_time:
            time_str = self._format_time_elapsed(total_time)
            message += f" (Total time: {time_str})"
        
        print(f"\n{self.icons['end']} [{self._get_timestamp()}] {message}")


# =============================================================================
# GLOBAL INSTANCE AND HELPERS
# =============================================================================

# Global logger instance
_logger = PipelineLogger()


def get_logger() -> PipelineLogger:
    """Get the global logger instance."""
    return _logger


def set_log_level(level: LogLevel):
    """Set the global logging level."""
    _logger.set_level(level)


def log_level_from_string(level_str: str) -> LogLevel:
    """Convert string to LogLevel enum."""
    level_map = {
        'silent': LogLevel.SILENT,
        'normal': LogLevel.NORMAL,
        'verbose': LogLevel.VERBOSE
    }
    
    level_str = level_str.lower()
    if level_str not in level_map:
        raise ValueError(f"Invalid log level: {level_str}. Available: {list(level_map.keys())}")
    
    return level_map[level_str]
