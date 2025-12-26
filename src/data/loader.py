"""
Data loading utilities for time series analysis.

This module provides a unified interface for loading time series data
from various sources including CSV files, APIs, and databases.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Unified data loader for time series data.
    
    Supports loading from CSV files, with automatic datetime parsing
    and basic validation. Designed to handle energy consumption data
    and similar time series datasets.
    
    Attributes:
        data: Loaded DataFrame with datetime index
        target_col: Name of the target variable column
        freq: Detected or specified frequency of the time series
    
    Example:
        >>> loader = DataLoader()
        >>> data = loader.load_csv('data/PJME_hourly.csv',
        ...                        datetime_col='Datetime',
        ...                        target_col='PJME_MW')
        >>> print(f"Loaded {len(data)} records")
    """
    
    def __init__(self):
        self.data = None
        self.target_col = None
        self.freq = None
        self._raw_data = None
    
    def load_csv(self, 
                 filepath: str,
                 datetime_col: str = 'Datetime',
                 target_col: str = None,
                 parse_dates: bool = True,
                 freq: str = None) -> pd.DataFrame:
        """
        Load time series data from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            datetime_col: Name of the datetime column
            target_col: Name of the target variable (optional)
            parse_dates: Whether to parse dates automatically
            freq: Frequency string (e.g., 'H' for hourly, 'D' for daily)
        
        Returns:
            DataFrame with datetime index
        
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If datetime column is not found
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        logger.info(f"Loading data from {filepath}")
        
        # Load the CSV
        df = pd.read_csv(filepath)
        self._raw_data = df.copy()
        
        # Parse datetime column
        if datetime_col not in df.columns:
            raise ValueError(f"Datetime column '{datetime_col}' not found. "
                           f"Available columns: {list(df.columns)}")
        
        if parse_dates:
            df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        # Set datetime as index
        df.set_index(datetime_col, inplace=True)
        df.sort_index(inplace=True)
        
        # Detect frequency if not provided
        if freq is None:
            freq = pd.infer_freq(df.index)
            if freq:
                logger.info(f"Detected frequency: {freq}")
        
        self.freq = freq
        self.target_col = target_col
        self.data = df
        
        logger.info(f"Loaded {len(df)} records from {df.index.min()} to {df.index.max()}")
        
        return df
    
    def load_from_kaggle(self, dataset_name: str, filename: str) -> pd.DataFrame:
        """
        Load data directly from Kaggle dataset.
        
        Note: Requires kaggle API credentials to be configured.
        
        Args:
            dataset_name: Kaggle dataset identifier (e.g., 'robikscube/hourly-energy-consumption')
            filename: Name of the file to load from the dataset
        
        Returns:
            DataFrame with the loaded data
        """
        try:
            import kaggle
            kaggle.api.dataset_download_file(dataset_name, filename, path='data/')
            return self.load_csv(f'data/{filename}')
        except ImportError:
            logger.warning("Kaggle package not installed. Install with: pip install kaggle")
            raise
    
    def get_target_series(self) -> pd.Series:
        """Extract the target variable as a Series."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")
        
        if self.target_col is None:
            # Return first numeric column if target not specified
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found in data")
            return self.data[numeric_cols[0]]
        
        return self.data[self.target_col]
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate summary statistics for the loaded data."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")
        
        series = self.get_target_series()
        
        return {
            'n_records': len(self.data),
            'start_date': self.data.index.min(),
            'end_date': self.data.index.max(),
            'frequency': self.freq,
            'missing_values': self.data.isnull().sum().sum(),
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'median': series.median()
        }
    
    def validate(self) -> Dict[str, bool]:
        """
        Run validation checks on the loaded data.
        
        Returns:
            Dictionary with validation results
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")
        
        series = self.get_target_series()
        
        checks = {
            'has_data': len(self.data) > 0,
            'datetime_index': isinstance(self.data.index, pd.DatetimeIndex),
            'sorted_index': self.data.index.is_monotonic_increasing,
            'no_duplicates': not self.data.index.duplicated().any(),
            'no_missing_target': not series.isnull().any(),
            'positive_values': (series >= 0).all() if not series.isnull().any() else False,
            'sufficient_length': len(self.data) >= 100
        }
        
        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            logger.info(f"{status} {check}: {passed}")
        
        return checks
    
    def train_test_split(self, 
                         test_size: float = 0.2,
                         by_date: str = None) -> tuple:
        """
        Split data into training and test sets.
        
        For time series, we always split chronologically (not randomly).
        
        Args:
            test_size: Fraction of data to use for testing (default 0.2)
            by_date: Optional date string to split on (e.g., '2017-01-01')
        
        Returns:
            Tuple of (train_data, test_data) DataFrames
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv() first.")
        
        if by_date:
            split_date = pd.to_datetime(by_date)
            train = self.data[self.data.index < split_date]
            test = self.data[self.data.index >= split_date]
        else:
            split_idx = int(len(self.data) * (1 - test_size))
            train = self.data.iloc[:split_idx]
            test = self.data.iloc[split_idx:]
        
        logger.info(f"Train set: {len(train)} samples ({train.index.min()} to {train.index.max()})")
        logger.info(f"Test set: {len(test)} samples ({test.index.min()} to {test.index.max()})")
        
        return train, test


def load_energy_data(filepath: str = 'data/PJME_hourly.csv') -> pd.DataFrame:
    """
    Convenience function to load the PJM energy consumption dataset.
    
    Args:
        filepath: Path to the PJME_hourly.csv file
    
    Returns:
        DataFrame with datetime index and PJME_MW column
    """
    loader = DataLoader()
    return loader.load_csv(filepath, 
                          datetime_col='Datetime',
                          target_col='PJME_MW')
