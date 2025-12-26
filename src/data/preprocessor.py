"""
Time series preprocessing utilities.

Handles missing values, outliers, scaling, and feature engineering
for time series data. Designed to handle real-world data challenges.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging

logger = logging.getLogger(__name__)


class TimeSeriesPreprocessor:
    """
    Comprehensive preprocessing pipeline for time series data.
    
    Handles common data quality issues including missing values,
    outliers, and applies scaling transformations. Maintains the
    ability to inverse transform for interpretable forecasts.
    
    Attributes:
        missing_method: Method used for handling missing values
        outlier_method: Method used for outlier detection
        scaler: Fitted scaler object for inverse transformations
    
    Example:
        >>> preprocessor = TimeSeriesPreprocessor(
        ...     handle_missing='interpolate',
        ...     outlier_method='iqr',
        ...     scaling='minmax'
        ... )
        >>> clean_data = preprocessor.fit_transform(raw_data)
        >>> original_scale = preprocessor.inverse_transform(predictions)
    """
    
    def __init__(self,
                 handle_missing: str = 'interpolate',
                 outlier_method: str = 'iqr',
                 outlier_threshold: float = 1.5,
                 scaling: str = None,
                 clip_outliers: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            handle_missing: Method for missing values 
                           ('interpolate', 'ffill', 'bfill', 'drop', 'mean')
            outlier_method: Method for outlier detection ('iqr', 'zscore', None)
            outlier_threshold: Threshold for outlier detection
                              (1.5 for IQR, 3 for z-score)
            scaling: Scaling method ('minmax', 'standard', None)
            clip_outliers: Whether to clip outliers instead of removing
        """
        self.missing_method = handle_missing
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.scaling = scaling
        self.clip_outliers = clip_outliers
        
        self.scaler = None
        self._fitted = False
        self._original_index = None
        self._stats = {}
    
    def fit(self, data: Union[pd.Series, pd.DataFrame]) -> 'TimeSeriesPreprocessor':
        """
        Fit the preprocessor to the data.
        
        Computes statistics needed for transformation without
        modifying the input data.
        
        Args:
            data: Time series data to fit on
        
        Returns:
            self
        """
        series = self._to_series(data)
        
        # Store original statistics
        self._stats = {
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'q1': series.quantile(0.25),
            'q3': series.quantile(0.75)
        }
        
        # Compute IQR bounds
        iqr = self._stats['q3'] - self._stats['q1']
        self._stats['lower_bound'] = self._stats['q1'] - self.outlier_threshold * iqr
        self._stats['upper_bound'] = self._stats['q3'] + self.outlier_threshold * iqr
        
        # Fit scaler if needed
        if self.scaling:
            if self.scaling == 'minmax':
                self.scaler = MinMaxScaler()
            elif self.scaling == 'standard':
                self.scaler = StandardScaler()
            else:
                raise ValueError(f"Unknown scaling method: {self.scaling}")
            
            # Fit on non-null values
            valid_data = series.dropna().values.reshape(-1, 1)
            self.scaler.fit(valid_data)
        
        self._fitted = True
        return self
    
    def transform(self, data: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        """
        Apply preprocessing transformations.
        
        Args:
            data: Time series data to transform
        
        Returns:
            Preprocessed time series as a pandas Series
        """
        if not self._fitted:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        series = self._to_series(data).copy()
        self._original_index = series.index
        
        # Step 1: Handle missing values
        series = self._handle_missing(series)
        
        # Step 2: Handle outliers
        if self.outlier_method:
            series = self._handle_outliers(series)
        
        # Step 3: Apply scaling
        if self.scaler:
            values = series.values.reshape(-1, 1)
            scaled = self.scaler.transform(values)
            series = pd.Series(scaled.flatten(), index=series.index, name=series.name)
        
        return series
    
    def fit_transform(self, data: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data: Union[pd.Series, np.ndarray]) -> pd.Series:
        """
        Reverse the scaling transformation.
        
        Args:
            data: Scaled data to inverse transform
        
        Returns:
            Data in original scale
        """
        if self.scaler is None:
            if isinstance(data, pd.Series):
                return data
            return pd.Series(data)
        
        if isinstance(data, pd.Series):
            values = data.values.reshape(-1, 1)
            index = data.index
        else:
            values = np.array(data).reshape(-1, 1)
            index = None
        
        original = self.scaler.inverse_transform(values).flatten()
        
        if index is not None:
            return pd.Series(original, index=index)
        return pd.Series(original)
    
    def _to_series(self, data: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        """Convert input to pandas Series."""
        if isinstance(data, pd.DataFrame):
            if len(data.columns) == 1:
                return data.iloc[:, 0]
            raise ValueError("DataFrame has multiple columns. Specify target column.")
        return data
    
    def _handle_missing(self, series: pd.Series) -> pd.Series:
        """Handle missing values based on configured method."""
        n_missing = series.isnull().sum()
        
        if n_missing == 0:
            return series
        
        logger.info(f"Handling {n_missing} missing values using '{self.missing_method}'")
        
        if self.missing_method == 'interpolate':
            # Time-based interpolation works well for regular time series
            series = series.interpolate(method='time')
            # Handle any remaining NaNs at edges
            series = series.fillna(method='bfill').fillna(method='ffill')
            
        elif self.missing_method == 'ffill':
            series = series.fillna(method='ffill')
            
        elif self.missing_method == 'bfill':
            series = series.fillna(method='bfill')
            
        elif self.missing_method == 'mean':
            series = series.fillna(self._stats['mean'])
            
        elif self.missing_method == 'drop':
            series = series.dropna()
            
        else:
            raise ValueError(f"Unknown missing method: {self.missing_method}")
        
        return series
    
    def _handle_outliers(self, series: pd.Series) -> pd.Series:
        """Detect and handle outliers."""
        if self.outlier_method == 'iqr':
            lower = self._stats['lower_bound']
            upper = self._stats['upper_bound']
            outliers = (series < lower) | (series > upper)
            
        elif self.outlier_method == 'zscore':
            z_scores = np.abs((series - self._stats['mean']) / self._stats['std'])
            outliers = z_scores > self.outlier_threshold
            lower = self._stats['mean'] - self.outlier_threshold * self._stats['std']
            upper = self._stats['mean'] + self.outlier_threshold * self._stats['std']
            
        else:
            return series
        
        n_outliers = outliers.sum()
        
        if n_outliers > 0:
            logger.info(f"Detected {n_outliers} outliers using '{self.outlier_method}'")
            
            if self.clip_outliers:
                series = series.clip(lower=lower, upper=upper)
            else:
                # Replace with interpolated values
                series[outliers] = np.nan
                series = series.interpolate(method='time')
        
        return series
    
    def get_stats(self) -> dict:
        """Return computed statistics."""
        return self._stats.copy()


def create_features(series: pd.Series, 
                   lags: List[int] = None,
                   rolling_windows: List[int] = None) -> pd.DataFrame:
    """
    Create time series features for ML models.
    
    Args:
        series: Input time series
        lags: List of lag values to create (e.g., [1, 24, 168] for hourly data)
        rolling_windows: List of rolling window sizes for moving averages
    
    Returns:
        DataFrame with original series and engineered features
    """
    df = pd.DataFrame(index=series.index)
    df['value'] = series
    
    # Lag features
    if lags:
        for lag in lags:
            df[f'lag_{lag}'] = series.shift(lag)
    
    # Rolling statistics
    if rolling_windows:
        for window in rolling_windows:
            df[f'rolling_mean_{window}'] = series.rolling(window=window).mean()
            df[f'rolling_std_{window}'] = series.rolling(window=window).std()
    
    # Time-based features (if datetime index)
    if isinstance(series.index, pd.DatetimeIndex):
        df['hour'] = series.index.hour
        df['dayofweek'] = series.index.dayofweek
        df['month'] = series.index.month
        df['is_weekend'] = (series.index.dayofweek >= 5).astype(int)
    
    return df


def train_test_split(data: Union[pd.Series, pd.DataFrame],
                     test_size: float = 0.2) -> Tuple:
    """
    Split time series data chronologically.
    
    Unlike sklearn's train_test_split, this preserves temporal order
    which is essential for time series data.
    
    Args:
        data: Time series data to split
        test_size: Fraction of data for testing
    
    Returns:
        Tuple of (train, test) data
    """
    split_idx = int(len(data) * (1 - test_size))
    
    if isinstance(data, pd.DataFrame):
        return data.iloc[:split_idx], data.iloc[split_idx:]
    
    return data.iloc[:split_idx], data.iloc[split_idx:]
