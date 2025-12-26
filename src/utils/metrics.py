"""
Evaluation metrics for time series forecasting.

Provides comprehensive metrics including MAE, RMSE, MAPE, SMAPE,
MASE, and coverage probability for confidence intervals.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Optional


def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    MAE = (1/n) * Σ|actual - predicted|
    
    Args:
        actual: Actual values
        predicted: Predicted values
    
    Returns:
        MAE value
    """
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()
    return np.mean(np.abs(actual - predicted))


def mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Squared Error (MSE).
    
    MSE = (1/n) * Σ(actual - predicted)²
    
    Args:
        actual: Actual values
        predicted: Predicted values
    
    Returns:
        MSE value
    """
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()
    return np.mean((actual - predicted) ** 2)


def root_mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).
    
    RMSE = √MSE
    
    Args:
        actual: Actual values
        predicted: Predicted values
    
    Returns:
        RMSE value
    """
    return np.sqrt(mean_squared_error(actual, predicted))


def mean_absolute_percentage_error(actual: np.ndarray, 
                                   predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    MAPE = (100/n) * Σ|actual - predicted| / |actual|
    
    Note: Returns inf if any actual value is zero.
    
    Args:
        actual: Actual values
        predicted: Predicted values
    
    Returns:
        MAPE value (as percentage)
    """
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()
    
    # Avoid division by zero
    mask = actual != 0
    if not mask.any():
        return np.inf
    
    return 100 * np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask]))


def symmetric_mean_absolute_percentage_error(actual: np.ndarray,
                                              predicted: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    
    SMAPE = (100/n) * Σ|actual - predicted| / ((|actual| + |predicted|) / 2)
    
    SMAPE is more robust than MAPE when actual values are near zero.
    
    Args:
        actual: Actual values
        predicted: Predicted values
    
    Returns:
        SMAPE value (as percentage)
    """
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()
    
    numerator = np.abs(actual - predicted)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    
    # Handle zero denominators
    mask = denominator != 0
    if not mask.any():
        return 0.0
    
    return 100 * np.mean(numerator[mask] / denominator[mask])


def mean_absolute_scaled_error(actual: np.ndarray,
                                predicted: np.ndarray,
                                training_series: np.ndarray = None,
                                seasonality: int = 1) -> float:
    """
    Calculate Mean Absolute Scaled Error (MASE).
    
    MASE compares forecast errors to naive forecast errors.
    MASE < 1 indicates the forecast is better than naive.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        training_series: Training data for calculating naive errors
        seasonality: Seasonal period (1 for non-seasonal naive)
    
    Returns:
        MASE value
    """
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()
    
    if training_series is None:
        training_series = actual
    else:
        training_series = np.asarray(training_series).flatten()
    
    # Calculate naive forecast errors on training data
    naive_errors = np.abs(training_series[seasonality:] - 
                          training_series[:-seasonality])
    scale = np.mean(naive_errors)
    
    if scale == 0:
        return np.inf
    
    # Calculate forecast errors
    forecast_errors = np.abs(actual - predicted)
    
    return np.mean(forecast_errors) / scale


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination).
    
    R² = 1 - SS_res / SS_tot
    
    Args:
        actual: Actual values
        predicted: Predicted values
    
    Returns:
        R² value (between 0 and 1 for good models)
    """
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()
    
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)


def coverage_probability(actual: np.ndarray,
                         lower: np.ndarray,
                         upper: np.ndarray) -> float:
    """
    Calculate coverage probability for prediction intervals.
    
    Returns the fraction of actual values that fall within
    the prediction intervals.
    
    Args:
        actual: Actual values
        lower: Lower bound of prediction interval
        upper: Upper bound of prediction interval
    
    Returns:
        Coverage probability (0 to 1)
    """
    actual = np.asarray(actual).flatten()
    lower = np.asarray(lower).flatten()
    upper = np.asarray(upper).flatten()
    
    within = (actual >= lower) & (actual <= upper)
    return np.mean(within)


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Calculate average width of prediction intervals.
    
    Args:
        lower: Lower bound of prediction interval
        upper: Upper bound of prediction interval
    
    Returns:
        Average interval width
    """
    lower = np.asarray(lower).flatten()
    upper = np.asarray(upper).flatten()
    return np.mean(upper - lower)


def evaluate_forecast(actual: Union[np.ndarray, pd.Series],
                      predicted: Union[np.ndarray, pd.Series],
                      conf_int: pd.DataFrame = None,
                      training_series: np.ndarray = None) -> Dict[str, float]:
    """
    Compute comprehensive forecast evaluation metrics.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        conf_int: DataFrame with 'lower' and 'upper' columns for CI
        training_series: Training data for MASE calculation
    
    Returns:
        Dictionary with all computed metrics
    """
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()
    
    metrics = {
        'mae': mean_absolute_error(actual, predicted),
        'mse': mean_squared_error(actual, predicted),
        'rmse': root_mean_squared_error(actual, predicted),
        'mape': mean_absolute_percentage_error(actual, predicted),
        'smape': symmetric_mean_absolute_percentage_error(actual, predicted),
        'r2': r_squared(actual, predicted)
    }
    
    if training_series is not None:
        metrics['mase'] = mean_absolute_scaled_error(
            actual, predicted, training_series
        )
    
    if conf_int is not None:
        metrics['coverage'] = coverage_probability(
            actual, conf_int['lower'].values, conf_int['upper'].values
        )
        metrics['interval_width'] = interval_width(
            conf_int['lower'].values, conf_int['upper'].values
        )
    
    return metrics


def print_metrics(metrics: Dict[str, float], 
                  precision: int = 4) -> str:
    """
    Format metrics as a readable string.
    
    Args:
        metrics: Dictionary of metric values
        precision: Decimal places to show
    
    Returns:
        Formatted string
    """
    lines = []
    for name, value in metrics.items():
        if name in ['mape', 'smape', 'coverage']:
            lines.append(f"{name.upper():>15}: {value:.{precision-2}f}%")
        else:
            lines.append(f"{name.upper():>15}: {value:.{precision}f}")
    
    return '\n'.join(lines)
