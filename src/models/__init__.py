"""Forecasting models."""

from .arima import ARIMAForecaster
from .exponential_smoothing import ExponentialSmoothingForecaster

__all__ = [
    'ARIMAForecaster',
    'ExponentialSmoothingForecaster'
]

# LSTM requires TensorFlow - import conditionally
try:
    from .lstm import LSTMForecaster
    __all__.append('LSTMForecaster')
except ImportError:
    pass
