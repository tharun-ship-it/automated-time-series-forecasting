"""
LSTM-based deep learning forecaster.

Implements a stacked LSTM architecture for time series forecasting
with support for confidence interval estimation via Monte Carlo dropout.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import warnings
import logging

logger = logging.getLogger(__name__)

# TensorFlow import with graceful fallback
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. LSTM model will not work.")


class LSTMForecaster:
    """
    LSTM neural network for time series forecasting.
    
    Uses a stacked LSTM architecture with dropout regularization.
    Supports Monte Carlo dropout for uncertainty estimation.
    
    Attributes:
        lookback: Number of timesteps to look back
        units: List of LSTM units for each layer
        model: Compiled Keras model
    
    Example:
        >>> model = LSTMForecaster(lookback=24, units=[64, 32])
        >>> model.fit(train_data, epochs=50)
        >>> forecast, ci = model.predict(steps=24, return_conf_int=True)
    """
    
    def __init__(self,
                 lookback: int = 24,
                 units: List[int] = None,
                 dropout: float = 0.2,
                 learning_rate: float = 0.001,
                 batch_size: int = 32):
        """
        Initialize the LSTM forecaster.
        
        Args:
            lookback: Number of past timesteps to use as input
            units: List of units for each LSTM layer (default: [64, 32])
            dropout: Dropout rate for regularization
            learning_rate: Learning rate for Adam optimizer
            batch_size: Batch size for training
        """
        if not TF_AVAILABLE:
            raise ImportError(
                "TensorFlow is required for LSTM. "
                "Install with: pip install tensorflow"
            )
        
        self.lookback = lookback
        self.units = units or [64, 32]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.model = None
        self.scaler_min = None
        self.scaler_max = None
        self._train_data = None
        self._history = None
    
    def _build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build the LSTM model architecture."""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            self.units[0],
            input_shape=input_shape,
            return_sequences=len(self.units) > 1
        ))
        model.add(Dropout(self.dropout))
        
        # Additional LSTM layers
        for i, units in enumerate(self.units[1:]):
            return_seq = i < len(self.units) - 2
            model.add(LSTM(units, return_sequences=return_seq))
            model.add(Dropout(self.dropout))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _create_sequences(self, 
                          data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences for LSTM."""
        X, y = [], []
        
        for i in range(len(data) - self.lookback):
            X.append(data[i:(i + self.lookback)])
            y.append(data[i + self.lookback])
        
        return np.array(X), np.array(y)
    
    def _scale_data(self, data: np.ndarray) -> np.ndarray:
        """Min-max scale the data to [0, 1]."""
        self.scaler_min = data.min()
        self.scaler_max = data.max()
        return (data - self.scaler_min) / (self.scaler_max - self.scaler_min)
    
    def _inverse_scale(self, data: np.ndarray) -> np.ndarray:
        """Inverse the scaling transformation."""
        return data * (self.scaler_max - self.scaler_min) + self.scaler_min
    
    def fit(self,
            data: pd.Series,
            epochs: int = 100,
            validation_split: float = 0.1,
            verbose: int = 0) -> 'LSTMForecaster':
        """
        Fit the LSTM model.
        
        Args:
            data: Time series data to fit
            epochs: Maximum number of training epochs
            validation_split: Fraction of data for validation
            verbose: Verbosity level (0, 1, or 2)
        
        Returns:
            self
        """
        self._train_data = data.copy()
        
        # Prepare data
        values = data.values.astype(np.float32)
        scaled = self._scale_data(values)
        X, y = self._create_sequences(scaled)
        
        # Reshape for LSTM [samples, timesteps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build model
        self.model = self._build_model(input_shape=(self.lookback, 1))
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train
        logger.info(f"Training LSTM with {len(X)} samples...")
        
        self._history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        final_loss = self._history.history['loss'][-1]
        logger.info(f"Training complete. Final loss: {final_loss:.6f}")
        
        return self
    
    def predict(self,
                steps: int = 10,
                return_conf_int: bool = False,
                n_simulations: int = 100,
                alpha: float = 0.05) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """
        Generate forecasts.
        
        Uses iterative prediction where each forecast becomes
        input for the next step.
        
        Args:
            steps: Number of steps to forecast
            return_conf_int: Whether to return confidence intervals
            n_simulations: Number of Monte Carlo simulations for CI
            alpha: Significance level for confidence intervals
        
        Returns:
            Tuple of (forecast, confidence_intervals)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Prepare the last sequence
        values = self._train_data.values.astype(np.float32)
        scaled = (values - self.scaler_min) / (self.scaler_max - self.scaler_min)
        current_sequence = scaled[-self.lookback:].copy()
        
        # Generate forecasts
        predictions = []
        
        for _ in range(steps):
            # Reshape for prediction
            X = current_sequence.reshape((1, self.lookback, 1))
            
            # Predict
            pred = self.model.predict(X, verbose=0)[0, 0]
            predictions.append(pred)
            
            # Update sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred
        
        # Inverse scale
        predictions = np.array(predictions)
        forecast = self._inverse_scale(predictions)
        
        # Create index
        last_date = self._train_data.index[-1]
        freq = pd.infer_freq(self._train_data.index)
        if freq is None:
            freq = 'H'
        
        future_index = pd.date_range(
            start=last_date + pd.Timedelta(1, unit=freq[0].lower() if freq else 'h'),
            periods=steps,
            freq=freq
        )
        
        forecast_series = pd.Series(forecast, index=future_index)
        
        if return_conf_int:
            ci = self._monte_carlo_ci(steps, n_simulations, alpha, future_index)
            return forecast_series, ci
        
        return forecast_series, None
    
    def _monte_carlo_ci(self,
                        steps: int,
                        n_simulations: int,
                        alpha: float,
                        index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Estimate confidence intervals using Monte Carlo dropout.
        
        Enables dropout during prediction to get uncertainty estimates.
        """
        values = self._train_data.values.astype(np.float32)
        scaled = (values - self.scaler_min) / (self.scaler_max - self.scaler_min)
        
        all_predictions = []
        
        # Run multiple forward passes with dropout enabled
        for _ in range(n_simulations):
            current_sequence = scaled[-self.lookback:].copy()
            sim_predictions = []
            
            for _ in range(steps):
                X = current_sequence.reshape((1, self.lookback, 1))
                
                # Predict with training=True to enable dropout
                pred = self.model(X, training=True).numpy()[0, 0]
                
                # Add small noise for uncertainty
                pred += np.random.normal(0, 0.01)
                
                sim_predictions.append(pred)
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = pred
            
            all_predictions.append(self._inverse_scale(np.array(sim_predictions)))
        
        all_predictions = np.array(all_predictions)
        
        # Calculate percentiles
        lower_q = (alpha / 2) * 100
        upper_q = (1 - alpha / 2) * 100
        
        lower = np.percentile(all_predictions, lower_q, axis=0)
        upper = np.percentile(all_predictions, upper_q, axis=0)
        
        return pd.DataFrame({
            'lower': lower,
            'upper': upper
        }, index=index)
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get model diagnostic information."""
        if self.model is None:
            return {'status': 'Model not fitted'}
        
        history = self._history.history
        
        return {
            'lookback': self.lookback,
            'architecture': self.units,
            'dropout': self.dropout,
            'final_loss': history['loss'][-1],
            'final_val_loss': history.get('val_loss', [None])[-1],
            'epochs_trained': len(history['loss']),
            'total_params': self.model.count_params()
        }
