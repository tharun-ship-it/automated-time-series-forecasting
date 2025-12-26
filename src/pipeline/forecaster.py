"""
Automated time series forecasting pipeline.

Orchestrates the entire workflow from data loading through
model training, ensemble prediction, and result generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import yaml
import logging
import warnings

from ..data.loader import DataLoader
from ..data.preprocessor import TimeSeriesPreprocessor, train_test_split
from ..models.arima import ARIMAForecaster
from ..models.exponential_smoothing import ExponentialSmoothingForecaster
from ..utils.metrics import evaluate_forecast

logger = logging.getLogger(__name__)


class TimeSeriesForecaster:
    """
    End-to-end automated time series forecasting pipeline.
    
    Handles the complete workflow including data loading, preprocessing,
    training multiple models, generating ensemble predictions, and
    producing results.
    
    Attributes:
        config: Pipeline configuration dictionary
        models: Dictionary of fitted model objects
        results: Dictionary containing forecasts and metrics
    
    Example:
        >>> forecaster = TimeSeriesForecaster(config_path='config/config.yaml')
        >>> results = forecaster.run()
        >>> print(results['metrics'])
    """
    
    AVAILABLE_MODELS = ['arima', 'exp_smoothing', 'lstm']
    
    def __init__(self,
                 config_path: str = None,
                 horizon: int = 24,
                 models: List[str] = None,
                 ensemble: bool = True,
                 ensemble_method: str = 'weighted',
                 preprocessing_config: Dict = None):
        """
        Initialize the forecasting pipeline.
        
        Args:
            config_path: Path to YAML configuration file
            horizon: Forecast horizon (number of steps)
            models: List of models to use ('arima', 'exp_smoothing', 'lstm')
            ensemble: Whether to create ensemble predictions
            ensemble_method: Method for combining forecasts ('mean', 'weighted')
            preprocessing_config: Dictionary with preprocessing settings
        """
        self.config = self._load_config(config_path) if config_path else {}
        
        # Override with explicit parameters
        self.horizon = horizon
        self.model_names = models or ['arima', 'exp_smoothing']
        self.ensemble = ensemble
        self.ensemble_method = ensemble_method
        self.preprocessing_config = preprocessing_config or {
            'handle_missing': 'interpolate',
            'outlier_method': 'iqr',
            'scaling': None
        }
        
        self.loader = None
        self.preprocessor = None
        self.models = {}
        self.results = {}
        
        self._train_data = None
        self._test_data = None
        self._validation_scores = {}
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return {}
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self,
                  filepath: str = None,
                  data: pd.Series = None,
                  datetime_col: str = 'Datetime',
                  target_col: str = None) -> 'TimeSeriesForecaster':
        """
        Load time series data.
        
        Args:
            filepath: Path to data file (CSV)
            data: Pre-loaded pandas Series (alternative to filepath)
            datetime_col: Name of datetime column
            target_col: Name of target variable column
        
        Returns:
            self
        """
        if data is not None:
            self._raw_data = data
        else:
            filepath = filepath or self.config.get('data', {}).get('source')
            if not filepath:
                raise ValueError("No data source specified")
            
            self.loader = DataLoader()
            df = self.loader.load_csv(
                filepath,
                datetime_col=datetime_col,
                target_col=target_col
            )
            self._raw_data = self.loader.get_target_series()
        
        logger.info(f"Loaded {len(self._raw_data)} data points")
        return self
    
    def preprocess(self, test_size: float = 0.2) -> 'TimeSeriesForecaster':
        """
        Preprocess the data and split into train/test sets.
        
        Args:
            test_size: Fraction of data for testing
        
        Returns:
            self
        """
        self.preprocessor = TimeSeriesPreprocessor(**self.preprocessing_config)
        processed = self.preprocessor.fit_transform(self._raw_data)
        
        self._train_data, self._test_data = train_test_split(
            processed, test_size=test_size
        )
        
        logger.info(f"Train: {len(self._train_data)}, Test: {len(self._test_data)}")
        return self
    
    def fit(self, data: pd.Series = None) -> 'TimeSeriesForecaster':
        """
        Fit all specified models.
        
        Args:
            data: Optional data to fit on (uses loaded data if not provided)
        
        Returns:
            self
        """
        if data is not None:
            self.load_data(data=data)
            self.preprocess()
        
        if self._train_data is None:
            raise ValueError("No data available. Call load_data() first.")
        
        for model_name in self.model_names:
            logger.info(f"Fitting {model_name}...")
            
            try:
                if model_name == 'arima':
                    model = ARIMAForecaster(auto_order=True)
                    model.fit(self._train_data)
                    
                elif model_name == 'exp_smoothing':
                    # Detect appropriate seasonal period
                    seasonal_periods = self._detect_seasonality()
                    model = ExponentialSmoothingForecaster(
                        auto=True,
                        seasonal_periods=seasonal_periods
                    )
                    model.fit(self._train_data)
                    
                elif model_name == 'lstm':
                    from ..models.lstm import LSTMForecaster
                    model = LSTMForecaster(lookback=24, units=[64, 32])
                    model.fit(self._train_data, epochs=50, verbose=0)
                
                else:
                    logger.warning(f"Unknown model: {model_name}")
                    continue
                
                self.models[model_name] = model
                
                # Validate on test set
                if len(self._test_data) > 0:
                    forecast, _ = model.predict(steps=len(self._test_data))
                    score = evaluate_forecast(
                        self._test_data.values,
                        forecast.values
                    )
                    self._validation_scores[model_name] = score['mape']
                    logger.info(f"{model_name} MAPE: {score['mape']:.2f}%")
                    
            except Exception as e:
                logger.error(f"Failed to fit {model_name}: {e}")
                continue
        
        return self
    
    def predict(self,
                steps: int = None,
                return_conf_int: bool = True) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """
        Generate forecasts from all models.
        
        Args:
            steps: Number of steps to forecast (default: self.horizon)
            return_conf_int: Whether to return confidence intervals
        
        Returns:
            Tuple of (ensemble_forecast, confidence_intervals)
        """
        steps = steps or self.horizon
        
        forecasts = {}
        confidence_intervals = {}
        
        for name, model in self.models.items():
            forecast, ci = model.predict(steps=steps, return_conf_int=True)
            forecasts[name] = forecast
            confidence_intervals[name] = ci
        
        # Create ensemble forecast
        if self.ensemble and len(forecasts) > 1:
            ensemble_forecast = self._create_ensemble(forecasts)
            forecasts['ensemble'] = ensemble_forecast
            
            # Ensemble confidence interval (widest bounds)
            if confidence_intervals:
                all_lower = pd.concat([ci['lower'] for ci in confidence_intervals.values()], axis=1)
                all_upper = pd.concat([ci['upper'] for ci in confidence_intervals.values()], axis=1)
                
                ensemble_ci = pd.DataFrame({
                    'lower': all_lower.min(axis=1),
                    'upper': all_upper.max(axis=1)
                })
                confidence_intervals['ensemble'] = ensemble_ci
        
        self.results['forecasts'] = forecasts
        self.results['confidence_intervals'] = confidence_intervals
        
        # Return ensemble if available, otherwise first model
        if 'ensemble' in forecasts:
            return forecasts['ensemble'], confidence_intervals.get('ensemble')
        else:
            first_model = list(forecasts.keys())[0]
            return forecasts[first_model], confidence_intervals.get(first_model)
    
    def _create_ensemble(self, forecasts: Dict[str, pd.Series]) -> pd.Series:
        """Create ensemble forecast using specified method."""
        if self.ensemble_method == 'mean':
            # Simple average
            forecast_df = pd.DataFrame(forecasts)
            return forecast_df.mean(axis=1)
        
        elif self.ensemble_method == 'weighted':
            # Weight by inverse MAPE (better models get higher weight)
            if not self._validation_scores:
                return pd.DataFrame(forecasts).mean(axis=1)
            
            weights = {}
            total_inv_mape = sum(1/v for v in self._validation_scores.values() if v > 0)
            
            for name in forecasts:
                if name in self._validation_scores and self._validation_scores[name] > 0:
                    weights[name] = (1 / self._validation_scores[name]) / total_inv_mape
                else:
                    weights[name] = 1 / len(forecasts)
            
            ensemble = sum(forecasts[name] * weights.get(name, 0) 
                          for name in forecasts)
            
            logger.info(f"Ensemble weights: {weights}")
            return ensemble
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def _detect_seasonality(self) -> Optional[int]:
        """Detect seasonal period from the data."""
        # For hourly data, common periods are 24 (daily) or 168 (weekly)
        freq = pd.infer_freq(self._train_data.index)
        
        if freq and 'H' in freq:
            return 24  # Daily seasonality for hourly data
        elif freq and 'D' in freq:
            return 7   # Weekly seasonality for daily data
        
        return None
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Generate comparison of model performance.
        
        Returns:
            DataFrame with metrics for each model
        """
        if not self._validation_scores:
            return pd.DataFrame()
        
        comparison = []
        
        for name, model in self.models.items():
            diagnostics = model.get_diagnostics()
            diagnostics['model'] = name
            diagnostics['mape'] = self._validation_scores.get(name, np.nan)
            comparison.append(diagnostics)
        
        return pd.DataFrame(comparison).set_index('model')
    
    def run(self,
            filepath: str = None,
            datetime_col: str = 'Datetime',
            target_col: str = None) -> Dict[str, Any]:
        """
        Execute the complete forecasting pipeline.
        
        Args:
            filepath: Path to data file
            datetime_col: Name of datetime column
            target_col: Name of target column
        
        Returns:
            Dictionary with forecasts, metrics, and model comparison
        """
        self.load_data(filepath, datetime_col=datetime_col, target_col=target_col)
        self.preprocess()
        self.fit()
        
        forecast, conf_int = self.predict()
        
        return {
            'forecast': forecast,
            'confidence_interval': conf_int,
            'metrics': self._validation_scores,
            'model_comparison': self.get_model_comparison()
        }


def main():
    """Command-line entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Automated Time Series Forecasting'
    )
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, help='Path to data file')
    parser.add_argument('--horizon', type=int, default=24,
                       help='Forecast horizon')
    
    args = parser.parse_args()
    
    forecaster = TimeSeriesForecaster(
        config_path=args.config,
        horizon=args.horizon
    )
    
    if args.data:
        results = forecaster.run(filepath=args.data)
    else:
        results = forecaster.run()
    
    print("\n=== Forecast Results ===")
    print(results['forecast'])
    print("\n=== Model Comparison ===")
    print(results['model_comparison'])


if __name__ == '__main__':
    main()
