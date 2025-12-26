"""
Exponential Smoothing forecasting models.

Implements Simple, Holt's, and Holt-Winters exponential smoothing
with automatic model selection capabilities.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import logging

logger = logging.getLogger(__name__)


class ExponentialSmoothingForecaster:
    """
    Exponential Smoothing model for time series forecasting.
    
    Supports Simple Exponential Smoothing, Holt's method (with trend),
    and Holt-Winters method (with trend and seasonality). Can
    automatically select the best configuration.
    
    Attributes:
        trend: Type of trend component ('add', 'mul', None)
        seasonal: Type of seasonal component ('add', 'mul', None)
        seasonal_periods: Number of periods in a complete seasonal cycle
        model_fit: Fitted statsmodels result object
    
    Example:
        >>> model = ExponentialSmoothingForecaster(
        ...     seasonal_periods=24,  # Hourly data with daily seasonality
        ...     auto=True
        ... )
        >>> model.fit(train_data)
        >>> forecast, ci = model.predict(steps=48)
    """
    
    def __init__(self,
                 trend: str = None,
                 seasonal: str = None,
                 seasonal_periods: int = None,
                 damped_trend: bool = False,
                 auto: bool = True,
                 use_boxcox: bool = False):
        """
        Initialize the Exponential Smoothing forecaster.
        
        Args:
            trend: Trend type ('add', 'mul', or None)
            seasonal: Seasonal type ('add', 'mul', or None)
            seasonal_periods: Length of seasonal cycle (e.g., 24 for hourly
                            with daily seasonality, 7 for daily with weekly)
            damped_trend: Whether to damp the trend
            auto: Whether to automatically select best configuration
            use_boxcox: Whether to apply Box-Cox transformation
        """
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.auto = auto
        self.use_boxcox = use_boxcox
        
        self.model = None
        self.model_fit = None
        self._train_data = None
        self._fitted_params = {}
    
    def fit(self, data: pd.Series) -> 'ExponentialSmoothingForecaster':
        """
        Fit the Exponential Smoothing model.
        
        Args:
            data: Time series data to fit
        
        Returns:
            self
        """
        self._train_data = data.copy()
        
        # Ensure positive values for multiplicative models
        if (data <= 0).any():
            logger.warning("Non-positive values detected. Using additive components only.")
            multiplicative_ok = False
        else:
            multiplicative_ok = True
        
        if self.auto:
            self._auto_select(data, multiplicative_ok)
            logger.info(f"Selected: trend={self.trend}, seasonal={self.seasonal}, "
                       f"damped={self.damped_trend}")
        
        # Fit the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            self.model = ExponentialSmoothing(
                data,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                damped_trend=self.damped_trend if self.trend else False,
                use_boxcox=self.use_boxcox if multiplicative_ok else False
            )
            
            self.model_fit = self.model.fit(optimized=True)
        
        # Store fitted parameters
        self._fitted_params = {
            'alpha': self.model_fit.params.get('smoothing_level', None),
            'beta': self.model_fit.params.get('smoothing_trend', None),
            'gamma': self.model_fit.params.get('smoothing_seasonal', None),
            'phi': self.model_fit.params.get('damping_trend', None)
        }
        
        logger.info(f"Model fitted. SSE: {self.model_fit.sse:.2f}")
        
        return self
    
    def predict(self,
                steps: int = 10,
                return_conf_int: bool = False,
                alpha: float = 0.05) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps to forecast
            return_conf_int: Whether to return confidence intervals
            alpha: Significance level for confidence intervals
        
        Returns:
            Tuple of (forecast, confidence_intervals) or just forecast
        """
        if self.model_fit is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Generate forecast
        forecast = self.model_fit.forecast(steps=steps)
        
        # Generate future index
        last_date = self._train_data.index[-1]
        freq = pd.infer_freq(self._train_data.index)
        if freq is None:
            freq = 'H'
        
        future_index = pd.date_range(
            start=last_date + pd.Timedelta(1, unit=freq[0].lower() if freq else 'h'),
            periods=steps,
            freq=freq
        )
        forecast.index = future_index
        
        if return_conf_int:
            # Approximate confidence intervals using historical residuals
            residuals = self.model_fit.resid
            sigma = residuals.std()
            z = 1.96 if alpha == 0.05 else 2.576  # 95% or 99% CI
            
            # Widen intervals for longer horizons
            horizon_factor = np.sqrt(np.arange(1, steps + 1))
            
            ci = pd.DataFrame({
                'lower': forecast - z * sigma * horizon_factor,
                'upper': forecast + z * sigma * horizon_factor
            }, index=future_index)
            
            return forecast, ci
        
        return forecast, None
    
    def _auto_select(self, data: pd.Series, multiplicative_ok: bool) -> None:
        """
        Automatically select the best model configuration.
        
        Tries different combinations and selects based on AIC.
        """
        best_aic = float('inf')
        best_config = (None, None, False)
        
        # Configurations to try
        trend_options = [None, 'add']
        seasonal_options = [None]
        damped_options = [False, True]
        
        if multiplicative_ok:
            trend_options.append('mul')
        
        if self.seasonal_periods and self.seasonal_periods > 1:
            seasonal_options = [None, 'add']
            if multiplicative_ok:
                seasonal_options.append('mul')
        
        for trend in trend_options:
            for seasonal in seasonal_options:
                for damped in damped_options:
                    # Skip invalid configurations
                    if damped and trend is None:
                        continue
                    if seasonal and not self.seasonal_periods:
                        continue
                    
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            
                            model = ExponentialSmoothing(
                                data,
                                trend=trend,
                                seasonal=seasonal,
                                seasonal_periods=self.seasonal_periods,
                                damped_trend=damped if trend else False
                            )
                            fit = model.fit(optimized=True)
                            
                            if fit.aic < best_aic:
                                best_aic = fit.aic
                                best_config = (trend, seasonal, damped)
                                
                    except Exception:
                        continue
        
        self.trend, self.seasonal, self.damped_trend = best_config
        logger.info(f"Best configuration (AIC={best_aic:.2f}): "
                   f"trend={self.trend}, seasonal={self.seasonal}")
    
    def get_components(self) -> pd.DataFrame:
        """
        Extract trend and seasonal components.
        
        Returns:
            DataFrame with level, trend, and seasonal components
        """
        if self.model_fit is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        components = pd.DataFrame(index=self._train_data.index)
        components['observed'] = self._train_data
        components['level'] = self.model_fit.level
        
        if self.trend:
            components['trend'] = self.model_fit.trend
        
        if self.seasonal:
            components['seasonal'] = self.model_fit.season
        
        components['fitted'] = self.model_fit.fittedvalues
        components['residuals'] = self.model_fit.resid
        
        return components
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get model diagnostic information.
        
        Returns:
            Dictionary with diagnostic metrics
        """
        if self.model_fit is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        residuals = self.model_fit.resid
        
        return {
            'trend': self.trend,
            'seasonal': self.seasonal,
            'seasonal_periods': self.seasonal_periods,
            'damped_trend': self.damped_trend,
            'aic': self.model_fit.aic,
            'bic': self.model_fit.bic,
            'sse': self.model_fit.sse,
            'alpha': self._fitted_params.get('alpha'),
            'beta': self._fitted_params.get('beta'),
            'gamma': self._fitted_params.get('gamma'),
            'residual_mean': residuals.mean(),
            'residual_std': residuals.std()
        }
