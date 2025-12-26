"""
ARIMA and SARIMA forecasting models.

Implements automatic order selection using information criteria
and provides diagnostic tools for model validation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
import logging

logger = logging.getLogger(__name__)


class ARIMAForecaster:
    """
    ARIMA/SARIMA model for time series forecasting.
    
    Supports automatic order selection via grid search with
    AIC/BIC criteria. Can handle both non-seasonal (ARIMA)
    and seasonal (SARIMA) patterns.
    
    Attributes:
        order: ARIMA order (p, d, q)
        seasonal_order: Seasonal order (P, D, Q, s) for SARIMA
        model_fit: Fitted statsmodels result object
    
    Example:
        >>> model = ARIMAForecaster(auto_order=True)
        >>> model.fit(train_data)
        >>> forecast, ci = model.predict(steps=24, return_conf_int=True)
    """
    
    def __init__(self,
                 order: Tuple[int, int, int] = None,
                 seasonal_order: Tuple[int, int, int, int] = None,
                 auto_order: bool = True,
                 max_p: int = 3,
                 max_d: int = 2,
                 max_q: int = 3,
                 criterion: str = 'aic'):
        """
        Initialize the ARIMA forecaster.
        
        Args:
            order: ARIMA order (p, d, q). If None and auto_order=True,
                   will be determined automatically.
            seasonal_order: Seasonal order (P, D, Q, s) for SARIMA.
            auto_order: Whether to automatically select order
            max_p: Maximum AR order for grid search
            max_d: Maximum differencing order
            max_q: Maximum MA order for grid search
            criterion: Information criterion ('aic' or 'bic')
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto_order = auto_order
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.criterion = criterion
        
        self.model = None
        self.model_fit = None
        self._train_data = None
    
    def fit(self, data: pd.Series) -> 'ARIMAForecaster':
        """
        Fit the ARIMA model to the data.
        
        Args:
            data: Time series data to fit
        
        Returns:
            self
        """
        self._train_data = data.copy()
        
        # Determine differencing order if needed
        if self.auto_order and self.order is None:
            d = self._determine_d(data)
            self.order = self._select_order(data, d)
            logger.info(f"Selected ARIMA order: {self.order}")
        elif self.order is None:
            self.order = (1, 1, 1)  # Default order
        
        # Fit the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if self.seasonal_order:
                self.model = SARIMAX(
                    data,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            else:
                self.model = ARIMA(
                    data,
                    order=self.order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            
            self.model_fit = self.model.fit(disp=False)
        
        logger.info(f"Model fitted. AIC: {self.model_fit.aic:.2f}, "
                   f"BIC: {self.model_fit.bic:.2f}")
        
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
        
        # Get forecast
        forecast_result = self.model_fit.get_forecast(steps=steps)
        forecast = forecast_result.predicted_mean
        
        # Generate future index
        last_date = self._train_data.index[-1]
        freq = pd.infer_freq(self._train_data.index)
        if freq is None:
            freq = 'H'  # Default to hourly for energy data
        
        future_index = pd.date_range(
            start=last_date + pd.Timedelta(1, unit=freq[0].lower() if freq else 'h'),
            periods=steps,
            freq=freq
        )
        forecast.index = future_index
        
        if return_conf_int:
            ci = forecast_result.conf_int(alpha=alpha)
            ci.index = future_index
            ci.columns = ['lower', 'upper']
            return forecast, ci
        
        return forecast, None
    
    def _determine_d(self, data: pd.Series) -> int:
        """
        Determine differencing order using stationarity tests.
        
        Uses ADF and KPSS tests to determine if differencing is needed.
        """
        # ADF test (null: unit root exists, i.e., non-stationary)
        adf_result = adfuller(data.dropna(), autolag='AIC')
        adf_pvalue = adf_result[1]
        
        # KPSS test (null: stationary)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_result = kpss(data.dropna(), regression='c')
        kpss_pvalue = kpss_result[1]
        
        # Decision logic
        if adf_pvalue < 0.05 and kpss_pvalue > 0.05:
            # Both tests suggest stationary
            d = 0
        elif adf_pvalue >= 0.05 and kpss_pvalue <= 0.05:
            # Both tests suggest non-stationary
            d = 1
        else:
            # Conflicting results, default to d=1
            d = 1
        
        logger.info(f"Stationarity tests - ADF p-value: {adf_pvalue:.4f}, "
                   f"KPSS p-value: {kpss_pvalue:.4f} -> d={d}")
        
        return d
    
    def _select_order(self, data: pd.Series, d: int) -> Tuple[int, int, int]:
        """
        Select optimal (p, d, q) using grid search.
        """
        best_score = float('inf')
        best_order = (1, d, 1)
        
        logger.info(f"Searching for optimal order (d={d})...")
        
        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                if p == 0 and q == 0:
                    continue
                
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = ARIMA(
                            data,
                            order=(p, d, q),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        fit = model.fit(disp=False)
                        
                        score = fit.aic if self.criterion == 'aic' else fit.bic
                        
                        if score < best_score:
                            best_score = score
                            best_order = (p, d, q)
                            
                except Exception:
                    continue
        
        logger.info(f"Best order: {best_order} ({self.criterion.upper()}: {best_score:.2f})")
        return best_order
    
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
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'aic': self.model_fit.aic,
            'bic': self.model_fit.bic,
            'residual_mean': residuals.mean(),
            'residual_std': residuals.std(),
            'n_observations': len(self._train_data)
        }
    
    def summary(self) -> str:
        """Return model summary as string."""
        if self.model_fit is None:
            return "Model not fitted."
        return str(self.model_fit.summary())
