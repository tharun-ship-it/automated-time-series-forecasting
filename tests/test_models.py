"""
Unit tests for the time series forecasting framework.

Run with: pytest tests/test_models.py -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class TestDataLoader:
    """Tests for DataLoader class."""
    
    def test_generate_sample_data(self):
        """Test that sample data generation works."""
        from src.data.loader import DataLoader
        
        loader = DataLoader()
        # Create a simple test CSV
        dates = pd.date_range('2020-01-01', periods=100, freq='H')
        values = np.random.randn(100) * 100 + 1000
        df = pd.DataFrame({'Datetime': dates, 'value': values})
        
        # Save and load
        df.to_csv('/tmp/test_data.csv', index=False)
        loaded = loader.load_csv('/tmp/test_data.csv', 
                                 datetime_col='Datetime',
                                 target_col='value')
        
        assert len(loaded) == 100
        assert isinstance(loaded.index, pd.DatetimeIndex)
    
    def test_validation(self):
        """Test data validation checks."""
        from src.data.loader import DataLoader
        
        loader = DataLoader()
        dates = pd.date_range('2020-01-01', periods=200, freq='H')
        values = np.random.randn(200) * 100 + 1000
        df = pd.DataFrame({'Datetime': dates, 'value': values})
        df.to_csv('/tmp/test_data.csv', index=False)
        
        loader.load_csv('/tmp/test_data.csv',
                       datetime_col='Datetime',
                       target_col='value')
        
        validation = loader.validate()
        
        assert validation['has_data'] == True
        assert validation['datetime_index'] == True
        assert validation['sorted_index'] == True


class TestPreprocessor:
    """Tests for TimeSeriesPreprocessor class."""
    
    def test_handle_missing_interpolate(self):
        """Test interpolation of missing values."""
        from src.data.preprocessor import TimeSeriesPreprocessor
        
        dates = pd.date_range('2020-01-01', periods=10, freq='H')
        values = [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]
        series = pd.Series(values, index=dates)
        
        preprocessor = TimeSeriesPreprocessor(handle_missing='interpolate')
        result = preprocessor.fit_transform(series)
        
        assert not result.isnull().any()
        assert len(result) == 10
    
    def test_outlier_detection_iqr(self):
        """Test IQR-based outlier detection."""
        from src.data.preprocessor import TimeSeriesPreprocessor
        
        dates = pd.date_range('2020-01-01', periods=100, freq='H')
        values = np.random.randn(100)
        values[50] = 100  # Add outlier
        series = pd.Series(values, index=dates)
        
        preprocessor = TimeSeriesPreprocessor(
            outlier_method='iqr',
            clip_outliers=True
        )
        result = preprocessor.fit_transform(series)
        
        # Outlier should be clipped
        assert result.iloc[50] < 100
    
    def test_train_test_split(self):
        """Test chronological train-test split."""
        from src.data.preprocessor import train_test_split
        
        dates = pd.date_range('2020-01-01', periods=100, freq='H')
        series = pd.Series(np.random.randn(100), index=dates)
        
        train, test = train_test_split(series, test_size=0.2)
        
        assert len(train) == 80
        assert len(test) == 20
        assert train.index.max() < test.index.min()


class TestARIMAForecaster:
    """Tests for ARIMAForecaster class."""
    
    def test_fit_predict(self):
        """Test basic fit and predict."""
        from src.models.arima import ARIMAForecaster
        
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        # Create simple trend + noise
        values = np.arange(200) + np.random.randn(200) * 5
        series = pd.Series(values, index=dates)
        
        model = ARIMAForecaster(order=(1, 1, 1), auto_order=False)
        model.fit(series)
        
        forecast, ci = model.predict(steps=10, return_conf_int=True)
        
        assert len(forecast) == 10
        assert ci is not None
        assert 'lower' in ci.columns
        assert 'upper' in ci.columns
    
    def test_auto_order_selection(self):
        """Test automatic order selection."""
        from src.models.arima import ARIMAForecaster
        
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        values = np.cumsum(np.random.randn(200))
        series = pd.Series(values, index=dates)
        
        model = ARIMAForecaster(auto_order=True, max_p=2, max_q=2)
        model.fit(series)
        
        assert model.order is not None
        assert len(model.order) == 3


class TestExponentialSmoothing:
    """Tests for ExponentialSmoothingForecaster class."""
    
    def test_fit_predict(self):
        """Test basic fit and predict."""
        from src.models.exponential_smoothing import ExponentialSmoothingForecaster
        
        dates = pd.date_range('2020-01-01', periods=200, freq='D')
        values = np.sin(np.arange(200) * 2 * np.pi / 7) * 10 + 100
        values += np.random.randn(200) * 2
        series = pd.Series(values, index=dates)
        
        model = ExponentialSmoothingForecaster(
            seasonal_periods=7,
            auto=True
        )
        model.fit(series)
        
        forecast, ci = model.predict(steps=14, return_conf_int=True)
        
        assert len(forecast) == 14
        assert ci is not None
    
    def test_get_components(self):
        """Test component extraction."""
        from src.models.exponential_smoothing import ExponentialSmoothingForecaster
        
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        values = np.arange(100) + np.random.randn(100) * 5
        series = pd.Series(values, index=dates)
        
        model = ExponentialSmoothingForecaster(trend='add', auto=False)
        model.fit(series)
        
        components = model.get_components()
        
        assert 'level' in components.columns
        assert 'fitted' in components.columns


class TestMetrics:
    """Tests for evaluation metrics."""
    
    def test_mape(self):
        """Test MAPE calculation."""
        from src.utils.metrics import mean_absolute_percentage_error
        
        actual = np.array([100, 200, 300, 400])
        predicted = np.array([110, 190, 310, 390])
        
        mape = mean_absolute_percentage_error(actual, predicted)
        
        assert mape > 0
        assert mape < 100
    
    def test_rmse(self):
        """Test RMSE calculation."""
        from src.utils.metrics import root_mean_squared_error
        
        actual = np.array([1, 2, 3, 4, 5])
        predicted = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        
        rmse = root_mean_squared_error(actual, predicted)
        
        assert rmse > 0
        assert rmse < 1
    
    def test_evaluate_forecast(self):
        """Test comprehensive evaluation."""
        from src.utils.metrics import evaluate_forecast
        
        actual = np.random.randn(50) * 10 + 100
        predicted = actual + np.random.randn(50) * 2
        
        metrics = evaluate_forecast(actual, predicted)
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'mape' in metrics
        assert 'r2' in metrics


class TestPipeline:
    """Tests for the forecasting pipeline."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        from src.pipeline.forecaster import TimeSeriesForecaster
        
        forecaster = TimeSeriesForecaster(
            horizon=24,
            models=['arima', 'exp_smoothing'],
            ensemble=True
        )
        
        assert forecaster.horizon == 24
        assert 'arima' in forecaster.model_names
        assert forecaster.ensemble == True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
