"""
Visualization utilities for time series forecasting.

Provides publication-ready plots for forecasts, model comparison,
diagnostics, and seasonal decomposition.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import warnings

# Default style settings
plt.style.use('seaborn-v0_8-whitegrid')


class ForecastPlotter:
    """
    Create publication-ready forecast visualizations.
    
    Provides methods for plotting forecasts with confidence intervals,
    comparing multiple models, and generating diagnostic plots.
    
    Attributes:
        figsize: Default figure size (width, height)
        colors: Color palette for different elements
    
    Example:
        >>> plotter = ForecastPlotter()
        >>> plotter.plot_forecast(historical, forecast, conf_int)
        >>> plotter.save('forecast.png')
    """
    
    def __init__(self,
                 figsize: Tuple[int, int] = (14, 6),
                 style: str = 'seaborn-v0_8-whitegrid'):
        """
        Initialize the plotter.
        
        Args:
            figsize: Default figure size
            style: Matplotlib style to use
        """
        self.figsize = figsize
        self.style = style
        
        self.colors = {
            'historical': '#2c3e50',
            'forecast': '#e74c3c',
            'confidence': '#e74c3c',
            'actual': '#27ae60',
            'arima': '#3498db',
            'exp_smoothing': '#9b59b6',
            'lstm': '#f39c12',
            'ensemble': '#e74c3c'
        }
        
        self._current_fig = None
        self._current_ax = None
    
    def plot_forecast(self,
                      historical: pd.Series,
                      forecast: pd.Series,
                      conf_int: pd.DataFrame = None,
                      actual: pd.Series = None,
                      title: str = 'Time Series Forecast',
                      ylabel: str = 'Value',
                      show_legend: bool = True) -> plt.Figure:
        """
        Plot forecast with optional confidence intervals.
        
        Args:
            historical: Historical data series
            forecast: Forecast series
            conf_int: DataFrame with 'lower' and 'upper' columns
            actual: Actual values for the forecast period (if available)
            title: Plot title
            ylabel: Y-axis label
            show_legend: Whether to show legend
        
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot historical data
        ax.plot(historical.index, historical.values,
                color=self.colors['historical'],
                linewidth=1.5,
                label='Historical')
        
        # Plot forecast
        ax.plot(forecast.index, forecast.values,
                color=self.colors['forecast'],
                linewidth=2,
                label='Forecast')
        
        # Plot confidence interval
        if conf_int is not None:
            ax.fill_between(forecast.index,
                           conf_int['lower'],
                           conf_int['upper'],
                           color=self.colors['confidence'],
                           alpha=0.2,
                           label='95% CI')
        
        # Plot actual values if available
        if actual is not None:
            ax.plot(actual.index, actual.values,
                   color=self.colors['actual'],
                   linewidth=1.5,
                   linestyle='--',
                   label='Actual')
        
        # Formatting
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
        plt.xticks(rotation=45)
        
        if show_legend:
            ax.legend(loc='upper left', framealpha=0.9)
        
        plt.tight_layout()
        
        self._current_fig = fig
        self._current_ax = ax
        
        return fig
    
    def plot_model_comparison(self,
                              historical: pd.Series,
                              forecasts: Dict[str, pd.Series],
                              actual: pd.Series = None,
                              title: str = 'Model Comparison',
                              ylabel: str = 'Value') -> plt.Figure:
        """
        Compare forecasts from multiple models.
        
        Args:
            historical: Historical data series
            forecasts: Dictionary mapping model names to forecast series
            actual: Actual values for comparison
            title: Plot title
            ylabel: Y-axis label
        
        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot historical
        ax.plot(historical.index, historical.values,
                color=self.colors['historical'],
                linewidth=1.5,
                label='Historical')
        
        # Plot each model's forecast
        for name, forecast in forecasts.items():
            color = self.colors.get(name, None)
            ax.plot(forecast.index, forecast.values,
                   color=color,
                   linewidth=2,
                   label=name.replace('_', ' ').title())
        
        # Plot actual if available
        if actual is not None:
            ax.plot(actual.index, actual.values,
                   color=self.colors['actual'],
                   linewidth=2,
                   linestyle='--',
                   label='Actual')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        ax.legend(loc='upper left', framealpha=0.9)
        plt.tight_layout()
        
        self._current_fig = fig
        return fig
    
    def plot_residuals(self,
                       residuals: pd.Series,
                       title: str = 'Residual Analysis') -> plt.Figure:
        """
        Create residual diagnostic plots.
        
        Includes residual time series, histogram, and ACF plot.
        
        Args:
            residuals: Model residuals
            title: Overall title
        
        Returns:
            Matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Residual time series
        axes[0, 0].plot(residuals.index, residuals.values,
                       color='#2c3e50', linewidth=0.8)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Residual')
        
        # Histogram
        axes[0, 1].hist(residuals.dropna(), bins=30, 
                       color='#3498db', edgecolor='white', alpha=0.7)
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].set_title('Residual Distribution')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frequency')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals.dropna(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # ACF of residuals
        from statsmodels.graphics.tsaplots import plot_acf
        plot_acf(residuals.dropna(), lags=40, ax=axes[1, 1], alpha=0.05)
        axes[1, 1].set_title('Autocorrelation of Residuals')
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        self._current_fig = fig
        return fig
    
    def plot_seasonal_decomposition(self,
                                    series: pd.Series,
                                    period: int = None,
                                    title: str = 'Seasonal Decomposition') -> plt.Figure:
        """
        Plot seasonal decomposition of the time series.
        
        Args:
            series: Time series data
            period: Seasonal period (auto-detected if None)
            title: Plot title
        
        Returns:
            Matplotlib Figure object
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Auto-detect period if not specified
        if period is None:
            freq = pd.infer_freq(series.index)
            if freq and 'H' in freq:
                period = 24
            elif freq and 'D' in freq:
                period = 7
            else:
                period = 12
        
        decomposition = seasonal_decompose(series, period=period, extrapolate_trend='freq')
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        
        components = [
            ('Observed', decomposition.observed, '#2c3e50'),
            ('Trend', decomposition.trend, '#e74c3c'),
            ('Seasonal', decomposition.seasonal, '#27ae60'),
            ('Residual', decomposition.resid, '#9b59b6')
        ]
        
        for ax, (name, data, color) in zip(axes, components):
            ax.plot(data.index, data.values, color=color, linewidth=1)
            ax.set_ylabel(name, fontsize=11)
            ax.set_xlim(data.index.min(), data.index.max())
        
        axes[0].set_title(title, fontsize=14, fontweight='bold')
        axes[-1].set_xlabel('Date', fontsize=11)
        
        plt.tight_layout()
        
        self._current_fig = fig
        return fig
    
    def save(self, filepath: str, dpi: int = 150, **kwargs) -> None:
        """
        Save the current figure.
        
        Args:
            filepath: Output file path
            dpi: Resolution in dots per inch
            **kwargs: Additional arguments for plt.savefig
        """
        if self._current_fig is None:
            raise ValueError("No figure to save. Create a plot first.")
        
        self._current_fig.savefig(filepath, dpi=dpi, 
                                  bbox_inches='tight', **kwargs)
        print(f"Figure saved to {filepath}")
    
    def show(self) -> None:
        """Display the current figure."""
        plt.show()


def plot_energy_forecast(historical: pd.Series,
                         forecast: pd.Series,
                         conf_int: pd.DataFrame = None,
                         actual: pd.Series = None,
                         title: str = 'Energy Consumption Forecast') -> plt.Figure:
    """
    Convenience function for plotting energy forecasts.
    
    Args:
        historical: Historical energy consumption
        forecast: Forecasted values
        conf_int: Confidence intervals
        actual: Actual values for comparison
        title: Plot title
    
    Returns:
        Matplotlib Figure
    """
    plotter = ForecastPlotter()
    return plotter.plot_forecast(
        historical, forecast, conf_int, actual,
        title=title,
        ylabel='Energy Consumption (MW)'
    )
