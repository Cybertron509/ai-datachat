"""
Time-Series Forecasting Module
Predicts future trends using ARIMA and Exponential Smoothing
"""
import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesForecaster:
    """Forecast future values for time-series data"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
    def detect_time_columns(self) -> list:
        """Detect potential time/date columns"""
        time_cols = []
        
        for col in self.df.columns:
            # Check if column is datetime type
            if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                time_cols.append(col)
            # Check if object column can be converted to datetime
            elif self.df[col].dtype == 'object':
                try:
                    pd.to_datetime(self.df[col], errors='raise')
                    time_cols.append(col)
                except:
                    pass
        
        return time_cols
    
    def prepare_time_series(self, date_col: str, value_col: str) -> Tuple[pd.Series, Dict]:
        """Prepare and validate time series data"""
        
        # Convert to datetime
        df_clean = self.df[[date_col, value_col]].copy()
        df_clean[date_col] = pd.to_datetime(df_clean[date_col])
        
        # Remove missing values
        df_clean = df_clean.dropna()
        
        # Sort by date
        df_clean = df_clean.sort_values(date_col)
        
        # Set date as index
        df_clean.set_index(date_col, inplace=True)
        
        # Check for duplicates - aggregate if found
        if df_clean.index.duplicated().any():
            df_clean = df_clean.groupby(df_clean.index).mean()
        
        series = df_clean[value_col]
        
        # Validate data quality
        validation = {
            'total_points': len(series),
            'missing_count': series.isnull().sum(),
            'date_range': (series.index.min(), series.index.max()),
            'frequency': self._infer_frequency(series.index),
            'has_trend': self._check_trend(series),
            'has_seasonality': self._check_seasonality(series)
        }
        
        return series, validation
    
    def _infer_frequency(self, index: pd.DatetimeIndex) -> str:
        """Infer the frequency of the time series"""
        try:
            freq = pd.infer_freq(index)
            if freq:
                return freq
        except:
            pass
        
        # Manual frequency detection
        diffs = index.to_series().diff().dropna()
        median_diff = diffs.median()
        
        if median_diff <= timedelta(days=1):
            return "Daily"
        elif median_diff <= timedelta(days=7):
            return "Weekly"
        elif median_diff <= timedelta(days=31):
            return "Monthly"
        elif median_diff <= timedelta(days=92):
            return "Quarterly"
        else:
            return "Yearly"
    
    def _check_trend(self, series: pd.Series) -> bool:
        """Simple trend detection"""
        if len(series) < 10:
            return False
        
        # Linear regression slope
        x = np.arange(len(series))
        y = series.values
        
        # Remove NaN
        valid = ~np.isnan(y)
        if valid.sum() < 10:
            return False
        
        x = x[valid]
        y = y[valid]
        
        slope = np.polyfit(x, y, 1)[0]
        return abs(slope) > (y.std() / len(y))
    
    def _check_seasonality(self, series: pd.Series) -> bool:
        """Simple seasonality detection"""
        if len(series) < 24:
            return False
        
        # Check for repeating patterns
        try:
            # Autocorrelation at lag 12 for monthly data
            if len(series) >= 24:
                acf_12 = series.autocorr(lag=12)
                return abs(acf_12) > 0.3
        except:
            pass
        
        return False
    
    def forecast_exponential_smoothing(
        self, 
        series: pd.Series, 
        periods: int = 180
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Forecast using Exponential Smoothing (simpler, more reliable)
        periods: number of days to forecast (default 180 = 6 months)
        """
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
        except ImportError:
            raise ImportError("statsmodels is required. Add to requirements.txt")
        
        # Determine seasonality
        seasonal_periods = None
        if len(series) >= 24:
            freq = self._infer_frequency(series.index)
            if 'Monthly' in freq:
                seasonal_periods = 12
            elif 'Quarterly' in freq:
                seasonal_periods = 4
            elif 'Weekly' in freq:
                seasonal_periods = 52
        
        # Fit model
        try:
            if seasonal_periods and len(series) >= 2 * seasonal_periods:
                model = ExponentialSmoothing(
                    series,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=seasonal_periods
                )
            else:
                model = ExponentialSmoothing(
                    series,
                    trend='add',
                    seasonal=None
                )
            
            fitted_model = model.fit()
            
        except:
            # Fallback to simple exponential smoothing
            model = ExponentialSmoothing(series, trend=None, seasonal=None)
            fitted_model = model.fit()
        
        # Generate forecast
        forecast = fitted_model.forecast(steps=periods)
        
        # Calculate confidence intervals (simple approximation)
        residuals = series - fitted_model.fittedvalues
        std_error = residuals.std()
        
        forecast_df = pd.DataFrame({
            'forecast': forecast.values,
            'lower_bound': forecast.values - 1.96 * std_error,
            'upper_bound': forecast.values + 1.96 * std_error
        }, index=forecast.index)
        
        return fitted_model.fittedvalues, forecast_df
    
    def forecast_arima(
        self,
        series: pd.Series,
        periods: int = 180
    ) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Forecast using ARIMA (Auto ARIMA for best parameters)
        periods: number of periods to forecast
        """
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            raise ImportError("statsmodels is required. Add to requirements.txt")
        
        # Test for stationarity
        adf_result = adfuller(series.dropna())
        is_stationary = adf_result[1] < 0.05
        
        # Use simple ARIMA parameters
        p, d, q = (1, 0 if is_stationary else 1, 1)
        
        try:
            # Fit ARIMA model
            model = SARIMAX(
                series,
                order=(p, d, q),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            fitted_model = model.fit(disp=False)
            
            # Generate forecast
            forecast_result = fitted_model.get_forecast(steps=periods)
            forecast = forecast_result.predicted_mean
            confidence_intervals = forecast_result.conf_int()
            
            forecast_df = pd.DataFrame({
                'forecast': forecast.values,
                'lower_bound': confidence_intervals.iloc[:, 0].values,
                'upper_bound': confidence_intervals.iloc[:, 1].values
            }, index=forecast.index)
            
            return fitted_model.fittedvalues, forecast_df
            
        except Exception as e:
            # Fallback to exponential smoothing
            return self.forecast_exponential_smoothing(series, periods)
    
    def evaluate_forecast(
        self,
        actual: pd.Series,
        fitted: pd.Series
    ) -> Dict:
        """Evaluate forecast accuracy on historical data"""
        
        # Align series
        common_index = actual.index.intersection(fitted.index)
        actual_aligned = actual.loc[common_index]
        fitted_aligned = fitted.loc[common_index]
        
        # Calculate metrics
        residuals = actual_aligned - fitted_aligned
        
        mae = np.abs(residuals).mean()
        rmse = np.sqrt((residuals ** 2).mean())
        mape = (np.abs(residuals / actual_aligned) * 100).mean()
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r_squared': 1 - (residuals.var() / actual_aligned.var())
        }