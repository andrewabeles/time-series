import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
import plotly.express as px

def split_time_series(df, test_size=0.2):
    """Splits time series dataframe into train and test sets based on time."""
    split_idx = int(len(df) * (1 - test_size))
    train, test = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
    return train, test

def test_stationarity(y, alpha=0.05, **kwargs):
    adf_result = adfuller(y.dropna(), **kwargs)
    adf_stat, p_value = adf_result[:2]
    summary = f"ADF P-value: {p_value}"
    if p_value < alpha:
        return f"{summary}. The time series is stationary."
    else:
        return f"{summary}. The time series is not stationary."

def get_difference(y, order=1, period=1):
    if order == 0:
        return y
    y_diff = y.diff(period)
    order -= 1
    return get_difference(y_diff, order=order, period=period)

def plot_seasonal_decomposition(df, y=None, period=1):
    decomp = seasonal_decompose(df[y].dropna(), period=period)
    decomp_df = df[[y]].copy()
    decomp_df['trend'] = decomp.trend 
    decomp_df['seasonal'] = decomp.seasonal 
    decomp_df['residual'] = decomp.resid
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=decomp_df[y], mode='lines', name='Original'))
    fig.add_trace(go.Scatter(x=df.index, y=decomp_df['trend'], mode='lines', name='Trend'))
    fig.add_trace(go.Scatter(x=df.index, y=decomp_df['seasonal'], mode='lines', name='Seasonality'))
    fig.add_trace(go.Scatter(x=df.index, y=decomp_df['residual'], mode='lines', name='Residual'))
    fig.update_layout(title="Time Series Decomposition", xaxis_title="Date", yaxis_title="Value")
    return fig 

def plot_autocorrelation(y, alpha=0.05, partial=False):
    title = 'Autocorrelation of Differenced Time Series'
    if partial:
        results = pacf(y.dropna(), alpha=alpha)
        title = 'Partial ' + title
    else:
        results = acf(y.dropna(), alpha=alpha)
    results_df = pd.DataFrame()
    results_df['correlation'] = results[0]
    results_df[['lower', 'upper']] = results[1]
    results_df['error'] = results_df['upper'] - results_df['correlation']
    results_df['statsig'] = (results_df['lower'] > 0) | (results_df['upper'] < 0)
    results_df['lag'] = results_df.index 
    fig = px.scatter(
        results_df.query("lag > 0"),
        x='lag',
        y='correlation',
        error_y='error',
        color='statsig',
        color_discrete_map={True: 'green', False: 'gray'},
        title=title
    )
    return fig

def plot_forecast_vs_actuals(y_actual, forecast, alpha=0.05, title=None, conf_int=True):
    y_forecast = forecast.predicted_mean
    forecast_df = pd.DataFrame({
        't': y_actual.index,
        'actual': y_actual.values,
        'forecast': forecast.predicted_mean
    })
    if conf_int:
        forecast_df[['forecast_lower', 'forecast_upper']] = forecast.conf_int(alpha)
    forecast_melt_df = forecast_df.melt(id_vars='t', var_name='series', value_name='value')
    fig = px.line(
        forecast_melt_df,
        x='t',
        y='value',
        color='series',
        title=title
    )
    return fig 
