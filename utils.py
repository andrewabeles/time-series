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

def get_difference(y, degree=1):
    y_diff = y.diff()
    degree -= 1
    if degree == 0:
        return y_diff
    else:
        return get_difference(y_diff, degree=degree)

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

def arima_grid_search(y_train, y_val, p_range=(0, 2), d_range=(0, 1), q_range=(0, 2)):
    best_aic, best_rmse = np.inf, np.inf 
    best_order = None 
    results = []

    for p, d, q in itertools.product(
            range(p_range[0], p_range[1] + 1),
            range(d_range[0], d_range[1] + 1),
            range(q_range[0], q_range[1] + 1)
        ):
        aic, rmse = evaluate_arima(y_train, y_val, p=p, d=d, q=q)
        results.append([p, d, q, aic, rmse])
        if aic < best_aic:
            best_aic, best_rmse, best_order = aic, rmse, (p, d, q)
    results_df = pd.DataFrame(results, columns=['p', 'd', 'q', 'AIC', 'RMSE'])
    return results_df.sort_values('AIC')

def evaluate_arima(y_train, y_val, p=0, d=0, q=0):
    try:
        model = ARIMA(y_train, order=(p, d, q))
        model_fit = model.fit()
        
        # Forecast on validation set
        y_forecast = model_fit.forecast(steps=len(y_val))
        
        # Compute RMSE
        rmse = np.sqrt(mean_squared_error(y_val, y_forecast))
        
        return model_fit.aic, rmse
    except:
        return np.inf, np.inf  # Return bad scores if model fails

def plot_forecast_vs_actuals(y_actual, y_forecast):
    forecast_df = pd.DataFrame({
        't': y_actual.index,
        'actual': y_actual.values,
        'forecast': y_forecast
    }).melt(id_vars='t', var_name='series', value_name='value')
    fig = px.line(
        forecast_df,
        x='t',
        y='value',
        color='series',
        title='Forecast vs. Actual (Validation Set)'
    )
    return fig 
