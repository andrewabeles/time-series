import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
import plotly.graph_objects as go
import plotly.express as px

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
