import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px 
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose

st.title("Time Series Analyzer")

uploaded_file = st.file_uploader("Upload time series CSV file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview")
    st.write(df.head())

    dt_colname = st.selectbox(
        "Select datetime column",
        df.select_dtypes(include=[object, 'datetime', 'datetimetz']).columns
    )
    df[dt_colname] = pd.to_datetime(df[dt_colname], utc=True)
    df.set_index(dt_colname, inplace=True)
    df = df.resample('D').mean()
    st.write(df.head())

    # determine target 
    y_colname = st.selectbox(
        "Select values column",
        df.select_dtypes(include='number').columns
    )

    # plot raw time series 
    raw_fig = px.line(
        df,
        y=df[y_colname]
    )
    st.plotly_chart(raw_fig)

    st.header("Seasonal Decomposition")
    period = st.number_input(
        "Period of Time Series",
        min_value=1,
        help="""The expected period of seasonality. For example, 4 if data is quarterly, 7 if daily, etc."""
    )
    decomp = seasonal_decompose(df[y_colname].dropna(), period=period)
    decomp_df = df.copy()
    decomp_df['trend'] = decomp.trend 
    decomp_df['seasonal'] = decomp.seasonal 
    decomp_df['residual'] = decomp.resid

    decomp_fig = go.Figure()

    # Add traces
    decomp_fig.add_trace(go.Scatter(x=df.index, y=df[y_colname], mode='lines', name='Original'))
    decomp_fig.add_trace(go.Scatter(x=df.index, y=decomp_df['trend'], mode='lines', name='Trend'))
    decomp_fig.add_trace(go.Scatter(x=df.index, y=decomp_df['seasonal'], mode='lines', name='Seasonality'))
    decomp_fig.add_trace(go.Scatter(x=df.index, y=decomp_df['residual'], mode='lines', name='Residual'))

    # Update layout
    decomp_fig.update_layout(title="Time Series Decomposition", xaxis_title="Date", yaxis_title="Value")

    st.plotly_chart(decomp_fig)