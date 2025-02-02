import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px 
from statsmodels.tsa.arima.model import ARIMA
from utils import (
    split_time_series, test_stationarity, get_difference, 
    plot_seasonal_decomposition, plot_autocorrelation, 
    arima_grid_search, plot_forecast_vs_actuals
)

st.title("Time Series Analyzer")

uploaded_file = st.file_uploader("Upload time series CSV file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.write(df.tail())

    dt_colname = st.selectbox(
        "Select datetime column",
        df.select_dtypes(include=[object, 'datetime', 'datetimetz']).columns
    )
    df[dt_colname] = pd.to_datetime(df[dt_colname], utc=True)
    df.set_index(dt_colname, inplace=True)

    # determine target 
    y_colname = st.selectbox(
        "Select values column",
        df.select_dtypes(include='number').columns
    )
    y = df[y_colname]

    # split data into train, validation, and test sets 
    st.write(f'Total Rows: {len(df)}')

    test_size = st.slider(
        "Select Test Set Size",
        min_value=0.1,
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="""The proportion of total rows that will be used for final model evaluation."""
    )
    val_size = st.slider(
        "Select Validation Set Size",
        min_value=0.1, 
        max_value=0.5,
        value=0.2,
        step=0.05,
        help="""The proportion of non-test rows that will be used for model selection."""
    )
    df_non_test, df_test = split_time_series(df, test_size=test_size)
    df_train, df_val = split_time_series(df_non_test, test_size=val_size)
    y_train = df_train[y_colname]
    y_val = df_val[y_colname]
    y_test = df_test[y_colname]
    st.write(f'Train Rows: {len(df_train)}')
    st.write(f'Validation Rows: {len(df_val)}')
    st.write(f'Test Rows: {len(df_test)}')

    # plot raw time series 
    raw_fig = px.line(
        y_train,
        title='Raw Time Series'
    )
    st.plotly_chart(raw_fig)

    st.header("Seasonal Decomposition")
    period = st.number_input(
        "Period of Time Series",
        min_value=1,
        help="""The expected period of seasonality. For example, 4 if data is quarterly, 7 if daily, etc."""
    )
    decomp_fig = plot_seasonal_decomposition(df_train, y=y_colname, period=period)
    st.plotly_chart(decomp_fig)

    st.header("Stationarity")
    confidence = st.selectbox(
        "Confidence Level",
        [0.99, 0.95, 0.9, 0.8],
        index=1
    )
    alpha = 1 - confidence 

    st.subheader("Stationarity Test of Raw Time Series")
    stationarity_raw = test_stationarity(y_train, alpha=alpha)
    st.write(stationarity_raw)

    # difference time series 
    degree = st.number_input(
        "Difference Degree",
        min_value=0,
        max_value=5,
        value=1
    )
    y_diff_colname = f'{y_colname}_diff'
    if degree > 0:
        y_train_diff = get_difference(y_train, degree=degree)
    else:
        y_train_diff = y_train

    diff_fig = px.line(
        y_train_diff,
        title='Differenced Time Series'
    )
    st.plotly_chart(diff_fig)

    st.subheader("Stationarity Test of Differenced Time Series")
    stationarity_diff = test_stationarity(y_train_diff, alpha=alpha)
    st.write(stationarity_diff)

    # autocorrelation 
    st.header("Autocorrelation")
    acf_fig = plot_autocorrelation(y_train_diff, alpha=alpha)
    pacf_fig = plot_autocorrelation(y_train_diff, alpha=alpha, partial=True)
    st.plotly_chart(acf_fig)
    st.plotly_chart(pacf_fig)

    # ARIMA modeling 
    st.header("ARIMA Model Selection")
    p = st.number_input("Select p", min_value=0, value=1)
    d = st.number_input("Select d", min_value=0, value=1)
    q = st.number_input("Select q", min_value=0, value=1)
    arima_model = ARIMA(df_train[y_colname], order=(p, d, q)).fit()
    forecast = arima_model.get_forecast(steps=len(df))
    y_val_forecast = arima_model.forecast(steps=len(df_val[y_colname]))
    val_forecast_vs_actuals_fig = plot_forecast_vs_actuals(df_val[y_colname], y_val_forecast)
    st.plotly_chart(val_forecast_vs_actuals_fig)

    st.write(arima_model.get_forecast(steps=len(df_val[y_colname])))
