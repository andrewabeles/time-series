import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from utils import (
    split_time_series, test_stationarity, get_difference, 
    plot_seasonal_decomposition, plot_autocorrelation, 
    plot_forecast_vs_actuals
)

st.title("Time Series Analyzer")

with st.sidebar:
    with st.expander("Data"):
        uploaded_file = st.file_uploader("Upload time series CSV file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

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
    with st.expander("ARIMA Parameters"):
        d = st.number_input(
            "Differencing Order",
            min_value=0,
            max_value=3,
            value=1
        )
        p = st.number_input(
            "AR Order",
            min_value=0,
            help="""Autoregressive order. The number of previous periods' values to use as predictors of the current value."""
        )
        q = st.number_input(
            "MA Order",
            min_value=0,
            help="""Moving average order. The number of previous shocks to use as predictors of the current value."""
        )
        trend_codes = {
            'none': None,
            'constant': 'c',
            'linear': 't',
            'constant and linear': 'ct'
        }
        if d > 1:
            trend_codes = {'none': None}
        elif d > 0:
            trend_codes = {k: v for k, v in trend_codes.items() if k not in ['constant', 'constant and linear']}
        trend_label = st.selectbox("Trend", [k for k in trend_codes.keys()])
        trend = trend_codes[trend_label]
    with st.expander("Seasonal Parameters"):
        include_seasonality_in_forecast = st.checkbox("Include Seasonality in Forecast")
        D = st.number_input(
            "Seasonal Differencing Order",
            min_value=0,
            max_value=3,
            value=1
        )
        P = st.number_input(
            "Seasonal AR Order",
            min_value=0
        )
        Q = st.number_input(
            "Seasonal MA Order",
            min_value=0
        )
        s = st.number_input(
            "Seasonal Period",
            min_value=2,
            help="""The expected period of seasonality. For example, 4 if data is quarterly, 7 if daily, etc."""
        )
    confidence = st.selectbox(
        "Confidence Level",
        [0.99, 0.95, 0.9, 0.8],
        index=1
    )
    alpha = 1 - confidence 

if uploaded_file is not None:
    # preview data
    st.write(df.tail())

    # split data 
    df_non_test, df_test = split_time_series(df, test_size=test_size)
    df_train, df_val = split_time_series(df_non_test, test_size=val_size)
    y_train = df_train[y_colname]
    y_val = df_val[y_colname]
    y_test = df_test[y_colname]
    col1a, col2a, col3a, col4a = st.columns(4)
    with col1a:
        st.write(f'Total Rows: {len(df)}')
    with col2a:
        st.write(f'Train Rows: {len(df_train)}')
    with col3a:
        st.write(f'Validation Rows: {len(df_val)}')
    with col4a:
        st.write(f'Test Rows: {len(df_test)}')

    # difference time series 
    y_diff_colname = f'{y_colname}_diff'
    y_train_diff = get_difference(y_train, order=d)

    col1b, col2b = st.columns(2)
    with col1b:
        orig_fig = px.line(y_train, title='Original Time Series')
        st.plotly_chart(orig_fig, key='orig_fig')
        stationarity = test_stationarity(y_train, alpha=alpha)
        st.write(stationarity)
        acf_fig = plot_autocorrelation(y_train_diff, alpha=alpha)
        st.plotly_chart(acf_fig, key='acf_fig')
    with col2b:
        # autocorrelation 
        diff_fig = px.line(y_train_diff, title='Differenced Time Series')
        st.plotly_chart(diff_fig, key='diff_fig')
        stationarity_diff = test_stationarity(y_train_diff, alpha=alpha)
        st.write(stationarity_diff)
        pacf_fig = plot_autocorrelation(y_train_diff, alpha=alpha, partial=True)
        st.plotly_chart(pacf_fig, key='pacf_fig')

    st.header("Seasonality")
    y_train_seasonal = get_difference(y_train, order=D, period=s)
    col1c, col2c = st.columns(2, vertical_alignment='top')
    with col1c:
        decomp_fig = plot_seasonal_decomposition(df_train, y=y_colname, period=s)
        st.plotly_chart(decomp_fig, key='decomp_fig')
        seasonal_acf_fig = plot_autocorrelation(y_train_seasonal, alpha=alpha)
        st.plotly_chart(seasonal_acf_fig, key='seasonal_acf_fig')
    with col2c:
        seasonal_fig = px.line(
            y_train_seasonal,
            title='Seasonally Differenced Time Series'
        )
        st.plotly_chart(seasonal_fig, key='seasonal_fig')
        seasonal_pacf_fig = plot_autocorrelation(y_train_seasonal, alpha=alpha, partial=True)
        st.plotly_chart(seasonal_pacf_fig, key='seasonal_pacf_fig')
        stationarity_seasonal = test_stationarity(y_train_diff, alpha=alpha)
        st.write(stationarity_seasonal)

    # Seasonal ARIMA modeling 
    st.header("ARIMA Forecasting")
    if not include_seasonality_in_forecast:
        P = 0
        D = 0 
        Q = 0
        s = 0
    model = SARIMAX(
        y_train, 
        trend=trend, 
        order=(p, d, q),
        seasonal_order=(P, D, Q, s)
    ).fit()

    col1d, col2d = st.columns(2)
    with col1d:
        # training set predictions 
        y_train_pred = model.get_prediction()
        train_pred_vs_actuals_fig = plot_forecast_vs_actuals(y_train, y_train_pred, title='Predictions vs. Training Data', conf_int=False)
        st.plotly_chart(train_pred_vs_actuals_fig, key='train_pred_vs_actuals_fig')
    with col2d:
        # validation set forecast 
        y_val_forecast = model.get_forecast(steps=len(y_val))
        val_forecast_vs_actuals_fig = plot_forecast_vs_actuals(y_val, y_val_forecast, title='Forecast vs. Validation Data')
        st.plotly_chart(val_forecast_vs_actuals_fig, key='val_forecast_vs_actuals_fig')
