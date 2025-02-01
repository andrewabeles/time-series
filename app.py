import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px 
from utils import test_stationarity, get_difference, plot_seasonal_decomposition, plot_autocorrelation

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

    # determine target 
    y_colname = st.selectbox(
        "Select values column",
        df.select_dtypes(include='number').columns
    )

    # plot raw time series 
    raw_fig = px.line(
        df,
        y=df[y_colname],
        title='Raw Time Series'
    )
    st.plotly_chart(raw_fig)

    st.header("Seasonal Decomposition")
    period = st.number_input(
        "Period of Time Series",
        min_value=1,
        help="""The expected period of seasonality. For example, 4 if data is quarterly, 7 if daily, etc."""
    )
    decomp_fig = plot_seasonal_decomposition(df, y=y_colname, period=period)
    st.plotly_chart(decomp_fig)

    st.header("Stationarity")
    confidence = st.selectbox(
        "Confidence Level",
        [0.99, 0.95, 0.9, 0.8],
        index=1
    )
    alpha = 1 - confidence 

    st.subheader("Stationarity Test of Raw Time Series")
    stationarity_raw = test_stationarity(df[y_colname], alpha=alpha)
    st.write(stationarity_raw)

    # difference time series 
    degree = st.slider(
        "Difference Degree",
        min_value=1,
        max_value=3,
        value=1
    )
    y_diff_colname = f'{y_colname}_diff'
    df[y_diff_colname] = get_difference(df[y_colname], degree=degree)

    diff_fig = px.line(
        df,
        y=y_diff_colname,
        title='Differenced Time Series'
    )
    st.plotly_chart(diff_fig)

    st.subheader("Stationarity Test of Differenced Time Series")
    stationarity_diff = test_stationarity(df[y_diff_colname], alpha=alpha)
    st.write(stationarity_diff)

    # autocorrelation 
    st.header("Autocorrelation")
    acf_fig = plot_autocorrelation(df[y_diff_colname], alpha=alpha)
    pacf_fig = plot_autocorrelation(df[y_diff_colname], alpha=alpha, partial=True)
    st.plotly_chart(acf_fig)
    st.plotly_chart(pacf_fig)
