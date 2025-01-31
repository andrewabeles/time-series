import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px 
from utils import plot_seasonal_decomposition, plot_autocorrelation

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



    # difference time series 
    st.header("Difference Time Series")
    y_diff_colname = f'{y_colname}_diff'
    df[y_diff_colname] = df[y_colname].diff()

    diff_fig = px.line(
        df,
        y=y_diff_colname,
        title='Differenced Time Series'
    )
    st.plotly_chart(diff_fig)

    # autocorrelation 
    confidence = st.selectbox(
        "Confidence Level",
        [0.99, 0.95, 0.9, 0.8],
        index=1
    )
    alpha = 1 - confidence 
    acf_fig = plot_autocorrelation(df[y_diff_colname], alpha=alpha)
    pacf_fig = plot_autocorrelation(df[y_diff_colname], alpha=alpha, partial=True)
    st.plotly_chart(acf_fig)
    st.plotly_chart(pacf_fig)
