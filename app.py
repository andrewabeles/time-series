import streamlit as st 
import pandas as pd 
import numpy as np 
import plotly.express as px 
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose

st.title("Time Series Analyzer")

uploaded_file = st.file_uploader("Upload time series CSV file")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview")
    st.write(df.head())

    # set datetime column as index 
    dt_colname = st.selectbox(
        "Select datetime column",
        df.select_dtypes(include=[object, 'datetime', 'datetimetz']).columns
    )
    df.set_index(dt_colname, inplace=True)

    # set time series frequency 
    freq_codes = {
        'nanosecond': 'N',
        'microsecond': 'U',
        'millisecond': 'L',
        'second': 'S',
        'minute': 'T',
        'hour': 'H',
        'day': 'D',
        'week': 'W',
        'month start': 'MS',
        'month end': 'ME',
        'quarter start': 'QS',
        'quarter end': 'QE',
        'year start': 'YS',
        'year end': 'YE'
    }
    freq_label = st.selectbox(
        "Select frequency",
        [k for k in freq_codes.keys()],
        index=6
    )
    freq = freq_codes[freq_label]
    df = df.asfreq(freq)
    
    # determine target 
    y_colname = st.selectbox(
        "Select values column",
        df.select_dtypes(include='number').columns
    )

    st.write(df.head())
