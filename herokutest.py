import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import urllib.request
from datetime import datetime, timedelta, date
from sklearn.preprocessing import MinMaxScaler

# Create a title, a subheader.
st.title("Coronavirus forecast")
st.subheader("This is an app for predicting new number of coronavirus cases during two weeks according to public data. ")
