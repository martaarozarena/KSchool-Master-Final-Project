import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

@st.cache
def data_load():
    
    url = "https://drive.google.com/file/d/1Mm8oluT401FkocpFvinr852FMPNsQMm1/view?usp=sharing"
    path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
    covid = pd.read_csv(path, parse_dates=["date"])
    covid_eur = covid[covid['continent']=='Europe']
    covid_eur['new_cases'] = covid_eur['new_cases'].fillna(0)
    covid_esp = covid_eur[covid_eur["location"]=="Spain"].reset_index()
    covid_esp.index = covid_esp.date
    covid_esp_newcases = covid_esp["new_cases"]
    covid_esp_newcases = covid_esp_newcases.resample('D').sum()
    covid_esp["MA3"] = covid_esp.new_cases.rolling(3).mean()
    covid_esp_ma3 = covid_esp["MA3"]
    covid_esp_diff3 = covid_esp_newcases - covid_esp_ma3
    url2 = "https://drive.google.com/file/d/1V19t2WshmMUonqZFfWAAF9X2siHiwmv2/view?usp=sharing"
    path2 = 'https://drive.google.com/uc?export=download&id='+url2.split('/')[-2]
    flights_esp = pd.read_csv(path2,parse_dates=["FLT_DATE"],index_col="FLT_DATE")
    flights_esp_arr = flights_esp["FLT_ARR_1"]
    flights_esp_arr = flights_esp_arr.resample("1D").sum()
    return covid_esp_diff3,flights_esp_arr


#@st.cache
def model(covid,arrflights,x):
    model4 = SARIMAX(covid[3:-12], order=(4, 0, 4),exog=arrflights[2:])
    results_SARIMAX = model4.fit()
    prediction= results_SARIMAX.get_forecast(steps=len(x),exog=x)
    return prediction.predicted_mean
    

# Create a title, a subheader and let the reader know the data is loading.
st.title("Coronavirus forecast")
st.subheader("This is an app for predicting new number of coronavirus cases")
data_load_state = st.text('Loading data...')

# Load coronavirus cases data and flights data.
covid_esp_diff3, flights_esp_arr = data_load()

# Notify the reader what to do when data is loaded.
data_load_state.text('Data loaded')

#select number of days to forecast and create the slider buttons
days=st.radio("Choose the number of days to forecast:",("1","2","3","4"))
x=[]
for i in range(1,int(days)+1):
    x1 = st.slider(f'Number of arrival flights day '+str(i), 0, 1000, 600)
    x.append(x1)
z = np.reshape(x, (-1,1))
predictions=model(covid_esp_diff3,flights_esp_arr,z)

#enseñar predicciones
st.text("Predicted values for the next "+str(days)+" days")
st.write(predictions)
st.line_chart(covid_esp_diff3[3:-12].append(predictions))