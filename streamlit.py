import streamlit as st
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

@st.cache
def data_load():
    
    covid=pd.read_csv("/home/dsc/Downloads/owid-covid-data 20200713.csv", parse_dates=["date"])  
    covid_eur = covid[covid['continent']=='Europe']
    covid_eur['new_cases'] = covid_eur['new_cases'].fillna(0)
    covid_esp=covid_eur[covid_eur["location"]=="Spain"].reset_index()
    covid_esp.index = covid_esp.date
    covid_esp_newcases=covid_esp["new_cases"]
    covid_esp_newcases = covid_esp_newcases.resample('D').sum()
    covid_esp["MA3"]=covid_esp.new_cases.rolling(3).mean()
    covid_esp_ma3=covid_esp["MA3"]
    covid_esp_diff3 = covid_esp_newcases - covid_esp_ma3
    flights_esp = pd.read_csv("/home/dsc/Downloads/vuelos_esp.csv",parse_dates=["FLT_DATE"],index_col="FLT_DATE")
    flights_esp=flights_esp["FLT_ARR_1"]
    flights_esp=flights_esp.resample("1D").sum()
    return covid_esp_diff3,flights_esp

#@st.cache
def model(covid_esp_diff3,flights_esp,x):
    model4 = SARIMAX(covid_esp_diff3[3:-12], order=(4, 0, 4),exog=flights_esp[2:])
    results_SARIMAX = model4.fit()
    prediction= results_SARIMAX.get_forecast(steps=len(x),exog=x)
    return prediction.predicted_mean
    

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
covid_esp_diff3, flights_esp = data_load()
# Notify the reader that the data was successfully loaded.
data_load_state.text('Choose the number of flights for the next 3 days')
#creamos el modelo
x1 = st.slider('number of flights day1', 0, 1000, 600)
x2 = st.slider('number of flights day2', 0, 1000, 600)
x3 = st.slider('number of flights day3', 0, 1000, 600)
x = [x1,x2,x3]
predictions=model(covid_esp_diff3,flights_esp,x)
#enseñar predicciones
st.write(predictions)
st.line_chart(covid_esp_diff3)