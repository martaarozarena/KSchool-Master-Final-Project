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
    

# Create a title, a subheader and let the reader know the data is loading.
st.title("Coronavirus forecast")
st.subheader("This is an app to help visualizing the coronavirus cases forecasting")
data_load_state = st.text('Loading data...')

# Load coronavirus cases data and flights data.
covid_esp_diff3, flights_esp = data_load()

# Notify the reader what to do when data is loaded.
data_load_state.text('Data loaded')

#select number of days to forecast and create the slider buttons
days=st.radio("choose the number of days to forecast:",("1","2","3","4"))
i=1
x=[]
for i in range(int(days)):
    x1 = st.slider(f'number of flights day '+str(i), 0, 1000, 600)
    i=i+1
    x.append(x1)
predictions=model(covid_esp_diff3,flights_esp,x)

#enseñar predicciones
st.text("Predicted values for the next "+str(days)+" days")
st.write(predictions)
st.line_chart(covid_esp_diff3[3:-12].append(predictions))