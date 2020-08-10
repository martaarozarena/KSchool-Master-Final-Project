import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import urllib.request
    
varc1=[1]
# Create a title, a subheader and let the reader know the data is loading.
st.title("Coronavirus forecast")
st.subheader("This is an app for predicting new number of coronavirus cases during a week")

#select number of days to forecast and create the slider buttons
days=7
x=[]
for i in range(1,int(days)+1):
    x1 = st.slider(f'Number of arrival flights day '+str(i), 0, 1000, 600)
    x.append(x1)
flightpredictions = np.reshape(x, (-1,1))


#select the values of the 2 diferent variables
st.subheader("testing policy")
st.text("0 - no testing policy\n1 - only those who both (a) have symptoms AND (b) meet specific criteria (eg key workers)\n2 - testing of anyone showing Covid-19 symptoms\n3 - open public testing (eg drive through testing available to asymptomatic people")

testing = st.slider('choose the testing policy applied next week:', 0, 3, 1)
testingpolicy=pd.Series(7*[testing])

st.subheader("Record government policy on contact tracing after a positive diagnosis")
st.text("0 - no contact tracing\n1 - limited contact tracing; not done for all cases\n2 - comprehensive contact tracing; done for all identified cases")

tracing = st.slider('choose the contact tracing policy applied next week:', 0, 3, 1)
tracingpolicy=pd.Series(7*[testing])

#Load model and make the predictions

model = joblib.load(urllib.request.urlopen("https://drive.google.com/uc?export=download&id=1NeoAEr1Ksa_6Mwu1mNABZqoQlrlmvdrd"))
results=model.get_forecast(steps=7,exog=flightpredictions)
st.title("Forecast for next week")
st.dataframe(results.predicted_mean)
st.line_chart(results.predicted_mean)
