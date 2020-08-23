import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import urllib.request
from datetime import datetime, timedelta, date
    
varc1=[1]
exogenas=pd.read_csv("/home/dsc/proyecto/data/exogenas.csv", parse_dates=[0], index_col=[0])

# Create a title, a subheader.
st.title("Coronavirus forecast")
st.subheader("This is an app for predicting new number of coronavirus cases during two weeks")

#chooose a country to predict the cases
st.sidebar.title("Choose a country")
country = st.sidebar.selectbox("", ("Denmark","Germany","Spain","Finland","Italy","Sweden","France","Norway","United Kingdom","United States","Canada","Mexico","Australia","Indonesia","Malaysia","Philippines","Thailand","Hong Kong","Vietnam","China","India","Japan","Singapore","Taiwan","Saudi Arabia","United Arab Emirates"))

#select the values of the 2 diferent variables
st.sidebar.subheader("testing policy")
st.sidebar.text("0 - no testing policy\n1 - only those who both (a) have symptoms AND (b) meet specific criteria (eg key workers)\n2 - testing of anyone showing Covid-19 symptoms\n3 - open public testing (eg drive through testing available to asymptomatic people")

testing = st.sidebar.slider('choose the testing policy applied next week:', 0, 3, 1)
testing2 = st.sidebar.slider('choose the testing policy applied the week after:', 0, 3, 1)



st.sidebar.subheader("Record government policy on contact tracing after a positive diagnosis")
st.sidebar.text("0 - no contact tracing\n1 - limited contact tracing; not done for all cases\n2 - comprehensive contact tracing; done for all identified cases")

tracing = st.sidebar.slider('choose the contact tracing policy applied next week:', 0, 3, 1)
tracing2 = st.sidebar.slider('choose the contact tracing policy applied the week after:', 0, 3, 1)

#extend the last value of the exogenous variables and add the 2 values the user introduced
#create the dataframe of the next 14 days
forecastexog=exogenas.loc[:,exogenas.columns.str.contains(country)]
forecastexog=forecastexog.iloc[-1:,:]
new_index = pd.date_range(date.fromordinal(date.today().toordinal()), date.fromordinal(date.today().toordinal()+13))
futur=pd.DataFrame(forecastexog,index=new_index)
futur.iloc[0,:]=forecastexog.iloc[-1,:]
futur.fillna(method="ffill",inplace=True)


#change the values introduced by the user
futur.loc[date.today():date.fromordinal(date.today().toordinal()+6),"testingpolicy_{}".format(country)]=7*[testing]
futur.loc[date.fromordinal(date.today().toordinal()+7):,"testingpolicy_{}".format(country)]=7*[testing2]
futur.loc[date.today():date.fromordinal(date.today().toordinal()+6),"contacttracing_{}".format(country)]=7*[tracing]
futur.loc[date.fromordinal(date.today().toordinal()+7):,"contacttracing_{}".format(country)]=7*[tracing2]


#Load right model and make the predictions
#model = joblib.load(urllib.request.urlopen("https://drive.google.com/uc?export=download&id=1shJ2zgyYwaVgd_w9h6aOzbo5o5LtoCZ6"))
model = joblib.load("/home/dsc/proyecto/data/{}SARIMAXmodel.pkl".format(country))
#scaler needed?
results=model.get_forecast(steps=14,exog=futur)
st.title("Number of cases next two weeks")
st.dataframe(results.predicted_mean.rename("Prediction"))
st.subheader("Graph")
st.line_chart(results.predicted_mean)