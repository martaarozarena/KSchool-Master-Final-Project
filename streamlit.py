import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import urllib.request
from datetime import datetime, timedelta, date
from sklearn.preprocessing import MinMaxScaler

#Personalize sidebar
#st.markdown(
#    """
#<style>
#.sidebar .sidebar-content {
#    background-image: linear-gradient(#22c9ae,#e6f3e7);
#    color: black;
#}
#</style>
#""",
#    unsafe_allow_html=True,
#)
#
##endogenous variables
#@st.cache
def data_load(selectedcountry):
    covid_url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
    covid = pd.read_csv(covid_url, parse_dates=['date'], index_col=['date'])
    # We filter the country, dates and the variable to predict

    country = selectedcountry
    variable = 'new_cases' #we could make it a selectbox
    initialdate = '2020-01-01'   # first day of the year, where most of our data starts
    initialdateshift = str(date.fromordinal(datetime.strptime(initialdate, '%Y-%m-%d').toordinal() + 6))
    enddate = str(date.fromordinal(date.today().toordinal()-1))   # yesterday's date: last day of available data

    # Filtering country and dates
    covid_ctry = covid[covid['location']==country]
    covid_ctry = covid_ctry.loc[initialdate:enddate]

    # Filter the variable to predict and applying 7-day rolling mean
    covid_ctry_var = covid_ctry[variable]
    covid_ctry_varR = covid_ctry_var.rolling(7).mean().dropna()
    
    return covid_ctry_varR


# Create a title, a subheader.
st.title("Coronavirus forecast")
st.subheader("This is an app for predicting new number of coronavirus cases during two weeks according to public data. ")

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

#extend the last value of the futur exogenous variables and add the 2 values the user introduced
#create the dataframe of the next 14 days and join it to the old exogenous

initialdate = '2020-01-01'   # first day of the year, where most of our data starts
initialdateshift = str(date.fromordinal(datetime.strptime(initialdate, '%Y-%m-%d').toordinal() + 6))
enddate = str(date.fromordinal(date.today().toordinal()-1))   # yesterday's date: last day of available data

exogenas=pd.read_csv("/home/dsc/proyecto/data/exogenas.csv", parse_dates=[0], index_col=[0])
exogenas= exogenas.loc[:, exogenas.columns.str.contains(country)]
exogenas = exogenas.loc[initialdateshift:enddate]

#futur exogenous
new_index = pd.date_range(date.fromordinal(date.today().toordinal()), date.fromordinal(date.today().toordinal()+13))
futur=pd.DataFrame(exogenas.iloc[-1:,:],index=new_index)
futur.iloc[0,:]=exogenas.iloc[-1,:]
futur.fillna(method="ffill",inplace=True)


#change the values introduced by the user in the futur exogenous dataframe
futur.loc[date.today():date.fromordinal(date.today().toordinal()+6),"testingpolicy_{}".format(country)]=7*[testing]
futur.loc[date.fromordinal(date.today().toordinal()+7):,"testingpolicy_{}".format(country)]=7*[testing2]
futur.loc[date.today():date.fromordinal(date.today().toordinal()+6),"contacttracing_{}".format(country)]=7*[tracing]
futur.loc[date.fromordinal(date.today().toordinal()+7):,"contacttracing_{}".format(country)]=7*[tracing2]

st.dataframe(futur)


# Re-scale exogenous date with new added days:

sc_in_fc = MinMaxScaler(feature_range=(0, 1))
exogen_joined=pd.concat([exogenas,futur])
scaled_input_fc = sc_in_fc.fit_transform(exogen_joined)
scaled_input_fc = pd.DataFrame(scaled_input_fc, index=exogen_joined.index ,columns=exogen_joined.columns)
X_fc = scaled_input_fc
st.line_chart(X_fc)


#Load right model and make the predictions
model = joblib.load(urllib.request.urlopen("https://drive.google.com/uc?export=download&id=1shJ2zgyYwaVgd_w9h6aOzbo5o5LtoCZ6"))
#model = joblib.load("/home/dsc/proyecto/data/{}SARIMAXmodel.pkl".format(country))
#model = joblib.load("/home/dsc/proyecto/data/SpainSARIMAXmodel.pkl")
#predictions
results=model.get_forecast(steps=14,exog=exogen_joined.iloc[-14:,:])
st.header("Number of cases next two weeks:")
st.dataframe(results.predicted_mean.rename("Prediction"))
st.subheader("Graph")
new_cases=data_load(country)
st.line_chart(results.predicted_mean)
st.line_chart(new_cases)