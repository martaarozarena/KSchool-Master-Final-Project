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


variable = 'new_cases_'
initialdate = '2020-01-01'   # first day of the year, where most of our data starts
# moving intialdate by 6, since we later apply 7-day rolling mean to our data:
initialdateshift = str(date.fromordinal(datetime.strptime(initialdate, '%Y-%m-%d').toordinal() + 6)) 
enddate = str(date.fromordinal(date.today().toordinal()-1))   # yesterday's date: last day of available data

#def data_load(selectedcountry):
    
#    covid_url = 'https://covid.ourworldindata.org/data/owid-covid-data.csv'
#    covid = pd.read_csv(covid_url, parse_dates=['date'], index_col=['date'])

    # We filter the country, dates and the variable to predict

#    country = selectedcountry
#    variable = 'new_cases' #we could make it a selectbox
#    initialdate = '2020-01-01'   # first day of the year, where most of our data starts
#    initialdateshift = str(date.fromordinal(datetime.strptime(initialdate, '%Y-%m-%d').toordinal() + 6))
#    enddate = str(date.fromordinal(date.today().toordinal()-1))   # yesterday's date: last day of available data

    # Filtering country and dates
#    covid_ctry = covid[covid['location']==country]
#    covid_ctry = covid_ctry.loc[initialdate:enddate]

    # Filter the variable to predict and applying 7-day rolling mean
#    covid_ctry_var = covid_ctry[variable]
#    covid_ctry_varR = covid_ctry_var.rolling(7).mean().dropna()
    
#    return covid_ctry_varR


# Create a title, a subheader.
st.title("Coronavirus forecast")
st.subheader("This is an app for predicting the number of new coronavirus cases for the next 2 weeks, using publicly available data. ")

#chooose a country to predict the cases
st.sidebar.title("Choose a country")
country = st.sidebar.selectbox("", ("Denmark","Germany","Spain","Finland","Italy","Sweden","France","Norway","United Kingdom",
                                    "United States","Canada","Mexico","Australia","Indonesia","Malaysia","Philippines","Thailand",
                                    "Vietnam","China","India","Japan","Singapore","Taiwan","Saudi Arabia","United Arab Emirates"))

#select the values of the 2 diferent variables
st.sidebar.subheader("Testing policy")
st.sidebar.text("0 - no testing policy\n1 - only those who both (a) have symptoms AND (b) meet specific criteria (eg key workers)\n2 - testing of anyone showing Covid-19 symptoms\n3 - open public testing (eg drive through testing available to asymptomatic people")

testing = st.sidebar.slider('Select testing policy for next week:', 0, 3, 1)
testing2 = st.sidebar.slider('Select testing policy for the week after:', 0, 3, 1)



st.sidebar.subheader("Record government policy on contact tracing after a positive diagnosis")
st.sidebar.text("0 - no contact tracing\n1 - limited contact tracing; not done for all cases\n2 - comprehensive contact tracing; done for all identified cases")

tracing = st.sidebar.slider('Select contact tracing policy for next week:', 0, 3, 1)
tracing2 = st.sidebar.slider('Select contact tracing policy for the week after:', 0, 3, 1)

#extend the last value of the futur exogenous variables and add the 2 values the user introduced
#create the dataframe of the next 14 days and join it to the old exogenous

#initialdate = '2020-01-01'   # first day of the year, where most of our data starts
#initialdateshift = str(date.fromordinal(datetime.strptime(initialdate, '%Y-%m-%d').toordinal() + 6))
#enddate = str(date.fromordinal(date.today().toordinal()-1))   # yesterday's date: last day of available data

datecol = 'date'
col = variable + country
url1 = 'https://raw.githubusercontent.com/martaarozarena/KSchool-Master-Final-Project/master/data/endogenous.csv'
covid_ctry_varR = pd.read_csv(url1, parse_dates=[datecol], index_col=[datecol], usecols=[datecol, col])

sc_out = MinMaxScaler(feature_range=(0, 1))
scaled_output = sc_out.fit_transform(covid_ctry_varR)
scaled_output  = pd.Series(scaled_output.flatten(), index=covid_ctry_varR.index, name=covid_ctry_varR.columns[0])


url2 = 'https://raw.githubusercontent.com/martaarozarena/KSchool-Master-Final-Project/master/data/exogenous.csv'
exog = pd.read_csv(url2, parse_dates=[datecol], index_col=[datecol])
exog = exog.loc[:, exog.columns.str.contains(country)]
#exog = exog.loc[initialdateshift:enddate]

#future exogenous
#new_index = pd.date_range(date.fromordinal(date.today().toordinal()), date.fromordinal(date.today().toordinal()+13))
#futur = pd.DataFrame(exogenas.iloc[-1:,:], index=new_index)
#futur.iloc[0,:] = exog.iloc[-1,:]
#futur.fillna(method="ffill",inplace=True)
forecastdays = 14
new_begin = str(date.fromordinal(datetime.strptime(enddate, '%Y-%m-%d').toordinal() + 1))
new_date = str(date.fromordinal(datetime.strptime(enddate, '%Y-%m-%d').toordinal() + forecastdays))
new_index = pd.date_range(initialdate, new_date, freq='D')
exog_futur = exog.reindex(new_index).interpolate()


#change the values introduced by the user in the future exogenous dataframe
exog_futur.loc[date.today():date.fromordinal(date.today().toordinal()+6), "H2_Testing policy_{}".format(country)] = 7 * [testing]
exog_futur.loc[date.fromordinal(date.today().toordinal()+7): ,"H2_Testing policy_{}".format(country)] = 7 * [testing2]
exog_futur.loc[date.today():date.fromordinal(date.today().toordinal()+6), "H3_Contact tracing_{}".format(country)] = 7 * [tracing]
exog_futur.loc[date.fromordinal(date.today().toordinal()+7):, "H3_Contact tracing_{}".format(country)] = 7 * [tracing2]

st.dataframe(exog_futur)


# Re-scale exogenous data with new added days:

sc_in_fc = MinMaxScaler(feature_range=(0, 1))
#exogen_joined=pd.concat([exog,futur])
scaled_input_fc = sc_in_fc.fit_transform(exog_futur)
scaled_input_fc = pd.DataFrame(scaled_input_fc, index=exog_futur.index, columns=exog_futur.columns)
X_fc = scaled_input_fc
st.line_chart(X_fc)


#Load right model and make the predictions
#url3 = 'https://github.com/martaarozarena/KSchool-Master-Final-Project/blob/master/models/' + country +'SARIMAXmodel.pkl?raw=true'
#model = pd.read_pickle(url3)
model = joblib.load('./models/' + country + 'SARIMAXmodel.pkl')
#model = joblib.load(urllib.request.urlopen("https://github.com/hnballes/exogenas/raw/master/SpainSARIMAXmodel%20(copy%201).pkl"))
#model = joblib.load("/home/dsc/proyecto/data/{}SARIMAXmodel.pkl".format(country))
#model = joblib.load("/home/dsc/proyecto/data/SpainSARIMAXmodel.pkl")
#predictions
results = model.get_forecast(steps=14, exog=exog_futur[new_begin:new_date])
mean_forecast = results.predicted_mean

forecast14 = sc_out.inverse_transform(mean_forecast.values.reshape(-1,1))
forecast14S = pd.Series(forecast14.flatten(), index=mean_forecast.index, name='new_cases_forecast')


st.header("Number of new coronavirus cases for the next two weeks:")
st.dataframe(forecast14S.rename("Forecast"))
st.subheader("Graph")
#new_cases = data_load(country)
st.line_chart(forecast14S)
st.line_chart(covid_ctry_varR)