import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import urllib.request
from datetime import datetime, timedelta, date
from sklearn.preprocessing import MinMaxScaler
import altair as alt

# Create a title, a header.
st.markdown("# Coronavirus forecast")
st.markdown("This is an app for predicting the number of daily new confirmed coronavirus cases and deaths for the next 14 days, \
            using publicly available data.")
st.markdown("Please select in the left sidebar the country where you want to make the predictions on. If you don't change anything else, \
            you see the forecast in a status quo scenario.")
st.markdown("Then you can play with the 2 values below, to see how modifying certain policies affects the forecasts.")
#st.markdown("")

# Chooose a country to predict the cases
st.sidebar.subheader("Select a country")
country = st.sidebar.selectbox('', ('Australia','Canada','China','Denmark','Finland','France','Germany','India','Indonesia',
                                    'Italy','Japan','Malaysia','Mexico','Norway','Philippines','Saudi Arabia','Singapore',
                                    'Spain','Sweden','Taiwan','Thailand','United Arab Emirates','United Kingdom','United States','Vietnam'))


var_c, varc = 'new_cases_', 'cases'
var_d, vard = 'new_deaths_', 'deaths'
initialdate = '2020-01-01'   # first day of the year, where most of our data starts
# moving intialdate by 6, since we later apply 7-day rolling mean to our data:
#initialdateshift = str(date.fromordinal(datetime.strptime(initialdate, '%Y-%m-%d').toordinal() + 6)) 
enddate = str(date.fromordinal(date.today().toordinal()-1))   # yesterday's date: last day of available data
datecol = 'date'
col_c = var_c + country
col_d = var_d + country


@st.cache
def get_endog(datecol, col):
    url1 = 'https://raw.githubusercontent.com/martaarozarena/KSchool-Master-Final-Project/master/data/endogenous.csv'
    covid_ctry_varR = pd.read_csv(url1, parse_dates=[datecol], index_col=[datecol], usecols=[datecol, col])
    return covid_ctry_varR

endog_ctry_c = get_endog(datecol, col_c)
endog_ctry_d = get_endog(datecol, col_d)


@st.cache
def get_exog(datecol, country):
    url2 = 'https://raw.githubusercontent.com/martaarozarena/KSchool-Master-Final-Project/master/data/exogenous.csv'
    exog = pd.read_csv(url2, parse_dates=[datecol], index_col=[datecol])
    exog = exog.loc[:, exog.columns.str.contains(country)]
    return exog

exog_ctry = get_exog(datecol, country)

last_test_value = int(exog_ctry.iloc[-1, exog_ctry.columns.str.contains('Testing|Contact')][0])
last_contact_value = int(exog_ctry.iloc[-1, exog_ctry.columns.str.contains('Testing|Contact')][1])

#@st.cache
def get_model(var):
    url3 = 'https://github.com/martaarozarena/KSchool-Master-Final-Project/raw/master/models/' + country.replace(" ", "") + '_' + var + 'model.pkl'
    smodel = joblib.load(urllib.request.urlopen(url3))
    return smodel

model_c = get_model(var_c)
model_d = get_model(var_d)

#url3 = 'https://github.com/martaarozarena/KSchool-Master-Final-Project/raw/master/models/' + country +'SARIMAXmodel.pkl'
#model = joblib.load(urllib.request.urlopen(url3))

#Load the country model and make the predictions
#url3 = 'https://github.com/martaarozarena/KSchool-Master-Final-Project/raw/master/models/' + country +'SARIMAXmodel.pkl'
#model = pd.read_pickle(url3)
#model = joblib.load(urllib.request.urlopen(url3))
#model = joblib.load(urllib.request.urlopen("https://github.com/hnballes/exogenas/raw/master/SpainSARIMAXmodel%20(copy%201).pkl"))
#model = joblib.load("/home/dsc/proyecto/data/{}SARIMAXmodel.pkl".format(country))
#model = joblib.load("/home/dsc/proyecto/data/SpainSARIMAXmodel.pkl")



# Select the values of the 2 diferent variables
st.sidebar.subheader("Testing policy")
st.sidebar.text("0 - no testing policy\n1 - only those who both (a) have symptoms AND (b) meet specific criteria (eg key workers)\n2 - testing of anyone showing Covid-19 symptoms\n3 - open public testing (eg drive through testing available to asymptomatic people")

testing = st.sidebar.slider(label='Select testing policy for next week:', min_value=0, max_value=3, value=last_test_value, step=1)
testing2 = st.sidebar.slider(label='Select testing policy for the week after:', min_value=0, max_value=3, value=last_test_value, step=1)


st.sidebar.subheader("Contact tracing policy after a positive diagnosis")
st.sidebar.text("0 - no contact tracing\n1 - limited contact tracing; not done for all cases\n2 - comprehensive contact tracing; done for all identified cases")

tracing = st.sidebar.slider(label='Select contact tracing policy for next week:', min_value=0, max_value=2, value=last_contact_value, step=1)
tracing2 = st.sidebar.slider(label='Select contact tracing policy for the week after:', min_value=0, max_value=2, value=last_contact_value, step=1)


# Building the future exogenous dataframe, interpolating from last observed values
forecastdays = 14
new_begin = str(date.fromordinal(datetime.strptime(enddate, '%Y-%m-%d').toordinal() + 1))
new_date = str(date.fromordinal(datetime.strptime(enddate, '%Y-%m-%d').toordinal() + forecastdays))
new_index = pd.date_range(initialdate, new_date, freq='D')
exog_futur = exog_ctry.reindex(new_index).interpolate()


# Change the values introduced by the user in the future exogenous dataframe
exog_futur.loc[date.today():date.fromordinal(date.today().toordinal()+6), "H2_Testing policy_{}".format(country)] = 7 * [testing]
exog_futur.loc[date.fromordinal(date.today().toordinal()+7): ,"H2_Testing policy_{}".format(country)] = 7 * [testing2]
exog_futur.loc[date.today():date.fromordinal(date.today().toordinal()+6), "H3_Contact tracing_{}".format(country)] = 7 * [tracing]
exog_futur.loc[date.fromordinal(date.today().toordinal()+7):, "H3_Contact tracing_{}".format(country)] = 7 * [tracing2]

#st.dataframe(exog_futur)


# Re-scale exogenous data with new added days:

sc_in_fc = MinMaxScaler(feature_range=(0, 1))
scaled_input_fc = sc_in_fc.fit_transform(exog_futur)
scaled_input_fc = pd.DataFrame(scaled_input_fc, index=exog_futur.index, columns=exog_futur.columns)
X_fc = scaled_input_fc
#st.line_chart(X_fc)


#Load the country model and make the predictions
#url3 = 'https://github.com/martaarozarena/KSchool-Master-Final-Project/raw/master/models/' + country +'SARIMAXmodel.pkl'
#model = pd.read_pickle(url3)
#model = joblib.load(urllib.request.urlopen(url3))
#model = joblib.load(urllib.request.urlopen("https://github.com/hnballes/exogenas/raw/master/SpainSARIMAXmodel%20(copy%201).pkl"))
#model = joblib.load("/home/dsc/proyecto/data/{}SARIMAXmodel.pkl".format(country))
#model = joblib.load("/home/dsc/proyecto/data/SpainSARIMAXmodel.pkl")


def fcast_plot(varx, vary, endog_ctry, col, model):
    # Scaling the endogenous data
    sc_out = MinMaxScaler(feature_range=(0, 1))
    scaled_output = sc_out.fit_transform(endog_ctry)
    scaled_output  = pd.Series(scaled_output.flatten(), index=endog_ctry.index, name=endog_ctry.columns[0])

    # Predictions
    results = model.get_forecast(steps=14, exog=X_fc[new_begin:new_date])
    mean_forecast = results.predicted_mean

    forecast14 = sc_out.inverse_transform(mean_forecast.values.reshape(-1,1))
    forecast14S = pd.Series(forecast14.flatten(), index=mean_forecast.index, name=varx+'forecast')

    # Get confidence intervals of  predictions
    confidence_intervals = results.conf_int()

    # Select lower and upper confidence limits
    lower_limits = confidence_intervals.loc[:,'lower ' + scaled_output.name]
    upper_limits = confidence_intervals.loc[:,'upper ' + scaled_output.name]

    # Apply inverse transform to get back to original scale
    forecast14_ll = sc_out.inverse_transform(lower_limits.values.reshape(-1,1))
    forecast14_llS = pd.Series(forecast14_ll.flatten(), index=lower_limits.index, name=varx+'forecast_ll')
    fcast_ll_df = forecast14_llS.to_frame().reset_index()

    forecast14_ul = sc_out.inverse_transform(upper_limits.values.reshape(-1,1))
    forecast14_ulS = pd.Series(forecast14_ul.flatten(), index=upper_limits.index, name=varx+'forecast_ul')
    fcast_ul_df = forecast14_ulS.to_frame().reset_index()

    conf_int = pd.concat([fcast_ll_df, fcast_ul_df.iloc[:, 1]], axis=1)

    last_endog = endog_ctry.tail(1)
    first_fut = forecast14S.head(1).to_frame()
    first_fut.columns = [endog_ctry.columns[0]]
    nexus = pd.concat([last_endog, first_fut]).reset_index()


    # Build dataframe for Altair graph
    past_rs = endog_ctry.reset_index()
    past_plt = alt.Chart(past_rs).mark_line().encode(
        x='date:T',
        y=col,
        tooltip=alt.Tooltip(col, format='.1f')
    ).interactive()

    nex = alt.Chart(nexus).mark_line(opacity=0.5, size=1.2).encode(
        x='index:T',
        y=col
    )

    future_rs = forecast14S.to_frame().reset_index()
    future_plt = alt.Chart(future_rs).mark_line(color='orange').encode(
        x=alt.X('index:T', axis=alt.Axis(title='Date')),
        y=alt.Y(varx+'forecast', axis=alt.Axis(title=None)),
        tooltip=alt.Tooltip(varx+'forecast', format='.1f')
    ).interactive()

    confint_plot = alt.Chart(conf_int).mark_area(opacity=0.2, color='orange').encode(
        alt.X('index:T'),
        alt.Y(varx+'forecast_ll'),
        alt.Y2(varx+'forecast_ul')
    )


    st.markdown("### Coronavirus confirmed {} 14 days forecast for {}".format(vary, country))
    st.markdown("Graph shows daily confirmed {}, showing the past in blue and the forecast in orange:".format(vary))

    st.altair_chart((past_plt + future_plt + nex + confint_plot).properties(
        width=650,
        height=350,
        title='{}: daily new  confirmed coronavirus {} (7-day rolling mean)'.format(country,vary)))

    st.markdown('Forecasted daily confirmed {}:'.format(vary))
    forecast14S_l = [ " %.0f" % elem for elem in forecast14S]
    st.text(str(forecast14S_l)[1:-1])


fcast_plot(var_c, varc, endog_ctry_c, col_c, model_c)
fcast_plot(var_d, vard, endog_ctry_d, col_d, model_d)