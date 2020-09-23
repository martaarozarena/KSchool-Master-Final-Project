#!/usr/bin/env python3
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.rcParams["figure.figsize"] = (15, 5)
from pmdarima import auto_arima, model_selection, ARIMA
from pmdarima.arima.utils import ndiffs
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta, date
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import joblib


# In[23]:



countries = 'Denmark|Germany|Spain|Finland|Italy|Sweden|France|Norway|United Kingdom'             '|United States|Canada|Mexico'             '|Australia|Indonesia|Malaysia|Philippines|Thailand|Vietnam|China|India|Japan|Singapore|Taiwan'             '|Saudi Arabia|United Arab Emirates'

datecol = 'date'
exog=pd.read_csv("data/exogenous.csv", parse_dates=[datecol], index_col=[datecol])
endog=pd.read_csv("data/endogenous.csv", parse_dates=[datecol], index_col=[datecol])
cases="cases"

for cases in ["cases","deaths"]:

    for exogcountries in countries.split('|'):
        country=exogcountries.replace(" ","")
        #filename = './models/' + country + 'SARIMAXmodel.pkl'
        filename = './models/' + country + '_new_'+cases+'_model.pkl'
        model = joblib.load(filename)
        if model.fittedvalues.index[-1]== exog.index[-1]:
            print("up to date")
        else:

            exogmodel=exog.loc[:, exog.columns.str.contains(exogcountries)]
            endogmodel=endog.loc[:, endog.columns.str.contains("new_{}_{}".format(cases,exogcountries))]
            #exogmodel=exogmodel.iloc[-1:,:]
            sc_in = MinMaxScaler(feature_range=(0, 1))
            scaled_input = sc_in.fit_transform(exogmodel)
            scaled_input = pd.DataFrame(scaled_input, index=exogmodel.index, columns=exogmodel.columns)
            X_update = scaled_input
            X_update=X_update.iloc[-1:,:]

            sc_out = MinMaxScaler(feature_range=(0, 1))
            scaled_output = sc_out.fit_transform(endogmodel)
            scaled_output  = pd.Series(scaled_output.flatten(), index=endogmodel.index, name=endogmodel.columns[0])
            y = scaled_output.resample('1D').sum()
            y=y.iloc[-1:]
            modelupdated=model.append(y, exog=X_update, refit=True,)
            joblib.dump(modelupdated, filename)
            print("{} new_{} updated".format(country,cases))

