#!/usr/bin/env python3
# Importing all necessary libraries

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
import datatools
import time


def create_model(countryin, varin):
    country = countryin
    var = varin
    variable = var + '_'
    col = variable + country
    datecol = 'date'
    initialdate = '2020-01-01'   # first day of the year, where most of our data starts


    pltpath = './plots/' + country + '_' + variable + '_'



    # We read the endogenous data (coronavirus data) (for now, from a local file)

    covid_ctry_varR = pd.read_csv('./data/endogenous.csv', parse_dates=[datecol], index_col=[datecol], usecols=[datecol, col])
    enddate = str(date.fromordinal(covid_ctry_varR.tail(1).index[0].toordinal())) # last day of available data


    # We now read the exogenous data (for now, from a local file):

    exogenous = pd.read_csv('./data/exogenous.csv', parse_dates=[datecol], index_col=[datecol])

    # We now need to filter the country:
    exogenous_ctryR = exogenous.loc[:, exogenous.columns.str.contains(country)]


    print('************* Read endog and exog csvs')


    # Normalize endogenous and exogenous data

    sc_in = MinMaxScaler(feature_range=(0, 1))
    scaled_input = sc_in.fit_transform(exogenous_ctryR)
    scaled_input = pd.DataFrame(scaled_input, index=exogenous_ctryR.index, columns=exogenous_ctryR.columns)
    X = scaled_input



    sc_out = MinMaxScaler(feature_range=(0, 1))
    scaled_output = sc_out.fit_transform(covid_ctry_varR)
    scaled_output  = pd.Series(scaled_output.flatten(), index=covid_ctry_varR.index, name=covid_ctry_varR.columns[0])
    y = scaled_output.resample('1D').sum()


    print('************* Endog and exog normalized')


    # # Split endogenous and exogenous data into train/test


    # We are going to use 85% for training, since most of the series is the big curve, 
    # and then we have the smaller changes in coronavirus cases towards the end
    train_size = int(len(covid_ctry_varR) * 0.85)
    test_size = len(covid_ctry_varR) - train_size
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]


    print('************* Data split into train and test')


    # # Estimate orders p and q of ARIMA model, using auto_arima


    # Perform different auto_arima searches and sort results by lowest AIC:
    result_table = datatools.autoarimas(y_train, X_train)

    print('************* Autoarimas done')

    # # Perform cross-validation on top 3 models and select the best. Then train and fit SARIMAX model with the one that gave best results


    # Extract top 3 models from previous step (first 3 elements, since they are sorted):
    model1 = ARIMA(order=result_table.iloc[0,0])
    model2 = ARIMA(order=result_table.iloc[1,0])
    model3 = ARIMA(order=result_table.iloc[2,0])

    print('************* Arima instances done')


    best_order = datatools.cross_val(y_train, X_train, model1, model2, model3)

    print('************* Cross validation done')



    best_model = sm.tsa.statespace.SARIMAX(y_train, order=best_order, exog=X_train)
    results = best_model.fit(full_output=False, disp=False)


    print('************* Sarimax model fitted')

    # # Perform/plot in-sample prediction and out-of-sample forecast and evaluate model MAE

    in_predictions, mean_forecast = datatools.in_out_fcast_plot(results, test_size, y, y_test, X_train, X_test)
    plt.close()

    print('************* In sample and out of sample forecast done')

    # # Scale data back to original values and plot


    trainPredict = sc_out.inverse_transform(in_predictions.values.reshape(-1,1))
    trainPredictS = pd.Series(trainPredict.flatten(), index=covid_ctry_varR[:train_size].index, name=covid_ctry_varR.columns[0])



    testPredict = sc_out.inverse_transform(mean_forecast.values.reshape(-1,1))
    testPredictS = pd.Series(testPredict.flatten(), index=covid_ctry_varR[train_size:].index, name=covid_ctry_varR.columns[0])
    testPredictS = testPredictS.clip(lower=0)

    mae_orig = mean_absolute_error(covid_ctry_varR[train_size:], testPredictS)
    mae_orig_perc = mae_orig / covid_ctry_varR[train_size:].mean()[0]
    print("Test MAE (original scale): {:.1f}".format(mae_orig))

    print('************* Did inverse transform to scale back data')


    # Plot the data in original scale
    plt.rcParams["figure.figsize"] = (20, 8)
    plt.plot(covid_ctry_varR.index, covid_ctry_varR, label='observed')

    # Plot mean predictions in original scale
    plt.plot(trainPredictS.index, trainPredictS, color='lightcoral', label='in-sample predictions (train)')

    plt.plot(testPredictS.index, testPredictS, color='r', label='test predictions')

    # Set labels, legends and show plot
    plt.xlabel('Date')
    plt.title('Coronavirus {} 7-day rolling mean in-sample and test predictions (original scale) for {}'.format(var,country))
    plt.text(0.82, 0.17, 'MAE: {:.1f}'.format(mae_orig), transform = plt.gcf().transFigure, fontsize=13)
    plt.rcParams["figure.figsize"] = (20, 8)
    plt.legend()
    plt.savefig(pltpath + 'inoutpredorig.png')
    plt.close()

    

    print('************* Plotted in-sample prediction')


    # Update model with test observations, to get it ready for future forecasts

    # Update model with test sample:
    res_updated = results.append(y_test, exog=X_test, full_output=False, disp=False)

    print('************* Updated model with test observations')



    # Save model, after updating it with test sample:
    
    # Set model name
    filename = './models/' + country.replace(" ", "") + '_' + variable + 'model.pkl'

    joblib.dump(res_updated, filename)

    print('************* Saved model, including also the test observations')

    # # Perform forecast

    forecastdays = 14
    new_begin = str(date.fromordinal(datetime.strptime(enddate, '%Y-%m-%d').toordinal() + 1))
    new_date = str(date.fromordinal(datetime.strptime(enddate, '%Y-%m-%d').toordinal() + forecastdays))
    new_index = pd.date_range(initialdate, new_date, freq='D')
    exog_conc = exogenous_ctryR.reindex(new_index).interpolate()

    print('************* Concatenated 14 new data lines to exog')



    # Re-scale exogenous date with new added days:
    sc_in_fc = MinMaxScaler(feature_range=(0, 1))
    scaled_input_fc = sc_in_fc.fit_transform(exog_conc)
    scaled_input_fc = pd.DataFrame(scaled_input_fc, index=exog_conc.index, columns=exog_conc.columns)
    X_fc = scaled_input_fc


    print('************* Re-scaled again exog with 14 new data lines')



    # Generate out of sample forecast

    forecast = res_updated.get_forecast(steps=forecastdays, exog=X_fc[new_begin:new_date])

    # Extract prediction mean and assign negative values to zero (negative numbers don't make sense here):
    mean_forecast = forecast.predicted_mean
    mean_forecast = mean_forecast.clip(lower=0)

    # Get confidence intervals of  predictions
    confidence_intervals = forecast.conf_int()

    # Select lower and upper confidence limits
    lower_limits = confidence_intervals.loc[:,'lower ' + y.name]
    upper_limits = confidence_intervals.loc[:,'upper ' + y.name]

    print('************* Done out of sample 14 days forecast')


    forecast14 = sc_out.inverse_transform(mean_forecast.values.reshape(-1,1))
    forecast14S = pd.Series(forecast14.flatten(), index=mean_forecast.index, name='new_cases_forecast')
    forecast14S = forecast14S.clip(lower=0)

    forecast14_ll = sc_out.inverse_transform(lower_limits.values.reshape(-1,1))
    forecast14_llS = pd.Series(forecast14_ll.flatten(), index=lower_limits.index, name='new_cases_forecast_ll')

    forecast14_ul = sc_out.inverse_transform(upper_limits.values.reshape(-1,1))
    forecast14_ulS = pd.Series(forecast14_ul.flatten(), index=upper_limits.index, name='new_cases_forecast_ul')

    print('************* Did inverse transform to scale back data, after 14 days forecast')



    # Plot the data
    plt.plot(covid_ctry_varR.index, covid_ctry_varR, label='observed')

    # Plot the mean predictions
    plt.plot(forecast14S.index, forecast14S, color='r', label='forecast')

    # Shade the area between your confidence limits
    plt.fill_between(forecast14_llS.index, forecast14_llS, forecast14_ulS, color='pink')

    # set labels, legends and show plot
    plt.xlabel('Date')
    plt.title('Coronavirus {} (7-day rolling mean) 14 days forecast (original scale) for {}'.format(var,country))
    plt.text(0.82, 0.17, 'Test MAE: {:.1f}'.format(mae_orig), transform = plt.gcf().transFigure, fontsize=13)
    plt.legend()
    plt.savefig(pltpath + 'outfcastorig.png')
    plt.close()

    print('************* Plotted 14 days forecast in original scale')


    # Print forecasted values:

    forecast14S_l = ["%.1f" % elem for elem in forecast14S]
    print('Next 14 days forecast values: ', forecast14S_l)

    
    return best_order, mae_orig, mae_orig_perc

