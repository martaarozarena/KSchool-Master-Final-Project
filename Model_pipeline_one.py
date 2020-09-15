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
import argparse
import time

# Initialising time:
start_time = time.time()

# Read the coutry from command line:
parser = argparse.ArgumentParser()
parser.add_argument('countryin')
args = parser.parse_args()

# Read endogenous and exogenous data and filter country/dates

# We filter the country, the variable to predict and the dates

country = args.countryin
variable = 'new_cases_'
col = variable + country
datecol = 'date'
initialdate = '2020-01-01'   # first day of the year, where most of our data starts
# moving intialdate by 6, since we later apply 7-day rolling mean to our data:
initialdateshift = str(date.fromordinal(datetime.strptime(initialdate, '%Y-%m-%d').toordinal() + 6)) 
enddate = str(date.fromordinal(date.today().toordinal()-1))   # yesterday's date: last day of available data

pltpath = './plots/' + country + '_'



# We read the endogenous data (coronavirus data) (for now, from a local file)

covid_ctry_varR = pd.read_csv('./data/endogenous.csv', parse_dates=[datecol], index_col=[datecol], usecols=[datecol, col])



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


# # Stationarity: estimate differencing term (d)

# Performing different tests to estimate best value of 'd'

## Adf Test
#d_adf = ndiffs(y_train, test='adf')
#print('ADF test: ', d_adf)

# KPSS test
#d_kpss = ndiffs(y_train, test='kpss')
#print('KPSS test: ', d_kpss)

# PP test:
#d_pp = ndiffs(y_train, test='pp')
#print('PP test: ', d_pp)


# Test stationarity and print results of ADF test:
#datatools.test_stationarity(y_train.diff().dropna())


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
results = best_model.fit()
#print(results.summary())
#results.plot_diagnostics(figsize=(15,10));

print('************* Sarimax model fitted')

# # Perform/plot in-sample prediction and out-of-sample forecast and evaluate model MAE

in_predictions, mean_forecast = datatools.in_out_fcast_plot(results, test_size, y, y_test, X_train, X_test)

# set title and show plot
#plt.title('Coronavirus 7-day rolling mean in-sample and test predictions (0-1 scale) for {}'.format(country))
#plt.savefig(pltpath + 'inoutpred.png')
#plt.show()
plt.close()

print('************* In sample and out of sample forecast done')

# # Scale data back to original values and plot


trainPredict = sc_out.inverse_transform(in_predictions.values.reshape(-1,1))
trainPredictS = pd.Series(trainPredict.flatten(), index=covid_ctry_varR[:train_size].index, name=covid_ctry_varR.columns[0])



testPredict = sc_out.inverse_transform(mean_forecast.values.reshape(-1,1))
testPredictS = pd.Series(testPredict.flatten(), index=covid_ctry_varR[train_size:].index, name=covid_ctry_varR.columns[0])

print('************* Did inverse transform to scale back data')


# Plot the data in original scale
plt.rcParams["figure.figsize"] = (20, 8)
plt.plot(covid_ctry_varR.index, covid_ctry_varR, label='observed')

# Plot mean predictions in original scale
plt.plot(trainPredictS.index, trainPredictS, color='lightcoral', label='in-sample predictions (train)')

plt.plot(testPredictS.index, testPredictS, color='r', label='test predictions')

# Set labels, legends and show plot
plt.xlabel('Date')
plt.title('Coronavirus 7-day rolling mean in-sample and test predictions (original scale) for {}'.format(country))
plt.rcParams["figure.figsize"] = (20, 8)
plt.legend()
plt.savefig(pltpath + 'inoutpredorig.png')
#plt.show()
plt.close()



print("Test MAE (original scale): %.3f" % mean_absolute_error(covid_ctry_varR[train_size:], testPredictS))


# # Save model so we can then update with future values


# Set model name
filename = './models/' + country + 'SARIMAXmodel.pkl'

# Pickle it
joblib.dump(results, filename)

print('************* Plotted in-sample prediction and saved model')



# Load the model back in
#loaded_model = joblib.load(filename)


#loaded_model.summary()


# # Update model with test observations, to get it ready for future forecasts



# Update model with test sample and re-fit parameters:
res_updated = results.append(y_test, exog=X_test, refit=True)

print('************* Updated model with test observations')



# Print summary of updated model and plot diagnostics, to confirm everything working as expected:
#print(res_updated.summary())
#res_updated.plot_diagnostics(figsize=(15,10));



# Plot the updated data
#plt.rcParams["figure.figsize"] = (20, 8)
#plt.plot(y.index, y, label='observed')

# plot in-sample predictions (train+test)
#plt.plot(res_updated.fittedvalues.index, res_updated.fittedvalues, color='lightcoral', label= 'updated model')
#plt.xlabel('Date')
#plt.title('Coronavirus 7-day rolling mean in-sample predictions (after updating model with test sample)')
#plt.legend()
#plt.show()


# Save model again, after updating it with test sample:

joblib.dump(res_updated, filename)

print('************* Saved model again, now including test observations')

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
#X_fc.plot()

print('************* Re-scaled again exog with 14 new data lines')



# Generate out of sample forecast

forecast = res_updated.get_forecast(steps=forecastdays, exog=X_fc[new_begin:new_date])

# Extract prediction mean
mean_forecast = forecast.predicted_mean

# Get confidence intervals of  predictions
confidence_intervals = forecast.conf_int()

# Select lower and upper confidence limits
lower_limits = confidence_intervals.loc[:,'lower ' + y.name]
upper_limits = confidence_intervals.loc[:,'upper ' + y.name]

# Print best estimate  predictions
#print(mean_forecast.values)

# plot the data
#plt.plot(y.index, y, label='observed')

# plot your mean predictions
#plt.plot(mean_forecast.index, mean_forecast, color='r', label='forecast')

# shade the area between your confidence limits
#plt.fill_between(lower_limits.index, lower_limits, upper_limits, color='pink')

# set labels, legends and show plot
#plt.xlabel('Date')
#plt.title('Coronavirus 7-day rolling mean forecast')
#plt.legend()
#plt.show()

print('************* Done out of sample 14 days forecast')


forecast14 = sc_out.inverse_transform(mean_forecast.values.reshape(-1,1))
forecast14S = pd.Series(forecast14.flatten(), index=mean_forecast.index, name='new_cases_forecast')

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
plt.title('Coronavirus (7-day rolling mean) 14 days forecast (original scale) for {}'.format(country))
plt.legend()
plt.savefig(pltpath + 'outfcastorig.png')
#plt.show()
plt.close()

print('************* Plotted 14 days forecast in original scale')


# Print forecasted values:

forecast14S_l = ["%.1f" % elem for elem in forecast14S]
print('Next 14 days forecast values: ', forecast14S_l)

runtime = time.gmtime(time.time() - start_time)
res = time.strftime('%M:%S', runtime)
print("Took %s mins/secs to create the %s model" % (res, country))

