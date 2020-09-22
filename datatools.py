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
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def plot_acf_pacf(timeseries):
    plt.rcParams.update({'figure.figsize':(18,10), 'figure.subplot.hspace':0.5,
                     'xtick.labelsize':'x-small', 'ytick.labelsize':'x-small', 'axes.titlesize':'small'})
    fig, axes = plt.subplots(3, 3)

    # Original Series
    axes[0, 0].plot(timeseries); axes[0, 0].set_title('Original Series')
    sm.graphics.tsa.plot_acf(timeseries, lags=40, ax=axes[0, 1])
    sm.graphics.tsa.plot_pacf(timeseries, lags=40, ax=axes[0, 2])

    # 1st Differencing
    axes[1, 0].plot(timeseries.diff()); axes[1, 0].set_title('1st Order Differencing')
    sm.graphics.tsa.plot_acf(timeseries.diff().dropna(), lags=40, ax=axes[1, 1])
    sm.graphics.tsa.plot_pacf(timeseries.diff().dropna(), lags=40, ax=axes[1, 2])

    # 2nd Differencing
    axes[2, 0].plot(timeseries.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
    sm.graphics.tsa.plot_acf(timeseries.diff().diff().dropna(), lags=40, ax=axes[2, 1])
    sm.graphics.tsa.plot_pacf(timeseries.diff().diff().dropna(), lags=40, ax=axes[2, 2])
    plt.show()
    
    

def test_stationarity(timeseries):
    
    # Determining rolling statistics
    rolling_mean = timeseries.rolling(window=3).mean()
    rolling_std = timeseries.rolling(window=3).std()

    # Plot rolling statistics
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)

    print('Result of Dicky-Fuller Test')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', '#Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    
    return dfoutput


def autoarimas(endog_series, exog_series):
    results = []
    best_aic = float('inf')
    for d in range (1,3):
        for step in [True, False]:
            try: 
                modelauto = auto_arima(endog_series, exogenous=exog_series, start_p=1, start_q=1, max_p=10, max_q=10, 
                                   d=d, error_action='ignore', suppress_warnings=True, stepwise=step)
            except: 
                continue
            
            aic = modelauto.aic()
            
            if aic < best_aic:
                best_aic = aic
                best_param = modelauto.order
            
            results.append([modelauto.order, modelauto.aic()])
                
    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    
    return result_table


def cross_val(endog_series, exog_series, model1, model2, model3):
    cv = model_selection.RollingForecastCV(step=5, h=14, initial=160)
    model1_cv_scores = model_selection.cross_val_score(model1, y=endog_series, exogenous=exog_series, scoring='mean_absolute_error', cv=cv)
    model2_cv_scores = model_selection.cross_val_score(model2, y=endog_series, exogenous=exog_series, scoring='mean_absolute_error', cv=cv)
    model3_cv_scores = model_selection.cross_val_score(model3, y=endog_series, exogenous=exog_series, scoring='mean_absolute_error', cv=cv)
    
    # Filter the nan scores
    model1_cv_scores = model1_cv_scores[~(np.isnan(model1_cv_scores))]
    model2_cv_scores = model2_cv_scores[~(np.isnan(model2_cv_scores))]
    model3_cv_scores = model3_cv_scores[~(np.isnan(model3_cv_scores))]
    
    # Print score list for each model
    model1_cv_scoreslist = ["%.4f" % elem for elem in model1_cv_scores]
    model2_cv_scoreslist = ["%.4f" % elem for elem in model2_cv_scores]
    model3_cv_scoreslist = ["%.4f" % elem for elem in model3_cv_scores]
    print("Model 1 CV scores: {}".format(model1_cv_scoreslist))
    print("Model 2 CV scores: {}".format(model2_cv_scoreslist))
    print("Model 3 CV scores: {}".format(model3_cv_scoreslist))

    # Pick model based on which has a lower average error rate
    m1_average_error = np.average(model1_cv_scores)
    m2_average_error = np.average(model2_cv_scores)
    m3_average_error = np.average(model3_cv_scores)
    errors = [m1_average_error, m2_average_error, m3_average_error]
    models = [model1, model2, model3]

    # Print out the best model (where errors is min, discarding nan values):
    better_index = errors.index(np.nanmin(errors))
    best_order = models[better_index].order
    print("Lowest average MAE: {} (model{})".format(errors[better_index], better_index + 1))
    print("Best model order: {}".format(best_order))
    
    return best_order


def in_out_fcast_plot(model, steps, endog, endog_test, exog_train, exog_test):
    # Generate in-sample predictions (train):
    in_predictions = model.predict(exog=exog_train)
    
    # Generate forecast for test:
    one_step_forecast = model.get_forecast(steps=steps, exog=exog_test)
    
    # Extract forecast mean and assign negative values to zero (negative numbers don't make sense here):
    mean_forecast = one_step_forecast.predicted_mean
    mean_forecast = mean_forecast.clip(lower=0)
    
    # Get confidence intervals of  forecast
    confidence_intervals = one_step_forecast.conf_int()
    
    # Select lower and upper confidence limits
    lower_limits = confidence_intervals.loc[:,'lower ' + endog.name]
    upper_limits = confidence_intervals.loc[:,'upper ' + endog.name]
    
    #Print test mean absolute error:
    print("Test MAE (0-1 scale): %.3f" % mean_absolute_error(endog_test, mean_forecast))
    
    # plot the data
    plt.rcParams.update({'figure.figsize':(20,10), 'xtick.labelsize':'small', 'ytick.labelsize':'small', 'axes.titlesize':'large'})
    plt.plot(endog.index, endog, label='observed')
    
    # plot in-sample predictions (train)
    plt.plot(in_predictions.index, in_predictions, color='lightcoral', label= 'in-sample predictions (train)')
    
    # plot mean forecast (test)
    plt.plot(mean_forecast.index, mean_forecast, color='r', label='test predictions')
    
    # shade the area between your confidence limits
    plt.fill_between(lower_limits.index, lower_limits, upper_limits, color='pink')
    
    plt.xlabel('Date')
    plt.legend()
    
    return in_predictions, mean_forecast