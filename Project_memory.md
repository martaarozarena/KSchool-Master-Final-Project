# Project memory

This project predicts coronavirus cases and deaths in 25 selected countries around the world, for the next 2 weeks. In order to aim for better predictions, the model is trained with various exogenous variables. In the frontend, the user is then allowed to modify a couple of these exogenous variables in the future and see how those changes impact the forecast. The frontend visualisation tool is also deployed in Google Cloud, where daily scripts are run in order to retrieve the latest data and update the models with last observed date.  

The detailed instructions to replicate the full project can be found in the [README.md](https://github.com/martaarozarena/KSchool-Master-Final-Project/blob/master/README.md).

## Introduction

This project has been created during a very special time in history in which all countries around the world are being affected by the novel coronavirus SARS-CoV-2. This created a unique opportunity to do something interesting with all the different data that was being generated, on a daily basis, and hence we arrived to the idea of predicting coronavirus cases and deaths in different countries around the world. 

The number of daily coronavirus cases and deaths are easily available in different websites, so this was the 'easy' part. Since the idea was to perform time series forecasting, we thought of ARIMA as the model to use. This gave us the opportunity to add different exogenous variables to the model in order to aim for better predictions, as the coronavirus cases/deaths curves by themselves do not show any repeated patterns over time. This led us to choose Python SARIMAX model for the forecasting.

The idea was to generate an optimized way of modeling/forecasting the coronavirus cases for one country and then iterate that process to generate the rest of the models. We chose 25 countries around the world and thus we would generate 50 models in total: one for the coronavirus cases, other for the coronavirus deaths and this for each of the 25 selected countries. This is relevant since there was a need to balance between how much time was spent on generating the 'best' forecast versus the time spent in the rest of the project, that is time spent generating all the 50 models and the public front end visualisation with the results.

Once we had the idea (predicting coronavirus cases/deaths in 25 countries) and the model for forecasting (SARIMAX), the next step was to investigate what data was publicly available for the exogenous variables.

## Raw data description

The countries we selected for our project were the following: 'Australia', 'Canada', 'China', 'Denmark', 'Finland', 'France', 'Germany', 'India', 'Indonesia', 'Italy', 'Japan', 'Malaysia', 'Mexico', 'Norway', 'Philippines', 'Saudi Arabia', 'Singapore', 'Spain', 'Sweden', 'Taiwan', 'Thailand', 'United Arab Emirates', 'United Kingdom', 'United States', 'Vietnam'.

We needed to look for time series that were available both on a **daily** basis and at **country** level, since the idea was to build one model per country and per each of the variables (cases/deaths). Below are the data sources we investigated:

* [Our World in Data COVID-19 data](https://github.com/owid/covid-19-data/tree/master/public/data): from here we download, on a daily basis, the daily numbers of coronavirus cases and deaths by country, as well as the daily positive rate by country. The CSV file downloaded follows a format of 1 row per location and date. We transform the data into a time series format and create 2 cleaned csvs, in a time series format: one called `endogenous.csv` (which has the cases and deaths) and the other one called `exogenous.csv`, which contains the first set of exogenous data: `positive_rate` by country
* [Oxford Covid-19 Government Response Tracker (OxCGRT)](https://github.com/OxCGRT/covid-policy-tracker): collects systematic information on which governments have taken which measures, and when. The following 11 country level policies have been integrated into the `exogenous` dataset, again after cleaning and transforming data into time series: 'C1_School closing', 'C2_Workplace closing', 'C3_Cancel public events', 'C4_Restrictions on gatherings', 'C5_Close public transport', 'C6_Stay at home requirements', 'C7_Restrictions on internal movement', 'C8_International travel controls', 'H1_Public information campaigns', 'H2_Testing policy', 'H3_Contact tracing'
* [YouGov Covid 19 Behaviour Tracker](https://yougov.co.uk/topics/international/articles-reports/2020/03/17/personal-measures-taken-avoid-covid-19): from here we take the data on percentage of people who say they are wearing a face mask when in public places in each country. YouGov has partnered with the Institute of Global Health Innovation (IGHI) at Imperial College London to gather global insights on people’s behaviours in response to COVID-19. The research covers 29 countries, interviewing around 21,000 people each week. The datafiles contain responses from nationally representative surveys of the general public about symptoms, testing, self-isolation, social distancing and behaviour.
The data downloaded is already in time series format, and is added to the `exogenous` dataset, at country level.
* [Crowdsourced air traffic data from The OpenSky Network 2020](https://zenodo.org/record/4034518#.X29OCWgzaF4): The data in this dataset is derived and cleaned from the full OpenSky dataset to illustrate the development of air traffic during the COVID-19 pandemic. The biggest problem encountered here was that most of the flights did not include origin/destination. They did have their callsign but we didn't find a right dataset to decode the callsigns into flight numbers, to then derive their origin/destination. We discarded this option and used the below flights data from Eurocontrol
* [Eurocontrol airport traffic dataset](https://ansperformance.eu/data/): from here we can download an excel from where we extract the number of daily arrivals by country (only for European countries). This data is then integrated into the `exogenous` dataset, in time series format at country level.


## Methodology
### Data exploration and cleaning

Before starting cleaning, an exploratory analisis was done, using pandas and seaborn libraries we could look exactly what data we got. the name of the columns, the type of data and number of missing values, also we used seaborn to see the evolution of the data visually per country and all together. Part of that exploration can be found in `coviddata` file.

After exploration, next step was the cleaning and normalization part. For this, we used pandas library. The first part consisted of filter the data and take only the information of our 25 countries from 1st of january on. Then we checked there were no missing data and in case there were, find the way of filling the gaps. Also data had to be prepared for SARIMAX model so exogenous variables had to be in a Dataframe with the variables in the columns and the dates in the index.

In our case, flights data was updated once a month so from the actualization date to the prediction date there were some missing values. The first idea was to create a predictive model for the flights (there is a `flightspred.ipynb` file in the old folder from github) but after analysing the influence of this variable in the model we realized that just making a constant line with last available value, the impact on the model was not big so that was the decision taken.

For the covid data, as countries do not report data properly day by day and lot of times coronavirus cases are reported several days after the confirmed infection or even reported negative values to correct the curve, we decided to make a 7 days rolling average. After this, there were still some missing values and we applied a linear interpolation.

The 7 days rolling average was done in all the exogenous variables as well and then a linear interpolation to avoid missing values.

Once we have all the variables cleaned, by day and without NaNs, it was time to normalize them so the scale was the same for all the variables and there was not one with more impact on the model than the others due to the difference in magnitude (e.g. flights in comparison with policy variables).

The normalization was done with minmaxscaler between 0 and 1

### Modeling
Once the data is cleaned and in the format needed for the SARIMAX model, we need to take a few more steps:
1. Train/test split: the `endogenous` and `exogenous` dataframes are split into train and test. We chose 85% of the data for the train size as data starts on Jan 1st 2020 and for the majority of the countries coronavirus cases start in March, and the deaths curve starts even later (meaning the first values of the series are all zero for many countries).
2. Stationarity: one big drawback of ARIMA models is that the series to be forecasted (`endogenous`) needs to be stationary in order to get somewhat good predictions. The typical cases/deaths curve are not stationary at all, so needed to be transformed. After looking at various options, applying 1st order differencing or 2nd order differencing resulted in a stationary series. This was also confirmed by the auto-correlation function (ACF), partial auto-correlation function (PACF) plots and different statistical tests (kpss, adf, pp). For this reason, we 'fixed' the differencing term of ARIMA (d) to be 1 or 2.  
We created a function called `plot_acf_pacf()` (which can be found [here](https://github.com/martaarozarena/KSchool-Master-Final-Project/blob/master/datatools.py)) that helps to quickly visualize the ACF and PACF plots for the 1st order differencing and the 2nd order differencing. An example output for US new cases can be founde below:

![acf_pacf_ex](https://drive.google.com/file/d/1_FIsmxhSh9go3PcDC8gKWdeOGcBPoJ2N/view?usp=sharing)

3. Selecting orders p and q of ARIMA. The best way to do this was through the `auto_arima` process of `pmdarima` library. This tool performs a grid search in order to identify the most optimal parameters and its results vary widely depending on the arguments included. The best solution we found was to loop over different auto_arimas changing some of its arguments, one of them being the differencing term (d) which we set in the previous step to 1 or 2, the other being the stepwise argument. So we created a function `autoarimas()` (which can be found [here](https://github.com/martaarozarena/KSchool-Master-Final-Project/blob/master/datatools.py)) with this loop and returned a table of the 4 ARIMA models and their AIC, sorted by AIC. We selected the top 3 and moved to the next phase. 
An example of the output of the `autoarimas` function for US new cases can be found below:  

  | parameters |      aic     |
  |:----------:|:------------:|
  |  (0, 1, 5) | -1460.611573 |
  |  (4, 2, 0) | -1444.685587 |
  |  (1, 1, 1) | -1434.721364 |
  |  (1, 2, 1) | -1423.208902 |

4. Cross validation: We instantiate the 3 ARIMA models from previous step and perform cross validation on each of them. For this we used the `RollingForecastCV` function of `pmdarima` library. Another important decision here was the selection of the KPI to measure forecasting accuracy. We decided to use the `mean absolute error` MAE as it is easy to interpret and seemed to return better forecasts in our case than the root mean squared error (RMSE). So we defined a function `cross_val()` (which can be found [here](https://github.com/martaarozarena/KSchool-Master-Final-Project/blob/master/datatools.py)) that cross validated our top 3 models and returned the best order, i.e. the one with lowest average MAE.
5. Train SARIMAX model: finally we train a SARIMAX model with the best (p,d,q) order extracted from previous step. With this fitted model, we perform a predictions on train (in-sample) and test (out-of-sample) and plot them, after transforming the results back to the original scale. These plots can be found in the `plots` folder. Below is an example of new cases in-sample/out-of-sample predictions in the United States:
![US new cases in-out predictions](https://github.com/martaarozarena/KSchool-Master-Final-Project/blob/master/plots/United%20States_new_cases__inoutpredorig.png)
6. Future forecasting: with the fitted model, we then add the test observations to the model and perform a 14 day forecast into the future. For the sake of visualising results, this forecast is done maintaing the exogenous variables as is, i.e. maintaining the last observed value for the next 14 days. These plots are also saved in the `plots` folder. Below is the US 14 day forecast of new cases in the original scale:
![US new cases 14 day forecast](https://github.com/martaarozarena/KSchool-Master-Final-Project/blob/master/plots/United%20States_new_cases__outfcastorig.png)  

  In the front end, we maintain all the exogenous variables also with the last observed value for the following 14 days but we allow the user to choose the value of 2 of the exogenous variables for the next 14 days: `testing policy` and `contact tracing policy`. We then do a live forecast with those exogenous values selected by the user, leaving the others as they were, and plot the results.


### Front end
In order to deploy the front end we analyzed several options:
  1. Jupyter-Hub it is a user-friendly web app but it has 2 big problems, we need to give access to people to see the front end and second we need the computer running to make it work non-stop.
  2. Heroku. An easy deployment for streamlit apps, with very few steps and working as a github (cloning repositories with the app) it is possible to run the front end. The main problem with this was the inability to schedule several scripts along the day, at least without paying an extra plugging. This platform was only showing streamlit but neither updating the endogenous and exogenous variables nor updating the models.
  3. Google cloud app engine. This deployment was much more complicated than heroku one, it requires to create containers for Google cloud app engine using Dockers moreover it had the same problem than Heroku; Running 3 scripts independently at different time schedules was complex and besides it was needed to create a google cloud compute engine VM to run the app
  4. Google cloud compute engine: with only a virtual machine it was possible to deploy the streamlit and make it public. This option was the easiest one and the cheapest since we had the google cloud trial period. The make it public we had 2 options, Ngrok library to create a URL or opening the ports and give access to the machine, we decided to open ports. 

The selected option was then the fourth one, we created a virtual machine and though the ssh connection system we transferred all the files from github to the machine. Once the files were in there, we had to open the ports. by default google creates a firewall to protect the machine from external connections therefore we had to make a firewall rule to allow everyone accessing streamlit. Opened ports were from 7000-9000 to make sure they were all opened in case streamlit uses other ports and not only the default one.
Next step was to create a static IP so we could access the app with the same IP always and finally the public URL was (http://34.78.90.249:8501)

To prepare the VM to run streamlit we had to schedule 2 different actions. first one was to run everyday at 3am the endog_exog scrip to update the data and second was to run the model_act scrip at 3.30 am to update the models. we used crontab to do this actions and the result was the following:
![crontab image]( https://drive.google.com/uc?export=download&id=1uWb_thqh2qK5wOg1a-zxHRXKxJzzgpvi)

As it is possible to see in the pic, it has also been added after the running of the file to copy all the outputs in another file called mycmd so in case something happens we can trace the error (command to see the file: `grep 'mycmd' /var/log/syslog`). Also all the scripts need to have at the begginning `#!/usr/bin/env python3` to tell crontab it is a python3 file so it can run it

Finally the only thing missing to run the files is to give them access to execute and write whatever they need. this is done in the terminal with the following command `chmod 777 file.txt`

Everything is ready to deploy streamlit with the following line `nohup streamlit run streamcovapp.py`. nohup is needed as it tells the machine not to stop the streamlit when we close the terminal.

## Summary of main results

The project has two main objectives: 1) was to try to predict new coronavirus cases/deaths and the 2) one was to deploy in a public url the work done, avoiding the need of installing anything on your laptop.
1. Regarding the coronavirus forecasts, the results vary a lot depending on the country. In order to understand the country's results all together, in the script [Model_pipeline_all.py](https://github.com/martaarozarena/KSchool-Master-Final-Project/blob/master/Model_pipeline_all.py) we generate a csv [results.csv](https://github.com/martaarozarena/KSchool-Master-Final-Project/blob/master/results.csv) with the summarised results per country, which are: country, variable (new cases/new deaths), order of the model, MAE and MAE%. This csv helps to do a quick visual comparison of the each country's model performance.  
In general, we can say that the forecast's MAE are relatively high resulting in poor forecasts, with some exceptions like US, Mexico, India, Indonesia, Japan, Finland or Philippines. Also we can see that the new cases forecasts tend to be better than the new deaths, which makes sense since the curve of new deaths starts later in time (less data from which to learn) and the numbers are low in many of the countries.  

2. The second objective was deployed after testing many different options. Here we encountered two difficulties: we wanted the visualisation tool to be able to do live forecasts after user introduced changes in the exogenous variables. The other issue was to update on a daily basis the data in the background, so the forecasts remained always up to date. These difficulties were finally overcome, as explained in the previous section, and the app continues its daily updates in the background.  

## Conclusions

As a conclusion we could say the models work well but most of them have a high error which means there is still room for improvement. There are several thing that could be analized:
  1. the interpolation of the variables to make it more accurate
  2. the order of the models. Could happen that other q,p,d values tend to better results
  3. and finally revising the exogenous variables. Could be that adding new variables and changing some of the ones we have help in the final result.


## User manual front end
### Public front end
There has been created a public website to see the streamlit app and play with it without the need of installing anything in the computer. The website is available though ![this link](34.78.90.249:8502/)
Once in the website there are two different parts on the app:
  1. A sidebar on the left side where there are 5 values you can play with. The first option is the country you want to forecast for and the other 4 represent the exogenous values we want to use for the prediction.
  2. The main page where we can see the forecast for the next 14 days for both deaths and infections cases. This forecast comes with 2 graphs showing in blue the data until today and in yellow the predictions for both deaths and infections.
  
### Local front end
When `endog_exog_series.py`and `model_pipeline_all.py` have been run in local, to run the streamlit app it is only needed to write in the terminal `streamlit run streamcovapp.py`.
There rest is exactly the same than using the Public front end.
