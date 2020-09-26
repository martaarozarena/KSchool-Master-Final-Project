# Project memory

This project predicts coronavirus cases and deaths in 25 selected countries around the world, for the next 2 weeks. In order to aim for better predictions, the model is trained with various exogenous variables. In the frontend, the user is then allowed to modify a couple of these exogenous variables in the future and see how those changes impact the forecast. The frontend visualisation tool is also deployed in Google Cloud, where daily scripts are run in order to retrieve the latest data and update the models with last observed date.

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
* [Eurocontrol airport traffic dataset](https://ansperformance.eu/data/): from here we can download an excel from where we extract the number of daily arrivals by country (only for European countries). This data is then integrated into the `exogenous` dataset.


## Methodology
### Front end
In order to deploy the front end we analyzed several options:
  1. Jupyter-Hub it is a user-friendly web app but it has 2 big problems, we need to give access to people to see the front end and second we need the computer running to make it work non-stop.
  2. Heroku. An easy deployment for streamlit apps, with very few steps and working as a github (cloning repositories with the app) it is possible to run the front end. The main problem with this was the inability to schedule several scripts along the day, at least without paying an extra plugging. This platform was only showing streamlit but neither updating the endogenous and exogenous variables nor updating the models.
  3. Google cloud app engine. This deployment was much more complicated than heroku one, it requires to create containers for Google cloud app engine using Dockers moreover it had the same problem than Heroku; Running 3 scripts independently at different time schedules was complex and besides it was needed to create a google cloud compute engine VM to run the app
  4. Google cloud compute engine: with only a virtual machine it was possible to deploy the streamlit and make it public. This option was the easiest one and the cheapest since we had the google cloud trial period. The make it public we had 2 options, Ngrok library to create a URL or opening the ports and give access to the machine, we decided to open ports. 

The selected option was then the fourth one, we created a virtual machine and though the ssh connection system we transferred all the files from github to the machine. Once the files were in there, we had to open the ports. by default google creates a firewall to protect the machine from external connections therefore we had to make a firewall rule to allow everyone accessing streamlit. Opened ports were from 7000-9000 to make sure they were all opened in case streamlit uses other ports and not only the default one.
Next step was to create a static IP so we could access the app with the same IP always and finally the public URL was 34.78.90.249:8501

To prepare the VM to run streamlit we had to schedule 2 different actions. first one was to run everyday at 3am the endog_exog scrip to update the data and second was to run the model_act scrip at 3.30 am to update the models. we used crontab to do this actions and the result was the following:
![crontab image]( https://drive.google.com/uc?export=download&id=1uWb_thqh2qK5wOg1a-zxHRXKxJzzgpvi)

As it is possible to see in the pic, it has also been added after the running of the file to copy all the outputs in another file called mycmd so in case something happens we can trace the error (command to see the file: grep 'mycmd' /var/log/syslog). Also all the scripts need to have at the begginning #!/usr/bin/env python3 to tell crontab it is a python3 file so it can run it

Finally the only thing missing to run the files is to give them access to execute and write whatever they need. this is done in the terminal with the following command chmod 777 file.txt

Everything is ready to deploy streamlit with the following line nohup streamlit run streamcovapp.py. nohup is needed as it tells the machine not to stop the streamlit when we close the terminal.

## Summary of main results

## Conclusions

* aquí va mi prueba


## User manual front end
