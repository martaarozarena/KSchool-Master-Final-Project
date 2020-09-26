# Project memory

This project predicts coronavirus cases and deaths in 25 selected countries around the world, for the next 2 weeks. In order to aim for better predictions, the model is trained with various exogenous variables. In the frontend, the user is then allowed to modify a couple of these exogenous variables in the future and see how those changes impact the forecast. The frontend visualisation tool is also deployed in Google Cloud, where daily scripts are run in order to retrieve the latest data and update the models with last observed date.

## Introduction

This project has been created during a very special time in history in which all countries around the world are being affected by the novel coronavirus SARS-CoV-2. This created a unique opportunity to do something interesting with all the different data that was being generated, on a daily basis, and hence we arrived to the idea of predicting coronavirus cases and deaths in different countries around the world. 

The number of daily coronavirus cases and deaths are easily available in different websites, so this was the 'easy' part. Since the idea was to perform time series forecasting, we thought of ARIMA as the model to use. This gave us the opportunity to add different exogenous variables to the model in order to aim for better predictions, as the coronavirus cases/deaths curves by themselves do not show any repeated patterns over time. This led us to choose Python SARIMAX model for the forecasting.

The idea was to generate an optimized way of modeling/forecasting the coronavirus cases for one country and then iterate that process to generate the rest of the models. We chose 25 countries around the world and thus we would generate 50 models in total: one for the coronavirus cases, other for the coronavirus deaths and this for each of the 25 selected countries. This is relevant since there was a need to balance between how much time was spent on generating the 'best' forecast versus the time spent in the rest of the project, that is time spent generating all the 50 models and the public front end visualisation with the results.

Once we had the idea (predicting coronavirus cases/deaths in 25 countries) and the model for forecasting (SARIMAX), the next step was to investigate what data was publicly available for the exogenous variables.

## Raw data description

We needed to look for daily series f


## Methodology

## Summary of main results

## Conclusions

## User manual front end
