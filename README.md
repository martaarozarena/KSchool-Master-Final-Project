# Coronavirus forecast in different countries

**Initial note**: This `readme` explains how to run this project. It contains special instructions since the purpose is that it can be locally run and also evaluated by the instructors.

This project predicts coronavirus cases and deaths in 25 selected countries around the world, for the next 2 weeks. In order to aim for better predictions, the model is trained with various exogenous variables. In the frontend, the user is then allowed to modify a couple of these exogenous variables in the future and see how those changes impact the forecast.

This tool is not intended for professional use - it is the result of a Data Science Master's final project and hence includes all necessary steps:

* **Data acquisition**: data is downloaded from various open data sources (cited at the end of this `readme`) and saved in the two main csvs used for this project - `endogenous.csv` and `exogenous.csv`
  * OurWorldInData.org (https://ourworldindata.org/coronavirus): daily numbers of coronavirus cases and deaths by country form the `endogenous` dataset. The daily positive rate by country is part of the `exogenous` dataset.
  * Oxford Covid-19 Government Response Tracker (OxCGRT) (https://www.bsg.ox.ac.uk/research/research-projects/coronavirus-government-response-tracker): collects systematic information on which governments have taken which measures, and when. 11 of these country level policies have been integrated into the `exogenous` dataset.
  * YouGov Covid 19 Behaviour Tracker (https://yougov.co.uk/topics/international/articles-reports/2020/03/17/personal-measures-taken-avoid-covid-19): percentage of people who say they are wearing a face mask when in public places in each country. This data is included in the `exogenous` dataset, at country level.
  * Eurocontrol airport traffic dataset (https://ansperformance.eu/covid/): the number of daily arrivals by country (only for European countries) is included in the `exogenous` dataset.
* **Data cleaning**: using Python `pandas`
* **Data exploration**: again with `pandas`
* **Modeling**: using Python `SARIMAX` to train one model per country and variable (cases/deaths)
* **Visualisation**: with `Streamlit`
* **Deployment**: using `Google Cloud` compute engine, which also runs daily scripts to update data and models with last observed date. Public url: [Live demo](http://34.78.90.249:8501/)


## Libraries to install
Please create a new `Conda` environment with the required libraries. Since the project has been done in Windows and Linux simultaneously, there are 2 requirements files depending on the system. 
* If you are in `Windows`, please type:
```bash
conda create --name <envname> --file <...>
```
* If you are in `Linux`, please type:
```bash
conda create --name <envname> --file <...>
```

## Folder creation
After cloning this repository in the desired project folder, you would have downloaded 4 folders (apart from the main notebooks and python scripts): `data`, `models`, `plots` and `old`.
The files contained in these folders are backup in case any process breaks while running main files, so you could still see the final results.

In order to start from a clean slate, please delete locally the contents of the `data`, `models` and `plots` folders since these would be populated when running the scripts. The `old` folder contains initial draft notebooks used for testing code. In order to delete locally the content of these folders, please type in command line:
```bash
del ".\data\*.*" /s /f
del ".\models\*.*" /s /f
del ".\plots\*.*" /s /f
```

## Notebooks and .py files description
For the purpose of evaluating this project, there are two main notebooks where the core work can be seen (including graphs). But for the purpose of running the entire project, it can all be done by running 3 commands (explained in detailed in the following section).

The two main **notebooks** are: 
1. [01_endog_exog_series.ipynb](https://github.com/martaarozarena/KSchool-Master-Final-Project/blob/master/01_endog_exog_series.ipynb): Here the four data sources used are downloaded, cleaned and processed
2. [02_country_model_pipeline.ipynb](https://github.com/martaarozarena/KSchool-Master-Final-Project/blob/master/02_country_model_pipeline.ipynb): This one contains the optimized process to create one `SARIMAX` model per country per variable (25 countries x 2 variables per country = 50 models in total). At the beginning of the notebook, in order to 'play' with different countries/models, you could change the variables `country` and `variable` to one of the options below:
 * `country`: Australia', 'Canada', 'China', 'Denmark', 'Finland', 'France', 'Germany', 'India', 'Indonesia', 'Italy', 'Japan', 'Malaysia', 'Mexico', 'Norway', 'Philippines', 'Saudi Arabia', 'Singapore', 'Spain', 'Sweden', 'Taiwan', 'Thailand', 'United Arab Emirates', 'United Kingdom', 'United States', 'Vietnam'
 * `variable`: 'new_cases_', 'new_deaths_'

The **python** files are:
1. [endog_exog_series.py](https://github.com/martaarozarena/KSchool-Master-Final-Project/blob/master/endog_exog_series.py): Here the four data sources used are downloaded, cleaned and saved into 2 csvs (`endogenous.csv` and `exogenous.csv`)
2. [datatools.py](https://github.com/martaarozarena/KSchool-Master-Final-Project/blob/master/datatools.py): Module created to define some functions used in the scripts/notebooks that creates the SARIMAX country models
3. [model_pipeline_one.py](https://github.com/martaarozarena/KSchool-Master-Final-Project/blob/master/Model_pipeline_one.py): Simplified version of the `02_country_model_pipeline.ipynb` notebook with just one function which takes arguments `country` and `variable`. The function:
 * Creates the SARIMAX model for that country and variable, saved in the `models` folder
 * Creates 2 plots per country/variable saved in the `plots` folder: one with the test prediction and the other one with a simulated forecast, where exogenous variables in the future don't change
 * Returns the SARIMAX (p,d,q) order, the MAE and MAE% - used later to create a results summary file
4. [model_pipeline_all.py](https://github.com/martaarozarena/KSchool-Master-Final-Project/blob/master/Model_pipeline_all.py): creates all the 50 models and summarises the results in the `results.csv` file created in the main project folder
5. [streamcovapp.py](https://github.com/martaarozarena/KSchool-Master-Final-Project/blob/master/streamcovapp.py): Streamlit python file for the front end visualisation.
6. [model_act.py](https://github.com/martaarozarena/KSchool-Master-Final-Project/blob/master/model_act.py): Script run daily in Google Cloud to update data (`endogenous.csv` and `exogenous.csv`) and add latest observations to all 50 models

## Running .py files
In order to run locally the full project, please run the following commands in this order. 

Since the second command (the one with the `model_pipeline_all.py`) gives a lengthy output, it is recommend to add ```> output.txt``` to the command to redirect all the output to a text file.
```bash
python .\endog_exog_series.py
```
```bash
python .\model_pipeline_all.py
```

## Visualising front end
Lastly, in order to locally visualize the front end please type the following command.
```bash
streamlit run .\streamcovapp.py
```
If there is any problem when running the two python commands in the previous section, the necessary files for visualising the front end are currently saved in GitHub as a back up plan. In order to run the streamlit file from GitHub back up files, please edit `streamcovapp.py` and look for `url1`, `url2`, `url3` and `url4`. Below these variables, you will find the GitHub urls which need to be uncommented and hence comment the local paths accordingly.  

<br/><br/>

**Acknowledgements and disclaimer**. This work relies on the following public data sources:
* Max Roser, Hannah Ritchie, Esteban Ortiz-Ospina and Joe Hasell (2020) - "Coronavirus Pandemic (COVID-19)". Published online at OurWorldInData.org. Retrieved from: 'https://ourworldindata.org/coronavirus' [Online Resource]
* Thomas Hale, Sam Webster, Anna Petherick, Toby Phillips, and Beatriz Kira. (2020). Oxford COVID-19 Government Response Tracker. Blavatnik School of Government.
* Jones, Sarah P., Imperial College London Big Data Analytical Unit and YouGov Plc. 2020, Imperial College London YouGov Covid Data Hub, v1.0, YouGov Plc, April 2020
* Eurocontrol data: This data is published by the EUROCONTROL Performance Review Unit in the interest of the exchange of information. It may be copied in whole or in part providing that this copyright notice © and disclaimer are included. The information may not be modified without prior written permission from the EUROCONTROL Performance Review Unit. The information does not necessarily reflect the official views or policy of EUROCONTROL, which makes no warranty, either implied or expressed, for the information contained in this document, including its accuracy, completeness or usefulness.



