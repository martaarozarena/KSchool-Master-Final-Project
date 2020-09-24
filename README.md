# Coronavirus forecast in different countries

**Initial note**: This `readme` explains how to run this project. It contains special instructions since the purpose is that it can be locally run and evaluated by the instructors.

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
After cloning this repository, you would have downloaded 4 folders (apart from the main notebooks and python scripts): `data`, `models`, `plots` and `old`.
The files contained in these folders are backup in case any process breaks while running main files, so you could still see the final results.

In order to start from a clean slate, please delete locally the contents of the `data`, `models` and `plots` folders since these would be populated when running the scripts. The `old` folder contains initial draft notebooks used for testing code. In order to delete locally the content of these folders, please type in command line:
```bash
del ".\data\*.*" /s /f
del ".\models\*.*" /s /f
del ".\plots\*.*" /s /f
```

## Notebooks


## Running .py files


## Visualising front end








**Acknowledgements and disclaimer**. This work relies on the following public data sources:
* Max Roser, Hannah Ritchie, Esteban Ortiz-Ospina and Joe Hasell (2020) - "Coronavirus Pandemic (COVID-19)". Published online at OurWorldInData.org. Retrieved from: 'https://ourworldindata.org/coronavirus' [Online Resource]
* Thomas Hale, Sam Webster, Anna Petherick, Toby Phillips, and Beatriz Kira. (2020). Oxford COVID-19 Government Response Tracker. Blavatnik School of Government.
* Jones, Sarah P., Imperial College London Big Data Analytical Unit and YouGov Plc. 2020, Imperial College London YouGov Covid Data Hub, v1.0, YouGov Plc, April 2020
* Eurocontrol data: This data is published by the EUROCONTROL Performance Review Unit in the interest of the exchange of information. It may be copied in whole or in part providing that this copyright notice © and disclaimer are included. The information may not be modified without prior written permission from the EUROCONTROL Performance Review Unit. The information does not necessarily reflect the official views or policy of EUROCONTROL, which makes no warranty, either implied or expressed, for the information contained in this document, including its accuracy, completeness or usefulness.



