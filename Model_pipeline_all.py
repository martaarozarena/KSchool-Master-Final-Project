#!/usr/bin/env python3
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import time
import model_pipeline_one


# Initialising time:
start_time = time.time()

countries = 'Denmark|Germany|Spain|Finland|Italy|Sweden|France|Norway|United Kingdom' \
            '|United States|Canada|Mexico' \
            '|Australia|Indonesia|Malaysia|Philippines|Thailand|Vietnam|China|India|Japan|Singapore|Taiwan' \
            '|Saudi Arabia|United Arab Emirates'

#countries = 'Denmark|Vietnam'

variables = ['new_cases', 'new_deaths']

results = []

for ctry in countries.split('|'):
    for var in variables:
        start_timeone = time.time()
        best_order, mae_orig, mae_orig_perc = model_pipeline_one.create_model(ctry, var)
        runtimeone = time.gmtime(time.time() - start_timeone)
        resone = time.strftime('%M:%S', runtimeone)
        print('************* Model for {} in {} created in {} mins/secs'.format(var, ctry, resone))
        resctry = [ctry, var, best_order, round(mae_orig, 1), '{:.2%}'.format(mae_orig_perc)]
        results.append(resctry)
        
summary = pd.DataFrame(results, columns=['ctry', 'endog', 'order', 'mae', 'mae_perc'])
summary.to_csv('./results.csv')

runtime = time.gmtime(time.time() - start_time)
res = time.strftime('%H:%M:%S', runtime)
print('************* All models created in {} hours/mins/secs'.format(res))
