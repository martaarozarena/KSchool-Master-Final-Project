# Project memory

This project predicts coronavirus cases and deaths in 25 selected countries around the world, for the next 2 weeks. In order to aim for better predictions, the model is trained with various exogenous variables. In the frontend, the user is then allowed to modify a couple of these exogenous variables in the future and see how those changes impact the forecast. The frontend visualisation tool is also deployed in Google Cloud, where daily scripts are run in order to retrieve the latest data and update the models with last observed date.

## Introduction




## Raw data description

## Methodology
### Front end
In order to deploy the front end we analyzed several options:
1. Heroku. An easy deployment for streamlit apps, with very few steps and working as a github (cloning repositories with the app) it is possible to run the front end. The main problem with this was the inability to schedule several scripts along the day, at least without paying an extra plugging. This platform was only showing streamlit but neither updating the endogenous and exogenous variables nor updating the models.
2. Google cloud app engine. This deployment was much more complicated than heroku one, it requires to create containers for Google cloud app engine using Dockers moreover it had the same problem than Heroku; Running 3 scripts independently at different time schedules was complex and besides it was needed to create a google cloud compute engine VM to run the app
3. Google cloud compute engine: with only a virtual machine and opening the ports it was possible to deploy the streamlit and make it public. This option was the easiest one and the cheapest since we had the google cloud trial period.

The selected option was then the third one, we created a virtual machine and though the ssh connection system we transferred all the files from github to the machine. Once the files were in there, we had to open the ports. by default google creates a firewall to protect the machine from external connections therefore we had to make a firewall rule to allow everyone accessing streamlit. Opened ports were from 7000-9000 to make sure they were all opened in case streamlit uses other ports and not only the default one.
Next step was to create a static IP so we could access the app with the same IP always and finally the public URL was 34.78.90.249:8501

To prepare the VM to run streamlit we had to schedule 2 different actions. first one was to run everyday at 3am the endog_exog scrip to update the data and second was to run the model_act scrip at 3.30 am to update the models. we used crontab to do this actions and the result was the following:
![crontab image](https://drive.google.com/file/d/1uWb_thqh2qK5wOg1a-zxHRXKxJzzgpvi/view?usp=sharing)

As it is possible to see in the pic, it has also been added after the running of the file to copy all the outputs in another file called mycmd so in case something happens we can trace the error (command to see the file: grep 'mycmd' /var/log/syslog). Also all the scripts need to have at the begginning #!/usr/bin/env python3 to tell crontab it is a python3 file so it can run it

Finally the only thing missing to run the files is to give them access to execute and write whatever they need. this is done in the terminal with the following command chmod 777 file2.txt

Everything is ready to deploy streamlit with the following line nohup streamlit run streamcovapp.py. nohup is needed as it tells the machine not to stop the streamlit when we close the terminal.

## Summary of main results

## Conclusions

## User manual front end
