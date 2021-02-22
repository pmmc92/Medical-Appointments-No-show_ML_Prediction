# Medical-Appointments-No-show_ML_Prediction

![Image_appointment](https://www.everseat.com/wp-content/uploads/2015/10/appointment_1500x1000-800x400.png)

# Table of contents
* [Introduction](#Introduction)
* [Objective](#Objective)
* [Environment](#Environment)
* [Dataset Used](#Dataset-Used)
* [Main Findings of Data Analysis](#Main-Findings-of-Data-Analysis)  
  * [Age and Gender](#Age-and-Gender)  
  * [Missing Consultation](#Missing-Consultation)
  * [Variables and No-shows](#Variables-and-No-shows)
  * [Location](#Location)


## Introduction
A recent article on Forbes magazine highlights the problem of missing medical appointments. One study found that medical no-shows cost the United  States healthcare system more than $150 billion a year and individual physicians an average of $200 per unused time slot. Apart from the economic issue, there is also consider the individual implications of no-shows on health. When patients miss appointments, continuity of care is interrupted. Medication efficacy can’t be monitored regularly. Preventive services and screenings can’t be delivered in a timely manner. Acute illnesses are more likely to go untreated and become chronic conditions with complications. In short, missing an appointment can be severely detrimental to one’s health. At the European level there are some small unicentric studies that signal the same problem and an equivalent economic cost. 

## Objective
To create a Machine Learning algorithm to predict medical appointment No-shows, thus, reducing health and economic impact of this phenomenon

## Environment
This work was done using python 3.8.x and VSCode with Jupyter extension. 

The list of packages was the following:

```
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as py 
import missingno as msgn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
``` 

## Dataset used
To study this and create the model I used the ["Medical Appointment No-shows"](https://www.kaggle.com/joniarroba/noshowappointments) freely available at Kaggle.com. 
This is a dataset with a registry of  110.527 medical appointments and 14 associated variables (characteristics)

There is no missing data on the dataset
![Missing Data](/graphs/missing_data.png)


## Main Findings of Data Analysis
### Age and Gender

![Age and Gender](/graphs/age_gender.png)

* Age distribution by gender is fairly similar
* The majority of people are active adults but there are also a large number of babies and small children

### Missing Consultation

![Missed Consultation](/graphs/missing_consultations.png)

* There is a significant percentage of missed appointments

### Variables and No-shows

![Variables](/graphs/variables.png)

* Receiving an SMS did not contribute to increase the atending (the contrary seems more likely!)
* The other variables seem to present little variance between the shows and no-shows

### Location

![Location](/graphs/neighbourhood.png)

* City location is an important factor to take in account!

### Correlation

![Correlation](/graphs/correlation.png)

* No-show presents little correlation with the other variables
* As expected Hipertension and Diabetes show high correlation with Age

## Applied Machine Learning

After data engineering and data preparation, I compared various machine learning models and **Catboost** was the best model.

Parameter | Catboost
------------ | -------------
Accuracy | 0.80
Recall | 0.55
ROC AUC Score | 0.53


![Confusion Matrix](/graphs/confusion_matrix.png)

This values doesn't seem spectacular but are in line with other ML applications using this dataset. Also, there is big inbalance on no-show observations on the data that can make harder to fit the model.

### Feature Importance

![Features](/graphs/features.png)

*  The time gap between the appointment day and the scheduling day is very important
*  Age is also important

