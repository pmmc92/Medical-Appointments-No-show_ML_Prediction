# Medical-Appointments-No-show_ML_Prediction

# Table of contents
1. [Introduction](#Introduction)
2. [Objective](#Objective)
3. [Environment](#Environment)
4. [Dataset Used](#Dataset-Used)
5. [Main Findings of Data Analysis](#Main-Findings-of-DataAnalysis)

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
