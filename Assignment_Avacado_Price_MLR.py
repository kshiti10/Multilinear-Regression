# -*- coding: utf-8 -*-
"""
Created 10 Mar 23:03:12 2024

@author: Kshitija
"""

#Problem Statement
'''
With the growing consumption of avocados in the USA, 
a freelance company would like to do some analysis on the patterns 
of consumption in different cities and would like to come up with 
a prediction model for the price of avocados. 
For this to be implemented, build a prediction model using multilinear 
regression and provide your insights on it.

'''


# Dataset
#AveragePrice: The average price of a single avocado
#Total Volume: Total number of avocados sold
#Total Bags
#Small Bags
#Large Bags
#XLarge Bags
#type: conventional or organic
#year: The year
#region: The city or region of the observation


import pandas as pd
import numpy as np
import seaborn as sns
ava=pd.read_csv("c:/23-Multi-linear_Regression/Avacado_Price.csv")
ava_new=ava.iloc[:,0:11]
ava_new.describe()
ava_new.head()

#Drop Column
ava_new=ava_new.rename(columns={'XLarge Bags':'XLarge_Bags'})
ava_new.isna().sum()
#There are no null values
import matplotlib.pyplot as plt
plt.bar(height=ava_new.AveragePrice,x=np.arange(1,18250,1))
sns.distplot(ava_new.AveragePrice)
#Data is normal slight right skewed
plt.boxplot(ava_new.AveragePrice)
# There are several outliers
plt.bar(height=ava_new.Total_Volume,x=np.arange(1,18250,1))
sns.distplot(ava_new.Total_Volume)
#Data is normal but right skewed
plt.boxplot(ava_new.Total_Volume)
#There are several outliers
#let us check tot_ava1
plt.bar(height=ava_new.tot_ava1,x=np.arange(1,18250,1))
sns.distplot(ava_new.tot_ava1)
#Data is normal but slight right skewed
plt.boxplot(ava_new.tot_ava1)
#There are several outliers
# let us check tot_ava2
plt.bar(height=ava_new.tot_ava2,x=np.arange(1,18250,1))
sns.distplot(ava_new.tot_ava2)
#Data is normal but slight right skewed
plt.boxplot(ava_new.tot_ava2)
#There are several outliers
# let us check tot_ava3
plt.bar(height=ava_new.tot_ava3,x=np.arange(1,18250,1))
sns.distplot(ava_new.tot_ava3)
#Data is normal but slight rt skewed
plt.boxplot(ava_new.tot_ava3)
#There are several outliers
#let us check Total_Bags
plt.bar(height=ava_new.Total_Bags,x=np.arange(1,18250,1))
sns.distplot(ava_new.Total_Bags)
#Data is normal but slight rt skewed
plt.boxplot(ava_new.Total_Bags)
#There are several outliers

#let us check Small_Bags
plt.bar(height=ava_new.Small_Bags,x=np.arange(1,18250,1))
sns.distplot(ava_new.Small_Bags)
#Data is normal but slight rt skewed
plt.boxplot(ava_new.Small_Bags)
#There are several outliers
#let us check Large_Bags
plt.bar(height=ava_new.Large_Bags,x=np.arange(1,18250,1))
sns.distplot(ava_new.Large_Bags)
#Data is normal but slight rt skewed
plt.boxplot(ava_new.Large_Bags)
#There are several outliers

######################################################
###Data preprocessing

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
ava_new["year"]=lb.fit_transform(ava["year"])
ava_new["type"]=lb.fit_transform(ava["type"])
ava_new.dtypes
from feature_engine.outliers import Winsorizer
import seaborn as sns
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['AveragePrice'])
ava_t=winsor.fit_transform(ava_new[['AveragePrice']])
sns.boxplot(ava_t.AveragePrice)
ava_new['AveragePrice']=ava_t['AveragePrice']
plt.boxplot(ava_new.AveragePrice)
# let us check Total_Volume
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Total_Volume'])
ava_t=winsor.fit_transform(ava_new[['Total_Volume']])
sns.boxplot(ava_t.Total_Volume)
ava_new['Total_Volume']=ava_t['Total_Volume']
plt.boxplot(ava_new.Total_Volume)

winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['tot_ava1'])
ava_t=winsor.fit_transform(ava_new[['tot_ava1']])
sns.boxplot(ava_t.tot_ava1)
ava_new['tot_ava1']=ava_t['tot_ava1']
plt.boxplot(ava_new.tot_ava1)
