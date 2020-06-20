#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 18:10:22 2020

@author: hamedniakan
"""

import pandas as pd
import numpy as np 
import os

def add_mean_income(df):
    df['income'] = df['income'].astype('float')
    df_agg = df.groupby(['year', 'quarter']).agg({'income' : np.nanmedian}).reset_index().rename(
            columns= {'income' : 'cen_income'})
    income_bins = [0,15000, 19000 , 29000, 39000, 49000, 74000,  99000, 124000, np.inf]
    labels_income = ['{}_{}'.format(income_bins[i],income_bins[i+1]) for i in range(len(income_bins)-1)]
    df_agg ['income_census_categorical'] = pd.cut(df_agg['cen_income'], right = False , bins=income_bins , labels=labels_income)
    
    
    df = df.merge(df_agg , how = 'left' , on = ['year','quarter'] )
    
    return df 

grid = True 
if grid :
    path = '/wsu/home/gn/gn85/gn8525/In_market_timing/chunked_preprocess/'
else :
    path = ''
collector = []    
for i in range(2000) :
    df = pd.read_csv(path+'chuncked_data_2/data_{}.csv'.format(i))
    df = add_mean_income(df)
    df.dropna(axis = 'index' , inplace = True )
    collector.append(df)
    

data_concat =  pd.concat(collector)

data_concat.to_pickle(path+'data_concat_2.pkl')
data_concat.to_csv(path+'data_concat_2.csv')
    
    
collector = []    
for file in os.listdir('data_grid'):
    df = pd.read_csv('data_grid/{}'.format(file))
    df = add_mean_income(df)
    df.dropna(axis = 'index' , inplace = True )
    collector.append(df)
    

data_concat =  pd.concat(collector)

data_concat.to_pickle(path+'data_concat_sample.pkl')
data_concat.to_csv(path+'data_concat_sample.csv', index = False)
        


