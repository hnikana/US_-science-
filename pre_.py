#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:18:49 2020

@author: hamedniakan
"""


import pandas as pd 
import numpy as np

#from datetime import datetime, timedelta
#from dateutil.relativedelta import relativedelta
#
import multiprocessing as mp
#import time


#
#from itertools import repeat

grid = True
if grid:
    path = '/wsu/home/gn/gn85/gn8525/In_market_timing/'
else : 
    path = ''

def add_missing_and_total_car_to_census (census_id): 
    df = Census_period_purchase[Census_period_purchase['censusTractId'] == census_id]
    df_ = pd.DataFrame()
    for year in range(df['year'].min() , int(df['year'].max())):
        for quarter in range(1,5):
            row = df[( df['year']== year ) &(df['quarter']== quarter )]
            print(row)
            if row.shape[0] == 0 :
                df_ = df_.append({'censusTractId': census_id , 'year' : year , 'quarter' : quarter , 'ID' : 0} , ignore_index=True )   
    df = df.append(df_ , ignore_index = True)
    df.sort_values(by = ['censusTractId','year' , 'quarter'] , inplace= True , ascending = True)
    df['total_car_census'] = df['ID'].rolling(window = 40 , min_periods = 0).sum()
    return df 





demo_header = pd.read_excel(path+'DemoWithHeaders.xlsx' , header = None).iloc[0,:]
Demo = pd.read_csv(path+'WSU_Demographics_Student.txt' , sep = '\t' ,names = demo_header) 
Demo.drop([0] , axis = 0 , inplace = True)
#Demo_sample = Demo[:1000]
Demo['HomeLengthOfResidence'] = Demo['HomeLengthOfResidence'].astype('int32')



Household_header = pd.read_excel(path+'HouseHoldWithHeaders.xlsx' , header = None).iloc[0,:]
Household = pd.read_csv(path+'WSU_HouseHolds_Student.txt' , sep = "\t" , names = Household_header  ) 
#Hosehold_sample = Hosehold [:1000]



Sale_header = pd.read_excel(path+'SalesWithHeaders.xlsx' , header = None).iloc[0,:]
Sales =  pd.read_csv(path+'WSU_Sales_Student.txt' ,sep = "\t" ,  names =Sale_header  , encoding='latin-1'  ) 
#Sales_sample = Sales[:1000]


CCI = pd.read_csv(path+'CCI_USA_2002.csv')
CCI['TIME'] = CCI['TIME'].apply(pd.to_datetime)
CCI.index = CCI['TIME']
CCI_quarter = pd.DataFrame()
CCI_quarter['Q_0'] = CCI['Value'].resample('QS').mean()
for i in range (1,5):
    CCI_quarter ['Q_'+str(i)] = CCI_quarter['Q_'+str(i-1)].shift(1)
    
CCI_quarter['year']  = CCI_quarter.index.year
CCI_quarter['quarter']  = CCI_quarter.index.quarter    
CCI_quarter.to_csv(path + 'CCI_quarter.csv' , index = False)

census_size = Household.groupby('censusTractId').size().reset_index(name = 'size')
census_size.to_csv(path + 'censu_size.csv') 

Sales['SaleDate'] = Sales['SaleDate'].apply(pd.to_datetime)
Sales['year'] = Sales['SaleDate'].dt.year
Sales['quarter'] = Sales['SaleDate'].dt.quarter

# to add census tract Id to the record of each household 
Sales = pd.merge(Sales , Household , left_on='HouseHoldId' , 
         right_on='HouseholdId' , how = 'left')


Demo = pd.merge(Demo , Sales[['HouseHoldId','censusTractId']] , left_on = 'ID' , right_on = 'HouseHoldId' , how = 'left')


Sales.to_csv(path + 'Sales_modified.csv' , index = False)

Census_period_purchase = Sales.groupby(['censusTractId','year', 'quarter']).count()[['ID']].reset_index() # to cpunt the number of purchase 

Demo.to_csv(path + 'Demo_modified.csv' , index = False)

pool = mp.Pool(8)
results = pool.map(add_missing_and_total_car_to_census , [census_id for census_id in Census_period_purchase['censusTractId'].unique()])
pool.close()

Census_period_purchase_modified = pd.concat(results)

Census_period_purchase_modified.to_csv(path + 'Census_period_purchase_modified.csv', index = False)
