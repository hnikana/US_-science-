# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 16:47:19 2020

@author: gn8525
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 15:30:45 2020

@author: hamedniakan
"""
# Reading the dataset 


# Check unique residency 

# check consistency of the records 

# constructing the dataset 

import pandas as pd 
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import multiprocessing as mp
import time
from itertools import repeat
grid = True
# Census_peroid_purchase_ padding with the first row 
# because in constructing the data it goes back to last 16 period and we padded to avoid getting error in case of data construction for the oldest record

#pad = pd.DataFrame([Census_period_purchase.iloc[-1,:]]*16)   
                                                            
#Census_period_purchase = pd.concat([Census_period_purchase , pad] , ignore_index=True)  

#Census_period_purchase.to_csv('/wsu/home/gn/gn85/gn8525/In_market_timing/Census_period_purchase.csv' , index = False)
if grid :
    path = '/wsu/home/gn/gn85/gn8525/In_market_timing/'
else:
    path = ''
Demo = pd.read_csv(path + 'Demo_modified.csv' )
Census_period_purchase = pd.read_csv(path + 'Census_period_purchase_modified.csv')
Sales = pd.read_csv(path + 'Sales_modified.csv' )
Sales['SaleDate'] = Sales['SaleDate'].apply(pd.to_datetime)

CCI_quarter = pd.read_csv (path + 'CCI_quarter.csv')
Household_header = pd.read_excel(path + 'HouseHoldWithHeaders.xlsx' , header = None).iloc[0,:]
Household = pd.read_csv(path + 'WSU_HouseHolds_Student.txt' , sep = "\t" , names = Household_header  )

census_size = Household.groupby('censusTractId').size().reset_index(name = 'size')
uszips = pd.read_excel(path+'uszips.xlsx')


bin_iteration = {'BoyAgeBw0And2':4,
       'GirlAgeBw0And2':4, 'UnknownAgeBw0And2':4, 'BoyAgeBw3And5':4,
       'GirlAgeBw3And5':4, 'UnknownAgeBw3And5':4, 'BoyAgeBw6And10':8,
       'GirlAgeBw6And10':8, 'UnknownAgeBw6And10':8, 'BoyAgeBw11And15':8,
       'GirlAgeBw11And15':8, 'UnknownAgeBw11And15':8, 'BoyAgeBw16And17':2,
       'GirlAgeBw16And17':2, 'UnknownAgeBw16And17':2 ,
       'MalesAgeBw18And24' : 12, 'FemalesAgeBw18And24': 12, 'UnknownAgeBw18And24' : 12,
       'MalesAgeBw25And34' : 16, 'FemalesAgeBw25And34' : 16, 'UnknownAgeBw25And34' : 16,
       'MalesAgeBw35And44' : 20, 'FemalesAgeBw35And44': 20, 'UnknownAgeBw35And44': 20,
       'MalesAgeBw45And54':16, 'FemalesAgeBw45And54':16, 'UnknownAgeBw45And54': 16,
       'MalesAgeBw55And64': 20, 'FemalesAgeBw55And64': 20, 'UnknownAgeBw55And64': 20 ,
       'MalesAgeBw65And74': 16, 'FemalesAgeBw65And74': 16, 'UnknownAgeBw65And74': 16,
       'MalesAge75Plus' : 12, 'FemalesAge75Plus':12, 'UnknownAge75Plus':12}   


# TODO move this one to pre_.py 

income_bin = {1 : 10000 , 2 : 17500  , 3 : 25000 ,4 : 35000 , 5 : 45000 ,
              6 : 55000 ,7 : 75000 ,8 : 113000 , 9 : 150000}



sales_unique_households = Sales['HouseHoldId'].unique()


def unique_residency_record_consistency (household, Demo_):
    length = Demo_[Demo_['ID'] == household]['HomeLengthOfResidence'].unique().shape[0]
    if length  == 1:   
        df_unique = Demo_[Demo_['ID'] == household].iloc[:,23:59].astype('int64')
        first_record = df_unique.iloc[0,:]
        if not df_unique.sum().equals(df_unique.shape[0] * first_record):
            return False
        else :
            df_unique_consistent = Demo_[Demo_['ID'] == household]
            return df_unique_consistent 
    else :
        return False 


def zip_state(zipcode , uszips):
    return uszips[uszips['zip']==zipcode]['state_name'].values[0]

def initializing(a , household , current_date , j, Sales_ , uszips):
    data = pd.DataFrame(columns = [ 'ID','date' , 'quarter'] + list(bin_iteration.keys()) + ['label' , '#OfCars' , 'CarAge', 'income' ,'censusId' , 'ZipCode' , 'State'])
    purchase_dates = pd.to_datetime(Sales_[Sales_['HouseHoldId']==household]['SaleDate']).sort_values(ascending = False )
    #def age_inferral (dates , ID , current_date = '2019_9_01' )
    # Comparing length of residency with purchase date , if the purchasde date is for this household or not 
    #age = a['FirstIndividualAge']
    
    data.loc[j,['ID']] = household
    data.iloc[j,2:-7] = a.iloc[0, 23:59].astype('int64') 
    data['label'][j] = 0
    data['#OfCars'][j] = purchase_dates.shape[0]
    data['CarAge'][j] = (current_date - purchase_dates.iloc[0] ) / np.timedelta64(90,'D')
    data['censusId'][j] = Household[Household['HouseholdId']==household]['censusTractId'].values[0]
    try:
        data['income'][j] = income_bin[a['Income'].values[0]]
    except :
        data['income'][j] = np.nan
    zipcode = a['Zip5'].values[0]
    data['ZipCode'][j] = zipcode
    try:
        data['State'][j] =  zip_state(zipcode , uszips)
    except:
        data['State'][j] =  np.nan
    return data , purchase_dates 





def Oldest_Average (df):
    temp_old = 0 
    df['SumAge']  = 0
    df['OldestAge'] = 0
    #df['OldestAge'][df.shape[0]-1]  = df['CarAge'][df.shape[0]-1]
    #df['SumAge'][df.shape[0]-1]  = df['CarAge'][df.shape[0]-1]
    for i in range(df.shape[0]-2, -1, -1):
        if df['label'][i+1]==1 :
            temp_old = temp_old + df['CarAge'][i+1]
            temp_sum = df['SumAge'][i+1].astype('float')
        df.loc[i ,'OldestAge'] = temp_old + df.loc[i ,'CarAge']
        df.loc[i,'SumAge'] = df.loc[i,'#OfCars'] * df.loc[i ,'CarAge']+ temp_sum 
    
    df['SumAge']  =  df['SumAge'].astype('float')
    df['OldestAge'] = df['OldestAge'].astype('float')
    
    
    df['AveCar'] = df ['SumAge'].divide(df['#OfCars']+0.000001 )
          
    return df 

#Based on the documentation
#income_bins = [0,15000, 19000 , 29000, 39000, 49000, 74000,  99000, 124000, np.inf]
#labels_income = ['{}_{}'.format(income_bins[i],income_bins[i+1]) for i in range(len(income_bins)-1)]
#num_of_cars_bins= [0,1,2,3,4,5,np.inf]
#labels_num_of_cars = ['No_car' , '1_car', '2_car', '3_car', '4_car', 'morethan_4_car']

def columns_encoder (df):
    income_bins = [0,15000, 19000 , 29000, 39000, 49000, 74000,  99000, 124000, np.inf]
    labels_income = ['{}_{}'.format(income_bins[i],income_bins[i+1]) for i in range(len(income_bins)-1)]
    num_of_cars_bins= [0,1,2,3,4,5,np.inf]
    labels_num_of_cars = ['No_car' , '1_car', '2_car', '3_car', '4_car', 'morethan_4_car']

    df ['income_categorical'] = pd.cut(df['income'], right = False , bins=income_bins , labels=labels_income)
    df['#OFCars_categorical'] = pd.cut(df['#OfCars'], right = False , bins=num_of_cars_bins , labels=labels_num_of_cars)
    return df




    
        
#def find_index(cen_id, year, quarter , df ):
#    
#    try :
#        index = df[(df['censusTractId']== cen_id )&( df['year']== year ) &(df['quarter']== quarter )].index.to_list()[0]
#        
#    except:
#        
#        df = set_index(cen_id, year, quarter , df )
#        index = df[(df['censusTractId']== cen_id )&( df['year']== year ) &(df['quarter']== quarter )].index.to_list()[0]
#        
#    return index , df 
#
##def add_row(row,year_, quarter_, df):
#
#    
#def set_index(cen_id, year, quarter , df ) :
#    year_ = year
#    quarter_ = quarter  
#    quarter  -= 1 
#    row = df[(df['censusTractId']== cen_id )&( df['year']== year ) &(df['quarter']== quarter )]
#    while row.shape[0] == 0:
#        if quarter > 0 :
##            print('1')
#            quarter -=1
#            row = df[(df['censusTractId']== cen_id )&( df['year']== year ) &(df['quarter']== quarter )]
#            
#        else:
##            print('2')
#            year -= 1 
#            quarter  = 4 
#            row = df[(df['censusTractId']== cen_id )&( df['year']== year ) &(df['quarter']== quarter )]
##    print('3')
#    row['year'] = year_
#    row['quarter'] = quarter_
##    print(row)
#    df = df.append(row , ignore_index = True )
#    df.sort_values(by = ['censusTractId','year' , 'quarter'] , inplace= True , ascending = False)
#    df.reset_index(drop = True)
#
#    
#    return df 





def constructor(household , Census_period_purchase_, Demo_, Sales_, uszips):  
    

    current_date = datetime.strptime('2019-03-07', '%Y-%m-%d')
    
    a = unique_residency_record_consistency (household, Demo_)
    
    if isinstance(a, pd.DataFrame) & ( household in sales_unique_households):
        
        j = 0 
    
        df , purchase_dates = initializing(a , household,  current_date ,j, Sales_ , uszips) # ??????

        for date in purchase_dates:
            df['CarAge'][j] = (current_date - date ) / np.timedelta64(90,'D')
            
            while current_date >= date:
                df['date'][j] = current_date
                j += 1    
            
                df.loc[j,:] = df.iloc[j-1,:]
                df['label'][j] = 0
                temp = df.iloc[j-1,:].copy(deep = True)
                temp.loc[:] = 0
                
                if j % 4 == 0:
                    
                    if df['GirlAgeBw0And2'][j] != 0 :
                        df['GirlAgeBw0And2'][j]-=1 
                    if df['BoyAgeBw0And2'][j] != 0 :
                        df['BoyAgeBw0And2'][j]-=1 
                    if df['UnknownAgeBw0And2'][j] != 0 :
                       df['UnknownAgeBw0And2'][j] -=1
                    if  df['GirlAgeBw3And5'][j] != 0 :
                        df['GirlAgeBw3And5'][j]-=1 
                        temp['GirlAgeBw0And2'] +=1
                    if df['BoyAgeBw3And5'][j] != 0 :
                        df['BoyAgeBw3And5'][j]-=1 
                        temp['BoyAgeBw0And2'] +=1 
                    if df['UnknownAgeBw3And5'][j] != 0 :
                        df['UnknownAgeBw3And5'][j]-=1
                        temp['UnknownAgeBw0And2'] +=1
                        
                    if df['GirlAgeBw16And17'][j]!= 0 :
                        df['GirlAgeBw16And17'][j] -=1
                        temp['GirlAgeBw11And15'] +=1
                    if df['BoyAgeBw16And17'][j] != 0 :
                        df['BoyAgeBw16And17'][j]-=1
                        temp['BoyAgeBw11And15'] +=1
                    if df['UnknownAgeBw16And17'][j] != 0  :
                        df['UnknownAgeBw16And17'][j]-=1 
                        temp['UnknownAgeBw11And15'] +=1
                    df['income'][j] = df['income'][j] / 1.025 # considering the annual growth of 2.5 % 
                    
                
                if j % 8 == 0:   
                    if df['GirlAgeBw6And10'][j] != 0 :
                        df['GirlAgeBw6And10'][j]-=1 
                        temp['GirlAgeBw3And5'] +=1 
                    if df['BoyAgeBw6And10'][j] != 0 :
                        df['BoyAgeBw6And10'][j]-=1 
                        temp['BoyAgeBw3And5'] +=1 
                    if df['UnknownAgeBw6And10'][j] != 0 : 
                        df['UnknownAgeBw6And10'][j] -=1
                        temp['UnknownAgeBw3And5'] +=1
                    if df['GirlAgeBw11And15'][j] != 0 :
                        df['GirlAgeBw11And15'][j]-=1 
                        temp['GirlAgeBw6And10'] +=1
                    if df['BoyAgeBw11And15'][j] != 0 :
                        df['BoyAgeBw11And15'][j]-=1
                        temp['BoyAgeBw6And10'] +=1
                    if df['UnknownAgeBw11And15'][j] != 0 :
                        df['UnknownAgeBw11And15'][j]-=1
                        temp['UnknownAgeBw6And10'] +=1
                    
                if j % 12 == 0:
                    if df['MalesAgeBw18And24'][j] != 0:
                        df['MalesAgeBw18And24'][j] -=1
                        temp['BoyAgeBw16And17'] +=1
                    if df['FemalesAgeBw18And24'][j] != 0:
                        df['FemalesAgeBw18And24'][j] -=1
                        temp['GirlAgeBw16And17'] +=1
                    if df['UnknownAgeBw18And24'][j] != 0:
                        df['UnknownAgeBw18And24'][j] -=1
                        temp['UnknownAgeBw16And17'] +=1
                        
                    if df['MalesAge75Plus'][j] != 0:
                        df['MalesAge75Plus'][j] -=1
                        temp['MalesAgeBw65And74'] +=1
                    if df['FemalesAge75Plus'][j] != 0:
                        df['FemalesAge75Plus'][j] -=1
                        temp['FemalesAgeBw65And74'] +=1
                    if df['UnknownAge75Plus'][j] != 0:
                        df['UnknownAge75Plus'][j] -=1
                        temp['UnknownAgeBw65And74'] +=1
                        
                if j % 18 == 0:
                    if df['MalesAgeBw25And34'][j] != 0:
                        df['MalesAgeBw25And34'][j] -=1
                        temp['MalesAgeBw18And24'] +=1
                    if df['FemalesAgeBw25And34'][j] != 0:
                        df['FemalesAgeBw25And34'][j] -=1
                        temp['FemalesAgeBw18And24'] +=1
                    if df['UnknownAgeBw25And34'][j] != 0:
                        df['UnknownAgeBw25And34'][j] -=1
                        temp['UnknownAgeBw18And24'] +=1
                        
                    if df['MalesAgeBw35And44'][j] != 0:
                        df['MalesAgeBw35And44'][j] -=1
                        temp['MalesAgeBw25And34'] +=1
                    if df['FemalesAgeBw35And44'][j] != 0:
                        df['FemalesAgeBw35And44'][j] -=1
                        temp['FemalesAgeBw25And34'] +=1
                    if df['UnknownAgeBw35And44'][j] != 0:
                        df['UnknownAgeBw35And44'][j] -=1
                        temp['UnknownAgeBw25And34'] +=1
                        
                    if df['MalesAgeBw45And54'][j] != 0:
                        df['MalesAgeBw45And54'][j] -=1
                        temp['MalesAgeBw35And44'] +=1
                    if df['FemalesAgeBw45And54'][j] != 0:
                        df['FemalesAgeBw45And54'][j] -=1
                        temp['FemalesAgeBw35And44'] +=1
                    if df['UnknownAgeBw45And54'][j] != 0:
                        df['UnknownAgeBw45And54'][j] -=1
                        temp['UnknownAgeBw35And44'] +=1
                    
                    if df['MalesAgeBw55And64'][j] != 0:
                        df['MalesAgeBw55And64'][j] -=1
                        temp['MalesAgeBw45And54'] +=1
                    if df['FemalesAgeBw55And64'][j] != 0:
                        df['FemalesAgeBw55And64'][j] -=1
                        temp['FemalesAgeBw45And54'] +=1
                    if df['UnknownAgeBw55And64'][j] != 0:
                        df['UnknownAgeBw55And64'][j] -=1
                        temp['UnknownAgeBw45And54'] +=1
                    
                    if df['MalesAgeBw65And74'][j] != 0:
                        df['MalesAgeBw65And74'][j] -=1
                        temp['MalesAgeBw55And64'] +=1
                    if df['FemalesAgeBw65And74'][j] != 0:
                        df['FemalesAgeBw65And74'][j] -=1
                        temp['FemalesAgeBw55And64'] +=1
                    if df['UnknownAgeBw65And74'][j] != 0:
                        df['UnknownAgeBw65And74'][j] -=1
                        temp['UnknownAgeBw55And64'] +=1
                
                
                df.iloc[j,3:-3] =df.iloc[j,3:-1] + temp.iloc[3:-1]
                #df.iloc[j, 1:17].mask(df.iloc[j,1:17] < 0, 0 , inplace = True)
                #df.iloc[j,44:59].where(m , 0 , inplcae = True)   
                current_date += relativedelta(months= -3)
                
                if df['CarAge'][j] -1 > 0 :
                    df['CarAge'][j] -= 1
                else:
                     df['CarAge'][j] = 0
                     
                    
                
                
                
            df['label'][j] = 1  
            df['#OfCars'][j] -=1
            current_date = date
        df['date'][j] = date    ##???update the function with grid 
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['CarAge'] = df['CarAge'].astype('float')
        df['#OfCars']  = df['#OfCars'].astype('int')
        
        


        columns = ['p_'+str(q) for q in range(16,0,-1)]
        columns.append('total_car_census')
        df = pd.merge(df , CCI_quarter.iloc[:,1:] , on =['year' , 'quarter'] , how = 'left')
    
        cen_id  = df['censusId'][j]
        df_ = []
       
        for k in range(df.shape[0]):
            year = df['year'][k]
            quarter = df['quarter'][k]
            
            
            try :
                index = Census_period_purchase_[(Census_period_purchase_['censusTractId']== cen_id )& 
                                                ( Census_period_purchase_['year']== year ) &
                                                (Census_period_purchase_['quarter']== quarter )].index.values[0]
                
                row = list (Census_period_purchase_.iloc[index-16:index,:]['ID'])
                row.append(Census_period_purchase_['total_car_census'].iloc[index])
                # loc refers to a label of index or column however iloc refers to the position (index)
                # of a rows or columns , in case of using iloc , first our dataset might not be big enough to 
                # return any index and if it it , it not returning back the right index we want 
                df_.append(row)
            except:
                
                df_.append([])
                
            
#            try:
#                index = Census_period_purchase[(Census_period_purchase['censusTractId']== cen_id )&( Census_period_purchase['year']== year ) &(Census_period_purchase['quarter']== quarter )].index.to_list()[0]
#                row = list (Census_period_purchase.iloc[index-16:index,:]['ID'])
#                df_.append(row)
#            except :
#                print(household , 'Error-1')
#                print(year , quarter , cen_id)
#                return (household , 'E1' , year , quarter , cen_id )
#                
#                
#                break 
#      
        try:
        
            df_ = pd.DataFrame(df_ , columns = columns) 
            df_ = df_/ census_size[census_size['censusTractId'] == cen_id]['size'].values[0]
        except :
            return(household, 'E-1')
        df = pd.concat([df,df_] , axis = 1 , sort = False)
        
        df = Oldest_Average(df)
        df = columns_encoder(df)
    
        return (household , df)
    else:
        
#        print (household , 'Error-2')
        return (household , 'E-2') 



          


#test =   Census_period_purchase.copy(True)  
#collector = []
#
#data_2 = pd.DataFrame()
#start = time.time()
#for household in all_households[:300] :
#    
#    _, df = constructor(household , Census_period_purchase)
#    if isinstance(df , pd.DataFrame):
#        data_2 = pd.concat([data_2 ,df ] , ignore_index=True) 
#
#duration = time.time() - start
#


def add_mean_income(df):
    df['income'] = df['income'].astype('float')
    df_agg = df.groupby(['year', 'quarter']).agg({'income' : np.nanmedian}).reset_index().rename(
            columns= {'income' : 'cen_income'})
    income_bins = [0,15000, 19000 , 29000, 39000, 49000, 74000,  99000, 124000, np.inf]
    labels_income = ['{}_{}'.format(income_bins[i],income_bins[i+1]) for i in range(len(income_bins)-1)]
    df_agg ['income_census_categorical'] = pd.cut(df_agg['cen_income'], right = False , bins=income_bins , labels=labels_income)
    
    
    df = df.merge(df_agg , how = 'left' , on = ['year','quarter'] )
    
    return df 



#census_ = Census_period_purchase['censusTractId'].unique()[:1000]
#Demo_ = Demo[Demo['censusTractId'].isin(census_)]
#Census_period_purchase_ = Census_period_purchase[Census_period_purchase['censusTractId'].isin(census_)]
#Sales_ = Sales[Sales['censusTractId'].isin(census_)]
#sales_unique_households = Sales_['HouseHoldId'].unique()   #????


    

##############
        
#start = time.time()
#all_households = Demo_['ID'].unique() 
#error_1 = []
#data_1 = pd.DataFrame()
#i = 0 
#        
#for household in all_households:
#    result = constructor(household, Census_period_purchase_, Demo_, Sales_) 
#    if isinstance(result[1],pd.DataFrame):
#        data_1 = pd.concat([data_1 ,result[1] ] , ignore_index=True)
#    else: 
#        error_1.append(list(result))
#    i+=1
#    if i % 1000 == 0:    
#        data_1.to_csv('data.csv' , index = False )   
#        error_1 = pd.DataFrame(error_1, columns = ['household', 'Error'])
#        error_1.to_csv('errors.csv' , index = False)
#        print(i)
#        
#data_1.to_csv('data_1.csv' , index = False )   
#error_1 = pd.DataFrame(error_1, columns = ['household', 'Error'])
#error_1.to_csv('errors.csv' , index = False)    
#duration = time.time() - start            
        
        
#error_1 = []
#data_1_ccollector =[]
#all_households = Demo_['ID'].unique()
#start = time.time()
#all_households_split = np.array_split(all_households , 5 , axis = 0 ) 
#for each in all_households_split:
#    a_args = [household  for household in each]
#    pool = mp.Pool(mp.cpu_count() )
#    results = pool.map(constructor ,a_args )
#    #results = pool.map(partial(constructor ,Census_period_purchase_, Demo_ , Sales_) ,[household  for household in each])
#    pool.close()
#    
#    for j in results:
#        if isinstance(j[1],pd.DataFrame):
#            data_1_ccollector.append(j[1])
#             
#        else:
#            error_1.append(j[0])
#            
#    data_1 = pd.concat(data_1_ccollector)
#    data_1.to_csv(path+'data_1_test.csv' , index = False )   
#    error_1 = pd.DataFrame(error_1, columns = ['household'])
#    error_1.to_csv(path+'error_1_test.csv' , index = False)
    

def multi_constructor(state_name) :
   # each = list(census_session[i])
    
#    i = 0
 #   Demo_ = Demo[Demo['censusTractId'].isin(each)]
    zip_codes = uszips[uszips['state_name'] == state_name]['zip'].unique()
    Demo_ = Demo[Demo['Zip5'].isin(zip_codes)]
    Demo_.reset_index(inplace=True , drop=True)
    state_census = Demo_['censusTractId'].unique()
    Census_period_purchase_ = Census_period_purchase[Census_period_purchase['censusTractId'].isin(state_census)]
    Census_period_purchase_.reset_index(inplace=True , drop=True)
    Sales_ = Sales[Sales['censusTractId'].isin(state_census)]
    Sales_.reset_index(inplace=True , drop=True)    
   
    
    error_ = []
    df_collector= []
    j = 0
    for each_census in state_census:
         
         all_households = Demo_[Demo_['censusTractId'] ==each_census]['ID'].unique()
         for household in all_households:
            result = constructor(household,Census_period_purchase_, Demo_, Sales_ ,uszips ) 
            if isinstance(result[1],pd.DataFrame):
                df_temp = add_mean_income(result[1]) 
                df_collector.append(df_temp)
            else: 
               error_.append(list(result))
        
         df  = pd.concat(df_collector)
         
         j+=1
         if j % 50 == 0: 
            
            df.to_csv(path + 'data_state_2/data_{}.csv'.format(state_name) , index = False )   
            print(state_name)
    
    df  = pd.concat(df_collector)  
    df.dropna(axis = 'index' , inplace = True )      
    df.to_csv(path+'data_state_2/data_{}.csv'.format(state_name) , index = False )   
    error= pd.DataFrame(error_, columns = ['household', 'Error'])
    error.to_csv(path+'data_state_2/errors_{}.csv'.format(state_name), index = False)    
    print(state_name)         
 #   i +=1            
 
#census_session = np.array_split( Census_period_purchase['censusTractId'].unique() , 2000 , axis = 0 )       
STATES = uszips['state_name'].unique()      
start = time.time()  
pool = mp.Pool(31)
results = pool.map(multi_constructor ,[state for state in STATES] )
    #results = pool.map(partial(constructor ,Census_period_purchase_, Demo_ , Sales_) ,[household  for household in each])
pool.close()
    

