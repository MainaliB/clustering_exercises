import acquire
import pandas as pd


import numpy as np
import pandas as pd
from env import host, user, password
import os

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def mall_data():
    query2 = '''select * 
    from customers'''
    mall = pd.read_sql(query2, get_connection('mall_customers'))
    mall.to_csv('mall.csv')
    return mall

def get_mall_data(cached = False):
    
        
    '''This function reads in mall data from Codeup database if cached == False 
    or if cached == True reads in malldf from a csv file, returns df
    '''
    if cached or os.path.isfile('mall.csv') == False:
        mall = mall_data()
    else:
        mall = pd.read_csv('mall.csv', index_col=0)
    
    return mall

def prep_mall(df):
    df = pd.concat([df, pd.get_dummies(df.gender, drop_first = True)], axis = 1)
    df.drop(columns = 'gender', inplace = True)
    
    return df


def split_data(df):
    '''takes in a dataframe and splits into train, test, and validate. Returns train, test, and validate'''
    train_validate, test = train_test_split(df, test_size = 0.15, random_state = 123)
    train, validate = train_test_split(train_validate, test_size = 0.15, random_state = 123)
    return train, test, validate

def scale_data(train, test, validate, scaler):
    '''Takes in a train, test, validate dataframe, scales is based on the type of scaled passed and returns the 
    scaled version of train, test, and validate'''

    scaler = scaler
    scaler = scaler.fit(train)
    train_scaled = pd.DataFrame(scaler.transform(train), columns = train.columns, index = train.index)

    validate_scaled = pd.DataFrame(scaler.transform(validate), columns = validate.columns, index = validate.index)


    test_scaled = pd.DataFrame(scaler.transform(test), columns = test.columns, index = test.index)
    
    return train_scaled, test_scaled, validate_scaled
    
    