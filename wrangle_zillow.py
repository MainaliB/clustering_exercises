import acquire
import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


df = acquire.get_data()




def get_single_unit(df):
    '''Takes in a dataframe, removes the duplicate column names and filters it based on the property land use description and returns a new
    dataframe of just single family residential property'''
    df = df.loc[:,~df.columns.duplicated()]
    df = df[df.propertylandusetypeid.isin([260, 261, 262, 279])]
    
    return df



def handle_missing_values(df, column_prop, row_prop):
    '''Takes in a dataframe, the proportion of the column with non NA, the proportion with the rows with Non NA 
    and returns dataframe with the na removed at given proportion'''
    threshold = int(round(column_prop * len(df), 0))
    df.dropna(axis = 1, thresh = threshold, inplace = True)
    threshold = int(round(row_prop * len(df.columns), 0))
    df.dropna(axis = 0, thresh = threshold, inplace = True)
    return df





def clean_zillow(df):
    '''takes in the zillow dataframe and removes redundant columns, and returns a clean version of the dataframe'''
    df.drop(columns = ['finishedsquarefeet12','propertylandusetypeid', 'calculatedbathnbr'], inplace = True)
    return df





def split_zillow(df):
    '''takes in a dataframe and splits into train, test, and validate'''
    train_validate, test = train_test_split(df, test_size = .2, random_state = 123)
    train, validate = train_test_split(train_validate, test_size = .2, random_state = 123)
    
    return train, test, validate





def fill_na(train, test, validate, cat_cols, cont_cols):
    '''takes in train, test, validate dataframe, list of categgorical variables, list of coninuous variables
    uses mode for categorical, and medican for continuous variable to fill the NA's and returns the train, test, 
    and validate dataframe'''
    for col in cat_cols:
        train[col].fillna(int(train[col].mode()), inplace = True)
        validate[col].fillna(int(train[col].mode()), inplace = True)
        test[col].fillna(int(train[col].mode()), inplace = True)
    for col in cont_cols:
        train[col].fillna(train[col].median(), inplace = True)
        validate[col].fillna(train[col].median(), inplace = True)
        test[col].fillna(train[col].median(), inplace = True)
    return train, test, validate



