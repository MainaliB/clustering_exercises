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

def check_outliers(df, cols_to_check):
    '''takes in a dataframe, list of columns to check outliers for
    and returns a df with the outliers removed'''
    
    for col in cols_to_check:
        iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
        upper_bound = df[col].quantile(0.75) +  1.5 * iqr
        lower_bound = df[col].quantile(0.25) - 1.5 * iqr
        df = df[(df[col] < upper_bound) & (df[col] > lower_bound)]
    return df  



def split_zillow(df):
    '''takes in a dataframe and splits into train, test, and validate'''
    train_validate, test = train_test_split(df, test_size = .2, random_state = 123)
    train, validate = train_test_split(train_validate, test_size = .2, random_state = 123)
    
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
    


### Creating a dataframe with the name of the columns, number of missing rows in that columns, and pct of missing rows

def get_missing_col_summary(df):
    summary_df = pd.DataFrame( columns = ['Number of rows missing', "pct of rows missing"], index = [df.columns])
    summary_df['Number of rows missing'] = list(df.isnull().sum())
    summary_df['pct of rows missing'] = (summary_df['Number of rows missing']/len(df)) * 100
    return summary_df

### Write a function that takes in a dataframe and returns a dataframe with 3 columns: the number of columns missing, percent of columns missing, and number of rows with n columns missing. Run the function and document takeaways from this on how you want to handle missing values.
def get_missing_row_summary(df):

    cols_summary = pd.DataFrame(df.isnull().sum(axis = 1).value_counts())
    cols_summary = cols_summary.reset_index()
    cols_summary = cols_summary.rename(columns = {'index': 'num_cols_missing'})

    cols_summary['pct_cols_missing'] = (cols_summary.num_cols_missing/len(df.columns)) * 100

    cols_summary = cols_summary.rename(columns = {0:'num_rows'})

    return cols_summary








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



