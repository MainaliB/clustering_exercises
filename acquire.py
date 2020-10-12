#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from env import host, user, password
import os


# In[2]:


# Create a function that creates a a url for accessing the database

def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'


# In[3]:


# query = sql statement to get the desired data
def get_data():
    '''This function reads in a sql query and the retrieves the data from the database using the get connection function. Converts the data in csv file and returns it'''
    
    query = '''select *
from properties_2017 p17
join 	(SELECT p_17.parcelid, p_17.logerror, p_17.transactiondate
	FROM predictions_2017 p_17
	JOIN (SELECT parcelid, Max(transactiondate) as tdate
          FROM predictions_2017
          GROUP BY parcelid )as sq1
	ON (sq1.parcelid=p_17.parcelid and sq1.tdate = p_17.transactiondate )) as tmp1
	on tmp1.parcelid = p17.parcelid
left join propertylandusetype plt on plt.propertylandusetypeid = p17.propertylandusetypeid
left join airconditioningtype ac on ac.airconditioningtypeid = p17.airconditioningtypeid
left join architecturalstyletype ach on ach.architecturalstyletypeid = p17.architecturalstyletypeid 
left join heatingorsystemtype ht on ht.heatingorsystemtypeid = p17.heatingorsystemtypeid
left join storytype st on st.storytypeid = p17.storytypeid
left join typeconstructiontype tcon on tcon.typeconstructiontypeid = p17.typeconstructiontypeid;
'''
    zillow = pd.read_sql(query, get_connection('zillow'))
    
    zillow.to_csv('zillow_csv')
    
    return zillow


# In[8]:


# creating a cache so that we wont have to run the whole get data every time
def get_zillow_data(cached = False):
    '''
    This function reads in titanic data from Codeup database if cached == False 
    or if cached == True reads in titanic df from a csv file, returns df
    '''
    if cached or os.path.isfile('zillow.csv') == False:
        zillow = get_data()
    else:
        zillow = pd.read_csv('zillow.csv', index_col=0)
    return zillow


# In[ ]:




