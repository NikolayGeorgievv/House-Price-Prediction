import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

df1 = pd.read_csv('../../data/Bengaluru_House_Data.csv')

#Removing some of the columns which are not required
df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis=1)

#Removing the rows which have null values
df3 = df2.dropna()

#Making sure the values in the size column are consistent
df3['BHK'] = df3['size'].apply(lambda x: int(x.split(' ')[0]))

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None



df3[~df3['total_sqft'].apply(is_float)]
df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)
print(df4.head())

#