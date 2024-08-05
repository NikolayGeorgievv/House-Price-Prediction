import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import matplotlib
import pickle

import json
df1 = pd.read_csv('../../data/Bengaluru_House_Data.csv')

# Removing some of the columns which are not required
df2 = df1.drop(['area_type', 'society', 'balcony', 'availability'], axis=1)

# Removing the rows which have null values
df3 = df2.dropna().copy()

# Making sure the values in the size column are consistent
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
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None


df3[~df3['total_sqft'].apply(is_float)]
df4 = df3.copy()
df4['total_sqft'] = df4['total_sqft'].apply(convert_sqft_to_num)

df5 = df4.copy()

df5['price_per_sqft'] = df5['price'] * 100000 / df5['total_sqft']
df5.location = df5.location.apply(lambda x: x.strip())
location_stats = df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats_less_than_10 = location_stats[location_stats <= 10]
df5.location = df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
df6 = df5[~(df5.total_sqft / df5.BHK < 300)]


# Removing the outliers in price range
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out


df7 = remove_pps_outliers(df6)


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('BHK'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices,
                                            bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')


df8 = remove_bhk_outliers(df7)
df9 = df8[df8.bath < df8.BHK + 2]
df10 = df9.drop(['size', 'price_per_sqft'], axis=1)

dummies = pd.get_dummies(df10.location)
df11 = pd.concat([df10, dummies.drop('other', axis=1)], axis=1)
df12 = df11.drop('location', axis=1)

# Splitting the data into training and testing data

X = df12.drop('price', axis=1)
y = df12.price

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

lr_clf = LinearRegression()
lr_clf.fit(X_train, y_train)
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), X, y, cv=cv)

print(lr_clf.score(x_test, y_test))


# Using GridSearchCV to find the best model
def find_best_model_using_gridsearchcv(X, y):
    algos = {
        'linear_regression': {
            'model': LinearRegression(),
            'params': {
                'fit_intercept': [True, False],
                'positive': [True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1, 2],
                'selection': ['random', 'cyclic']
            }
        }
    }

    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])


# print(find_best_model_using_gridsearchcv(X, y))


# Function to predict the price of the house
def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# Export the model to a file
# with open('banglore_home_prices_model.pickle', 'wb') as f:
#     pickle.dump(lr_clf, f)

# Export data into json file
# columns = {
#     'data_columns': [col.lower() for col in X.columns]
# }
# with open('columns.json', 'w') as f:
#     f.write(json.dumps(columns))







