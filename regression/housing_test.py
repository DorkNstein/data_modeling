# !#/usr/local/bin/python2

import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from scikit_scoring import get_regression_values

data = pd.read_csv('../input/Regression/kc_house_data_1.csv')
output = 'price'
data.head(3)
print(data.isnull().any())
print(data.dtypes)

data['date'] = pd.to_datetime(data['date'])
data = data.set_index('id')
data.price = data.price.astype(int)
data.bathrooms = data.bathrooms.astype(int)
data.floors = data.floors.astype(int)
data.head(5)

data["house_age"] = data["date"].dt.year - data['yr_built']
data['renovated'] = data['yr_renovated'].apply(lambda yr: 0
                                               if yr == 0
                                               else 1)

data = data.drop('date', axis=1)
data = data.drop('yr_renovated', axis=1)
data = data.drop('yr_built', axis=1)
data.head(5)

pd.set_option('precision', 2)

data[output] = np.log(data[output])
data['sqft_lot'] = np.log(data['sqft_lot'])

get_regression_values(data, output)
