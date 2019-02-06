# !#/usr/local/bin/python2

import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from scikit_scoring import get_regression_values


data = pd.read_csv('../input/Regression/insurance.csv')
output = 'charges'
data.head(3)
print(data.isnull().any())
print(data.dtypes)

# """ Insurance data formatting """

# data['Date'] = pd.to_datetime(data['Date'])
# data['year'] = pd.to_datetime(data['year'])
# data = data.set_index('id')
data.age = data.age.astype(int) 
data.children = data.children.astype(int)

data['sex_id'] = data['sex'].factorize()[0]
sex_id_data = data[['sex', 'sex_id']].drop_duplicates().sort_values('sex_id')
sex_to_id = dict(sex_id_data.values)
id_to_sex = dict(sex_id_data[['sex_id', 'sex']].values)

data['smoker_id'] = data['smoker'].factorize()[0]
smoker_id_data = data[['smoker', 'smoker_id']].drop_duplicates().sort_values('smoker_id')
smoker_to_id = dict(smoker_id_data.values)
id_to_smoker = dict(smoker_id_data[['smoker_id', 'smoker']].values)

data['region_id'] = data['region'].factorize()[0]
region_id_data = data[['region', 'region_id']].drop_duplicates().sort_values('region_id')
region_to_id = dict(region_id_data.values)
id_to_region = dict(region_id_data[['region_id', 'region']].values)

print(data.head(5))

get_regression_values(data, output)