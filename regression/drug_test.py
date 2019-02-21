from scikit_scoring import get_regression_values
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dateutil
import time
from sklearn.preprocessing import LabelEncoder
import sys
sys.path.append('../')

ts = time.time()

data = pd.read_csv('../input/drug_purchases.csv',
                   parse_dates=['purchase_date'])
output = 'unit_price'

# Convert date from string to date times
# data[['purchase_month']] = data['purchase_date'].apply(
#     lambda x: pd.Series(x.strftime("%Y-%m")))
# data[['purchase_year']] = data['purchase_date'].apply(
#     lambda x: pd.Series(x.strftime("%Y")))


data = data.drop(columns=['id', 'log_id', 'layout_id', 'staging_record_id', 'created_by', 'created_at', 'modified_by', 'modified_at',
                          'deleted', 'address', 'city', 'ndc11code', 'person_code', 'pharmacy', 'zip', 'purchase_date', 'lattitude', 'longitude'], axis=1)

ts_2 = time.time()
print(ts_2 - ts)
print(data.shape)
print(data.head())


data["drug_class"] = LabelEncoder().fit_transform(data["drug_class"])
data["drug_indication"] = LabelEncoder(
).fit_transform(data["drug_indication"])
data["state"] = LabelEncoder().fit_transform(data["state"])
data[["drug_class", "drug_indication", "state"]].head(11)

print(data.dtypes)


# # """ Data analysis """

get_regression_values(data, output)
