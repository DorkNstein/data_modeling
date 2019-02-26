from scikit_scoring import get_regression_values
from pprint import pprint
import json
import pandas as pd
import numpy as np
from datetime import datetime
import time

ts = time.time()
df = pd.read_csv('input/drug_purchase_300K_d.csv',
                 delimiter="|", header=None, parse_dates=[19])
df.columns = ['memberid', 'memberage', 'membergender', 'pharmacyname', 'pharmcity', 'pharmacystate', 'pharmzip', 'pharmlat', 'pharmlong', 'ndc11code',
              'drug_productname', 'drug_ahfsclass', 'drug_shortname', 'drug_indication', 'calcawpunitprice', 'drug_qty', 'daysupply', 'calcawpfullprice', 'netplancost', 'filldate']
output = 'netplancost'

df[['fillmonth', 'fillyear']] = df['filldate'].apply(
    lambda x: pd.Series(x.strftime("%Y-%m,%Y").split(",")))


indication_types = ["", "INCURABLE", "REPEATABLE"]
gender_types = {
    "Not Provided": 0,
    "M": 1,
    "F": 2
}
with open('list_of_db_diseases.json') as f:
    data = json.load(f)

# display(data)

df['lasting_factor'] = df['drug_indication'].map(lambda x: data[x])
df['gender_factor'] = df['membergender'].apply(lambda x: gender_types.get(x))

# df.head()


df_2 = df[df['memberage'] != "\N"]
# df_2[['memberid', 'lasting_factor', 'drug_indication', 'gender_factor', 'memberage',
#       'calcawpfullprice', 'netplancost']].sort_values(['netplancost'], ascending=[False])

df_2['memberage'] = df_2['memberage'].astype('str').astype('int64')
df_2['gender_factor'] = df_2['gender_factor'].astype('str').astype('int64')
df_2['lasting_factor'] = df_2['lasting_factor'].astype('str').astype('int64')

df_2['pharmzip'] = df_2['pharmzip'].apply(
    lambda x: x if x.isdigit() else 0)
df_2['pharmzip'] = df_2['pharmzip'].astype('str').astype('int64')
# display(df_2.dtypes)

df_2['state'] = df_2['pharmacystate'].factorize()[0]
state_df_2 = df_2[['pharmacystate', 'state']
                   ].drop_duplicates().sort_values('state')
pharmacystate_to_state = dict(state_df_2.values)
state_to_pharmacystate = dict(state_df_2[['state', 'pharmacystate']].values)


df_3 = df_2[['netplancost', 'lasting_factor', 'memberage',
             'gender_factor', 'pharmzip', 'state']]

print(df_3.info())
get_regression_values(df_3, output)


ts_2 = time.time()
print("Total Time: %f" % (ts_2 - ts))
