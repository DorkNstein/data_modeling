from fbprophet import Prophet
from scikit_scoring import get_regression_values, decision_tree_regressor_regressor
from pprint import pprint
import json
import pandas as pd
import numpy as np
from datetime import datetime
import time
import matplotlib.pyplot as plt

ts = time.time()
df = pd.read_csv('input/drug_purchase_300K_d.csv',
                 delimiter="|", header=None, parse_dates=[19])
df.columns = ['memberid', 'memberage', 'membergender', 'pharmacyname', 'pharmcity', 'pharmacystate', 'pharmzip', 'pharmlat', 'pharmlong', 'ndc11code',
              'drug_productname', 'drug_ahfsclass', 'drug_shortname', 'drug_indication', 'calcawpunitprice', 'drug_qty', 'daysupply', 'calcawpfullprice', 'netplancost', 'filldate']


df[['fillmonth', 'fillyear']] = df['filldate'].apply(
    lambda x: pd.Series(x.strftime("%Y-%m,%Y").split(",")))


indication_types = ["", "INCURABLE", "SUPPLIES", "REPEATABLE", "EXTERNAL_HELP"]
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

# df_2['pharmzip'] = df_2['pharmzip'].apply(
#     lambda x: x if x.isdigit() else 0)
# df_2['pharmzip'] = df_2['pharmzip'].astype('str').astype('int64')
# display(df_2.dtypes)

df_2['state'] = df_2['pharmacystate'].factorize()[0]
state_df_2 = df_2[['pharmacystate', 'state']
                  ].drop_duplicates().sort_values('state')
pharmacystate_to_state = dict(state_df_2.values)
state_to_pharmacystate = dict(state_df_2[['state', 'pharmacystate']].values)


# df_3 = df_2[['netplancost', 'lasting_factor', 'memberage',
#              'gender_factor', 'pharmzip', 'state']]
# output = 'netplancost'

df_3 = df_2[['netplancost', 'memberage',
             'gender_factor', 'state', 'calcawpfullprice']]

df_3['plancost_factor'] = df_3['netplancost'] * df_2['lasting_factor']
output = 'plancost_factor'


### ************ REGRESSION ****************** ########

# print(df_2.groupby(['pharmacystate'])['state'])
# get_regression_values(df_3, output)

# ''''' Inputs: lasting_factor, memberage, gender_factor, pharmzip, state '''''''
y_predict, X_test, Y_test = decision_tree_regressor_regressor(
    df_3, output)
# print(predicted_y)

count = 0
# for x in np.nditer(y_predict, order='C'):
#     print(x)
#     count += 1
#     if (count == 100):
#         break

for i in range(100):
    diff = Y_test[i] - y_predict[i]
    print("diff: {3} \t {0} === {1}, \tinputs:{2}".format(
        y_predict[i], Y_test[i], X_test[i], diff if diff > 0.001 else 0))


### ************ PROPHET ****************** ########

# df_mem = df_2[df_2['memberid'] == 19962706]
# m_df = df_mem[['fillmonth', 'netplancost']]
# mod_data = m_df.copy()
# m_df_sum = pd.DataFrame({'netplancost': mod_data.groupby(
#     ["fillmonth"])['netplancost'].sum()}).reset_index()
# m_df1_sum = m_df_sum.copy()
# m_df1_sum.rename(columns={'fillmonth': 'ds',
#                           'netplancost': 'y'},
#                  inplace=True)
# print(m_df1_sum.shape)

# m = Prophet(weekly_seasonality=False, daily_seasonality=False)
# m.fit(m_df1_sum)
# m_forecast = m.make_future_dataframe(periods=180, freq='D')
# m_forecast = m.predict(m_forecast)


# fig = m.plot(m_forecast, xlabel='fillmonths',
#              ylabel='Total drug sales price')
# plt.title('Drug Market')
# plt.show()

# m.plot_components(m_forecast)
# plt.show()


##### ******************** TIME ******************** ######

ts_2 = time.time()
print("Total Time: %f" % (ts_2 - ts))
