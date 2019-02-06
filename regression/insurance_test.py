# !#/usr/local/bin/python2

import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
import time

# data = pd.read_csv('./pcorr/examples/kc_house_data_1.csv')
data = pd.read_csv('../input/Regression/insurance.csv')
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

# data = data.drop(['4046','4225','4770','Large Bags','Small Bags','XLarge Bags','Total Volume'],axis=1)
# totalus_data = data.loc[(data['region']) == 'TotalUS']  
# totalus_data.head(5)
correlation = data.corr(method = 'pearson')
columns = correlation.nlargest(10, 'charges').index
print(columns)
columns
correlation_map = np.corrcoef(data[columns].values.T)
correlation_map
print(correlation_map)

X = data[columns]
target = X['charges']
features = X.drop('charges', axis = 1)

xa = features.describe()
ya = target.describe()
print(xa)
print(ya)

Y = target.values
X = features.values


# # """ Data analysis """

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42)

from sklearn.linear_model import LinearRegression, Ridge, LassoCV, Lasso, LassoLarsCV, ElasticNet, BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostClassifier, RandomForestRegressor

pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()), ('LR', LinearRegression())])))
pipelines.append(('ScaledRIDGE', Pipeline([('Scaler', StandardScaler()), ('RIDGE', Ridge())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()), ('LASSO', Lasso())])))
pipelines.append(('ScaledLASSOCV', Pipeline([('Scaler', StandardScaler()), ('LASSOCV', LassoCV())])))
pipelines.append(('ScaledLASSOLarsCV', Pipeline([('Scaler', StandardScaler()), ('LASSOLarsCV', LassoLarsCV())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()), ('EN', ElasticNet())])))
pipelines.append(('ScaledBAYESIAN', Pipeline([('Scaler', StandardScaler()), ('BAYESIAN', BayesianRidge())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeRegressor())])))
pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()), ('GBM', GradientBoostingRegressor())])))
# pipelines.append(('ScaledADA', Pipeline([('Scaler', StandardScaler()), ('ADABOOST', AdaBoostClassifier(base_estimator=RandomForestRegressor(), random_state=0, n_estimators=100))])))

results = []
names = []
for name, model in pipelines:
    ts = time.time()
    kfold = KFold(n_splits = 10, random_state = 21)
    cv_results = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = 'neg_mean_squared_error')
    cv_results_abs = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = 'neg_mean_absolute_error')
    # cv_results_sq_log = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = 'neg_mean_squared_log_error')
    cv_results_median_abs = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = 'neg_median_absolute_error')
    cv_r2 = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = 'r2')
    cv_explained_variance = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = 'explained_variance')
    ts_2 = time.time()
    results.append(cv_results)
    names.append(name)
    msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
    msg_abs = "%f (%f)" % (cv_results_abs.mean(), cv_results_abs.std())
    # # msg_sq_log = "%f (%f)" % (cv_results_sq_log.mean(), cv_results_sq_log.std())
    msg_median_abs = "%f (%f)" % (cv_results_median_abs.mean(), cv_results_median_abs.std())
    msg_r2 = "%f (%f)" % (cv_r2.mean(), cv_r2.std())
    msg_explained_variance = "%f (%f)" % (cv_explained_variance.mean(), cv_explained_variance.std())
    print(name)
    print(msg_explained_variance)
    print(msg_abs)
    print(msg)
    # print(msg_sq_log)
    print(msg_median_abs)
    print(msg_r2)
    print("%f" % (ts_2 - ts))
    print('\n')





# from sklearn.model_selection import GridSearchCV

# scaler = StandardScaler().fit(X_train)
# rescaledX = scaler.transform(X_train)
# param_grid = dict(n_estimators = np.array([50, 100, 200, 300, 400]))
# model = GradientBoostingRegressor(random_state = 21)
# kfold = KFold(n_splits = 10, random_state = 21)
# grid = GridSearchCV(estimator = model, param_grid = param_grid, scoring = 'neg_mean_squared_error', cv = kfold)
# grid_result = grid.fit(rescaledX, Y_train)

# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#   print("%f (%f) with: %r" % (mean, stdev, param))

# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# from sklearn.metrics import mean_squared_error

# scaler = StandardScaler().fit(X_train)
# rescaled_X_train = scaler.transform(X_train)
# model = GradientBoostingRegressor(random_state=21, n_estimators=400)
# model.fit(rescaled_X_train, Y_train)

# # transform the validation dataset
# rescaled_X_test = scaler.transform(X_test)
# predictions = model.predict(rescaled_X_test)
# print (mean_squared_error(Y_test, predictions))

# compare = pd.DataFrame({'Prediction': predictions, 'Test Data' : Y_test})
# compare.head(10)

# actual_y_test = np.exp(Y_test)
# actual_predicted = np.exp(predictions)
# diff = abs(actual_y_test - actual_predicted)

# compare_actual = pd.DataFrame({'Test Data': actual_y_test, 'Predicted Price' : actual_predicted, 'Difference' : diff})
# compare_actual = compare_actual.astype(int)
# compare_actual.head(10)