# !#/usr/local/bin/python2

import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from scikit_scoring import get_regression_values

data = pd.read_csv('../input/Regression/avocado.csv')
output = 'AveragePrice'
data.head(3)
print(data.isnull().any())
print(data.dtypes)

# """ Avacado data formatting """

data['Date'] = pd.to_datetime(data['Date'])
data['year'] = pd.to_datetime(data['year'])

data['type_id'] = data['type'].factorize()[0]
type_id_data = data[['type', 'type_id']].drop_duplicates().sort_values('type_id')
type_to_id = dict(type_id_data.values)
id_to_type = dict(type_id_data[['type_id', 'type']].values)
data.head(5)

# # """ Data analysis """

get_regression_values(data, output)

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