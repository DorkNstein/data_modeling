import numpy as np
import pandas as pd

from scikit_scoring import get_classification_values, base_model_classification, draw_confusion_matrices

# data = pd.read_csv('./pcorr/examples/kc_house_data_1.csv')
data = pd.read_csv('../input/caravan-insurance-challenge.csv')
output = 'CARAVAN'

data.head(3)
print(data.isnull().any())
print(data.dtypes)

# """ Caravan data formatting """

# data['Date'] = pd.to_datetime(data['Date'])
# data['year'] = pd.to_datetime(data['year'])
# data = data.set_index('id')
# data.age = data.age.astype(int) 
# data.children = data.children.astype(int)

print(data.head(5))

# Drop any unneccesary columns
data = data.drop(['ORIGIN'], axis=1)

# """ """ Classification """ """
get_classification_values(data, output)
# Confusion matrix and confusion tables:
# from sklearn.metrics import confusion_matrix

# clf1, test_predictions1 = base_model_classification(X_train, X_test, Y_train, Y_test)

# class_names = np.unique(np.array(Y_test))
# confusion_matrices = [
#     #( "Support Vector Machines", confusion_matrix(y,run_cv(X,y,SVC)) ),
#     ( "Random Forest", confusion_matrix(Y_test, test_predictions1)),
#     #( "K-Nearest-Neighbors", confusion_matrix(y,run_cv(X,y,KNN)) ),
#     #( "Gradient Boosting Classifier", confusion_matrix(y,run_cv(X,y,GBC)) ),
#     #( "Logisitic Regression", confusion_matrix(y,run_cv(X,y,LR)) )
# ]
# draw_confusion_matrices(confusion_matrices,class_names)