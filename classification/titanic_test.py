#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
# import os
# try:
# 	os.chdir(os.path.join(os.getcwd(), 'input/titanic_data'))
# 	print(os.getcwd())
# except:
# 	pass

#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')
import pandas as pd


#%%
train_df = pd.read_csv('./input/titanic_data/train.csv')
test_df = pd.read_csv('./input/titanic_data/test.csv')
combine = [train_df, test_df]


#%%
print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)


#%%
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])


#%%
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


#%%
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()


#%%
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape


#%%
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()


#%%
guess_ages = np.zeros((2,3))
guess_ages


#%%
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &                                   (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()


#%%
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)


#%%
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()


#%%
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()


#%%
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


#%%
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()


#%%
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()


#%%
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)


#%%
freq_port = train_df.Embarked.dropna().mode()[0]
freq_port


#%%
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


#%%
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()


#%%
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()


#%%
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)


#%%
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)

#%% [markdown]
# ### Fit model

#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier


#%%
from sklearn.model_selection import train_test_split


#%%
X_train, X_test, y_train, y_test = train_test_split(train_df.drop("Survived", axis=1), train_df["Survived"], test_size=0.3)


#%%
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#%%
def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    '''
    print the accuracy score, classification report and confusion matrix of classifier
    '''
    if train:
        '''
        training performance
        '''
        print("Train Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))
        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))

        res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
        
    elif train==False:
        '''
        test performance
        '''
        print("Test Result:\n")        
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))
        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))    
        

#%% [markdown]
# ## Decision Tree

#%%

print("######## DecisionTreeClassifier #########")
clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)

print_score(clf, X_train, y_train, X_test, y_test, train=True)

print_score(clf, X_train, y_train, X_test, y_test, train=False) # Test


#%% [markdown]
# So our decision tree has an accuracy of 0.76
#%% [markdown]
# ![](http://)## Bagging (oob_score=False)



#%%

print("######## Bagging Classifier #########")
bag_clf = BaggingClassifier(base_estimator=clf, n_estimators=1000,
                            bootstrap=True, n_jobs=-1,
                            random_state=42)

bag_clf.fit(X_train, y_train)

print_score(bag_clf, X_train, y_train, X_test, y_test, train=True)

print_score(bag_clf, X_train, y_train, X_test, y_test, train=False)

#%% [markdown]
# The bagging model has an average accuracy of 0.767
#%% [markdown]
# ## Bagging (oob_score=True)
# 
# Use out-of-bag samples to estimate the generalization accuracy

#%%

print("######## Bagging Classifier oob_score #########")
bag_clf = BaggingClassifier(base_estimator=clf, n_estimators=1000,
                            bootstrap=True, oob_score=True,
                            n_jobs=-1, random_state=42)


#%%
bag_clf.fit(X_train, y_train)


#%%
bag_clf.oob_score_


#%%
print_score(bag_clf, X_train, y_train, X_test, y_test, train=True)


#%%
print_score(bag_clf, X_train, y_train, X_test, y_test, train=False)

#%% [markdown]
# Setting oob True also generated same score.
#%% [markdown]
# # Random Forest
# 
# [paper](http://ect.bell-labs.com/who/tkh/publications/papers/odt.pdf)
# 
# * Ensemble of Decision Trees
# 
# * Training via the bagging method (Repeated sampling with replacement)
#   * Bagging: Sample from samples
#   * RF: Sample from predictors. $m=sqrt(p)$ for classification and $m=p/3$ for regression problems.
# 
# * Utilise uncorrelated trees
# 
# Random Forest
# * Sample both observations and features of training data
# 
# Bagging
# * Samples only observations at random
# * Decision Tree select best feature when splitting a node

#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


#%%
def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    '''
    print the accuracy score, classification report and confusion matrix of classifier
    '''
    if train:
        '''
        training performance
        '''
        print("Train Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, clf.predict(X_train))))
        print("Classification Report: \n {}\n".format(classification_report(y_train, clf.predict(X_train))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, clf.predict(X_train))))

        res = cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
        
    elif train==False:
        '''
        test performance
        '''
        print("Test Result:\n")        
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, clf.predict(X_test))))
        print("Classification Report: \n {}\n".format(classification_report(y_test, clf.predict(X_test))))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, clf.predict(X_test))))    
        


#%%

print("######## RandomForestClassifier #########")

rf_clf = RandomForestClassifier(random_state=42)


#%%
rf_clf.fit(X_train, y_train)


#%%
print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)


#%%
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)

#%% [markdown]
# The accuracy came as .768 but we can tune the parameters using grid search.
#%% [markdown]
# ## Grid Search

#%%
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV


#%%
rf_clf = RandomForestClassifier(random_state=42)


#%%
params_grid = {"max_depth": [3, None],
               "min_samples_split": [2, 3, 10],
               "min_samples_leaf": [1, 3, 10],
               "bootstrap": [True, False],
               "criterion": ['gini', 'entropy']}


#%%
grid_search = GridSearchCV(rf_clf, params_grid,
                           n_jobs=-1, cv=5,
                           verbose=1, scoring='accuracy')


#%%

print("######## grid_search #########")

grid_search.fit(X_train, y_train)


#%%
grid_search.best_score_


#%%
grid_search.best_estimator_.get_params()


#%%
print_score(grid_search, X_train, y_train, X_test, y_test, train=True)


#%%
print_score(grid_search, X_train, y_train, X_test, y_test, train=False)

#%% [markdown]
# # Extra-Trees (Extremely Randomized Trees) Ensemble
# 
# [scikit-learn](http://scikit-learn.org/stable/modules/ensemble.html#bagging)
# 
# * Random Forest is build upon Decision Tree
# * Decision Tree node splitting is based on gini or entropy or some other algorithms
# * Extra-Trees make use of random thresholds for each feature unlike Decision Tree
# 

#%%
from sklearn.ensemble import ExtraTreesClassifier


#%%

print("######## ExtraTreesClassifier #########")
xt_clf = ExtraTreesClassifier(random_state=42)

xt_clf.fit(X_train, y_train)

print_score(xt_clf, X_train, y_train, X_test, y_test, train=True)

print_score(xt_clf, X_train, y_train, X_test, y_test, train=False)


#%%
Y_pred = xt_clf.predict(test_df.drop('PassengerId',axis=1))

Y_pred

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submissions_xt.csv', index=False)

#%% [markdown]
# # That is all for bagging. Now moving onto boosting
#%% [markdown]
# # Boosting (Hypothesis Boosting)
# 
# * Combine several weak learners into a strong learner. 
# 
# * Train predictors sequentially
#%% [markdown]
# # AdaBoost / Adaptive Boosting
# 
# [Robert Schapire](http://rob.schapire.net/papers/explaining-adaboost.pdf)
# 
# [Wikipedia](https://en.wikipedia.org/wiki/AdaBoost)
# 
# [Chris McCormick](http://mccormickml.com/2013/12/13/adaboost-tutorial/)
# 
# [Scikit Learn AdaBoost](http://scikit-learn.org/stable/modules/ensemble.html#adaboost)
# 
# 1995
# 
# As above for Boosting:
# * Similar to human learning, the algo learns from past mistakes by focusing more on difficult problems it did not get right in prior learning. 
# * In machine learning speak, it pays more attention to training instances that previously underfitted.
# 
# Source: Scikit-Learn:
# 
# * Fit a sequence of weak learners (i.e., models that are only slightly better than random guessing, such as small decision trees) on repeatedly modified versions of the data. 
# * The predictions from all of them are then combined through a weighted majority vote (or sum) to produce the final prediction.
# * Initially, those weights are all set to $w_i = 1/N$, so that the first step simply trains a weak learner on the original data. 
# * For each successive iteration, the sample weights are individually modified and the learning algorithm is reapplied to the reweighted data. 
# * At a given step, those training examples that were incorrectly predicted by the boosted model induced at the previous step have their weights increased, whereas the weights are decreased for those that were predicted correctly. 
# * As iterations proceed, examples that are difficult to predict receive ever-increasing influence. Each subsequent weak learner is thereby forced to concentrate on the examples that are missed by the previous ones in the sequence.
# 
# 

#%%
from sklearn.ensemble import AdaBoostClassifier

print("######## AdaBoostClassifier #########")


ada_clf = AdaBoostClassifier()

ada_clf.fit(X_train, y_train)

print_score(ada_clf, X_train, y_train, X_test, y_test, train=True)

print_score(ada_clf, X_train, y_train, X_test, y_test, train=False)


#%%
Y_pred = ada_clf.predict(test_df.drop('PassengerId',axis=1))

Y_pred

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submissions_ada.csv', index=False)

#%% [markdown]
# ## AdaBoost with Random Forest

#%%
from sklearn.ensemble import RandomForestClassifier


#%%

print("######## AdaBoostClassifier With RandomForestClassifier #########")

ada_clf = AdaBoostClassifier(RandomForestClassifier())

ada_clf.fit(X_train, y_train)

print_score(ada_clf, X_train, y_train, X_test, y_test, train=True)

print_score(ada_clf, X_train, y_train, X_test, y_test, train=False)


print("######## AdaBoostClassifier With RandomForestClassifier as base_estimator #########")


ada_clf = AdaBoostClassifier(base_estimator=RandomForestClassifier())

ada_clf.fit(X_train, y_train)

print_score(ada_clf, X_train, y_train, X_test, y_test, train=True)

print_score(ada_clf, X_train, y_train, X_test, y_test, train=False)


#%%
Y_pred = ada_clf.predict(test_df.drop('PassengerId',axis=1))

Y_pred

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submissions_ada_random.csv', index=False)

#%% [markdown]
# Works for both regression and classification
# 
# [Wikipedia](https://en.wikipedia.org/wiki/Gradient_boosting)
# 
# * Sequentially adding predictors
# * Each one correcting its predecessor
# * Fit new predictor to the residual errors
# 
# Compare this to AdaBoost: 
# * Alter instance weights at every iteration
# 
#%% [markdown]
# **Step 1. **
# 
#   $$Y = F(x) + \epsilon$$
# 
# **Step 2. **
# 
#   $$\epsilon = G(x) + \epsilon_2$$
# 
#   Substituting (2) into (1), we get:
#   
#   $$Y = F(x) + G(x) + \epsilon_2$$
#     
# **Step 3. **
# 
#   $$\epsilon_2 = H(x)  + \epsilon_3$$
# 
# Now:
#   
#   $$Y = F(x) + G(x) + H(x)  + \epsilon_3$$
#   
# Finally, by adding weighting  
#   
#   $$Y = \alpha F(x) + \beta G(x) + \gamma H(x)  + \epsilon_4$$
# 
# Gradient boosting involves three elements:
# 
# * **Loss function to be optimized**: Loss function depends on the type of problem being solved. In the case of regression problems, mean squared error is used, and in classification problems, logarithmic loss will be used. In boosting, at each stage, unexplained loss from prior iterations will be optimized rather than starting from scratch.
# 
# * **Weak learner to make predictions**: Decision trees are used as a weak learner in gradient boosting.
# 
# * **Additive model to add weak learners to minimize the loss function**: Trees are added one at a time and existing trees in the model are not changed. The gradient descent procedure is used to minimize the loss when adding trees.

#%%
from sklearn.ensemble import GradientBoostingClassifier

print("######## GradientBoostingClassifier #########")

#%%
gbc_clf = GradientBoostingClassifier()
gbc_clf.fit(X_train, y_train)


#%%
print_score(gbc_clf, X_train, y_train, X_test, y_test, train=True)


#%%
print_score(gbc_clf, X_train, y_train, X_test, y_test, train=False) # Test


#%%
Y_pred = gbc_clf.predict(test_df.drop('PassengerId',axis=1))

Y_pred

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submissions_gbc.csv', index=False)

#%%

print("######## xgb #########")

import xgboost as xgb


#%%
xgb_clf = xgb.XGBClassifier(max_depth=5, n_estimators=10000, learning_rate=0.3,
                            n_jobs=-1)


#%%
xgb_clf.fit(X_train, y_train)


#%%
print_score(xgb_clf, X_train, y_train, X_test, y_test, train=True)


#%%
print_score(xgb_clf, X_train, y_train, X_test, y_test, train=False)

#%% [markdown]
# # The best model is XGB in these runs

#%%
Y_pred = xgb_clf.predict(test_df.drop('PassengerId',axis=1))

Y_pred

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submissions_xgb.csv', index=False)

#%% [markdown]
# # Ensemble of ensembles - model stacking
# 
# * **Ensemble with different types of classifiers**: 
#   * Different types of classifiers (E.g., logistic regression, decision trees, random forest, etc.) are fitted on the same training data
#   * Results are combined based on either 
#     * majority voting (classification) or 
#     * average (regression)
#   
# 
# * **Ensemble with a single type of classifier**: 
#   * Bootstrap samples are drawn from training data 
#   * With each bootstrap sample, model (E.g., Individual model may be decision trees, random forest, etc.) will be fitted 
#   * All the results are combined to create an ensemble. 
#   * Suitabe for highly flexible models that is prone to overfitting / high variance. 
# 
# ***
# 
# ## Combining Method
# 
# * **Majority voting or average**: 
#   * Classification: Largest number of votes (mode) 
#   * Regression problems: Average (mean).
#   
#   
# * **Method of application of meta-classifiers on outcomes**: 
#   * Binary outcomes: 0 / 1 from individual classifiers
#   * Meta-classifier is applied on top of the individual classifiers. 
#   
#   
# * **Method of application of meta-classifiers on probabilities**: 
#   * Probabilities are obtained from individual classifiers. 
#   * Applying meta-classifier
#   
#%% [markdown]
# ## Model 1 : Decision Trees

#%%
from sklearn.tree import DecisionTreeClassifier


print("######## DecisionTreeClassifier #########")

tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)

print_score(tree_clf, X_train, y_train, X_test, y_test, train=True)
print_score(tree_clf, X_train, y_train, X_test, y_test, train=False)

#%% [markdown]
# ## Model 2: Random Forest

#%%
from sklearn.ensemble import RandomForestClassifier

print("######## RandomForestClassifier #########")


rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train.ravel())

print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)


#%%
en_en = pd.DataFrame()


#%%
tree_clf.predict_proba(X_train)


#%%
en_en['tree_clf'] = pd.DataFrame(tree_clf.predict_proba(X_train))[1]
en_en['rf_clf'] =  pd.DataFrame(rf_clf.predict_proba(X_train))[1]
col_name = en_en.columns
en_en = pd.concat([en_en, pd.DataFrame(y_train).reset_index(drop=True)], axis=1)


#%%
en_en.head()


#%%
tmp = list(col_name)
tmp.append('ind')
en_en.columns = tmp

#%% [markdown]
# # Meta Classifier

#%%
from sklearn.linear_model import LogisticRegression

print("######## LogisticRegression #########")


m_clf = LogisticRegression(fit_intercept=False)

m_clf.fit(en_en[['tree_clf', 'rf_clf']], en_en['ind'])


#%%
en_test = pd.DataFrame()


#%%
en_test['tree_clf'] = pd.DataFrame(tree_clf.predict_proba(X_test))[1]
en_test['rf_clf'] =  pd.DataFrame(rf_clf.predict_proba(X_test))[1]
col_name = en_en.columns
en_test['combined'] = m_clf.predict(en_test[['tree_clf', 'rf_clf']])


#%%
col_name = en_test.columns
tmp = list(col_name)
tmp.append('ind')


#%%
tmp


#%%
en_test = pd.concat([en_test, pd.DataFrame(y_test).reset_index(drop=True)], axis=1)


#%%
en_test.columns = tmp


#%%
print(pd.crosstab(en_test['ind'], en_test['combined']))


#%%
print(round(accuracy_score(en_test['ind'], en_test['combined']), 4))


#%%
print(classification_report(en_test['ind'], en_test['combined']))

#%% [markdown]
# # Using Single Classifier

#%%
#For self:
#df.Attrition.value_counts() / df.Attrition.count()


#%%
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import AdaBoostClassifier


#%%
pd.Series(list(y_train)).value_counts() / pd.Series(list(y_train)).count()


#%%
class_weight = {0:0.61, 1:0.38}


#%%
forest = RandomForestClassifier(class_weight=class_weight)

print("######## AdaBoostClassifier With forest as base_estimator #########")

#%%
ada = AdaBoostClassifier(base_estimator=forest, n_estimators=100,
                         learning_rate=0.5, random_state=42)


#%%
ada.fit(X_train, y_train.ravel())


#%%
print_score(ada, X_train, y_train, X_test, y_test, train=True)
print_score(ada, X_train, y_train, X_test, y_test, train=False)


print("######## BaggingClassifier With ada as base_estimator #########")


#%%
bag_clf = BaggingClassifier(base_estimator=ada, n_estimators=50,
                            max_samples=1.0, max_features=1.0, bootstrap=True,
                            bootstrap_features=False, n_jobs=-1,
                            random_state=42)


#%%
bag_clf.fit(X_train, y_train.ravel())


#%%
print_score(bag_clf, X_train, y_train, X_test, y_test, train=True)
print_score(bag_clf, X_train, y_train, X_test, y_test, train=False)


#%%
Y_pred = bag_clf.predict(test_df.drop('PassengerId',axis=1))

Y_pred

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submissions_bag_last.csv', index=False)


#%%



