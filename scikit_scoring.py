# !#/usr/local/bin/python2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, LassoCV, Lasso, LassoLarsCV, ElasticNet, BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# """ """ Data Cleanup """ """

from sklearn.model_selection import train_test_split


def data_cleanup(data, output):
    correlation = data.corr(method='pearson')
    columns = correlation.nlargest(10, output).index
    print("columns:")
    print(columns)
    correlation_map = np.corrcoef(data[columns].values.T)
    correlation_map
    print("correlation_map:")
    print(correlation_map)

    X = data[columns]
    target = X[output]
    features = X.drop(output, axis=1)

    xa = features.describe()
    ya = target.describe()
    print(xa)
    print(ya)

    Y = target.values
    X = features.values
    unique_count = len(np.unique(Y))
    classType = 'multi'
    if (unique_count <= 2):
        classType = 'binary'

    print(classType)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.20, random_state=42)

    return X_train, X_test, Y_train, Y_test, features, classType


def get_classification_values(data, output):
    X_train, X_test, Y_train, Y_test, features, classType = data_cleanup(
        data, output)
    calc_feature_importance(X_train, X_test, Y_train, Y_test, features)
    get_classification_scores(X_train, X_test, Y_train, Y_test, classType)

# """ """ Regression """ """


def get_regression_values(data, output):
    X_train, X_test, Y_train, Y_test, features, classType = data_cleanup(
        data, output)
    get_regression_scores(X_train, X_test, Y_train, Y_test)


def get_regression_scores(X_train, X_test, Y_train, Y_test):
    pipelines = []
    pipelines.append(('ScaledLR', Pipeline(
        [('Scaler', StandardScaler()), ('LR', LinearRegression())])))
    pipelines.append(('ScaledRIDGE', Pipeline(
        [('Scaler', StandardScaler()), ('RIDGE', Ridge())])))
    pipelines.append(('ScaledLASSO', Pipeline(
        [('Scaler', StandardScaler()), ('LASSO', Lasso())])))
    pipelines.append(('ScaledLASSOCV', Pipeline(
        [('Scaler', StandardScaler()), ('LASSOCV', LassoCV())])))
    pipelines.append(('ScaledLASSOLarsCV', Pipeline(
        [('Scaler', StandardScaler()), ('LASSOLarsCV', LassoLarsCV())])))
    pipelines.append(('ScaledEN', Pipeline(
        [('Scaler', StandardScaler()), ('EN', ElasticNet())])))
    pipelines.append(('ScaledBAYESIAN', Pipeline(
        [('Scaler', StandardScaler()), ('BAYESIAN', BayesianRidge())])))
    pipelines.append(('ScaledKNN', Pipeline(
        [('Scaler', StandardScaler()), ('KNN', KNeighborsRegressor())])))
    pipelines.append(('ScaledCART', Pipeline(
        [('Scaler', StandardScaler()), ('CART', DecisionTreeRegressor())])))
    pipelines.append(('ScaledGBM', Pipeline(
        [('Scaler', StandardScaler()), ('GBM', GradientBoostingRegressor())])))

    results = []
    names = []

    for name, model in pipelines:
        ts = time.time()
        kfold = KFold(n_splits=10, random_state=21)
        cv_results = cross_val_score(
            model, X_train, Y_train, cv=kfold, scoring='neg_mean_squared_error')
        cv_results_abs = cross_val_score(
            model, X_train, Y_train, cv=kfold, scoring='neg_mean_absolute_error')
        # cv_results_sq_log = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = 'neg_mean_squared_log_error')
        cv_results_median_abs = cross_val_score(
            model, X_train, Y_train, cv=kfold, scoring='neg_median_absolute_error')
        cv_r2 = cross_val_score(model, X_train, Y_train,
                                cv=kfold, scoring='r2')
        cv_explained_variance = cross_val_score(
            model, X_train, Y_train, cv=kfold, scoring='explained_variance')
        ts_2 = time.time()
        results.append(cv_results)
        names.append(name)
        msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
        msg_abs = "%f (%f)" % (cv_results_abs.mean(), cv_results_abs.std())
        # # msg_sq_log = "%f (%f)" % (cv_results_sq_log.mean(), cv_results_sq_log.std())
        msg_median_abs = "%f (%f)" % (
            cv_results_median_abs.mean(), cv_results_median_abs.std())
        msg_r2 = "%f (%f)" % (cv_r2.mean(), cv_r2.std())
        msg_explained_variance = "%f (%f)" % (
            cv_explained_variance.mean(), cv_explained_variance.std())
        print(name)
        print(msg_explained_variance)
        print(msg_abs)
        print(msg)
        # print(msg_sq_log)
        print(msg_median_abs)
        print(msg_r2)
        print("%f" % (ts_2 - ts))
        print('\n')


# """ """ Classification """ """


def base_model_classification(X_train, X_test, Y_train, Y_test):
    # Base Model for classification Evaluation
    clf1 = RandomForestClassifier()
    clf1.fit(X_train, Y_train)
    test_predictions1 = clf1.predict(X_test)
    return clf1, test_predictions1


def draw_confusion_matrices(confusion_matrices, class_names):
    class_names = class_names.tolist()
    for cm in confusion_matrices:
        classifier, cm = cm[0], cm[1]
        print(cm)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm)
        plt.title('Confusion matrix for %s' % classifier)
        fig.colorbar(cax)
        ax.set_xticklabels([''] + class_names)
        ax.set_yticklabels([''] + class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()


# Calculate Feature Importance
def calc_feature_importance(X_train, X_test, Y_train, Y_test, features):

    clf1, test_predictions1 = base_model_classification(
        X_train, X_test, Y_train, Y_test)
    importances = clf1.feature_importances_[:10]
    std = np.std([tree.feature_importances_ for tree in clf1.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")
    features = features.columns
    feature_range = len(features) if len(features) < 10 else 10
    for f in range(feature_range):
        print("%d. %s (%f)" % (f + 1, features[f], importances[indices[f]]))
    # # Plot the feature importances of the forest
    # #import pylab as pl
    # plt.figure()
    # plt.title("Feature importances")
    # plt.bar(range(feature_range), importances[indices], yerr=std[indices], color="r", align="center")
    # plt.xticks(range(feature_range), indices)
    # plt.xlim([-1, feature_range])
    # plt.show()


def get_classification_scores(X_train, X_test, Y_train, Y_test, classType):
    classifiers = [
        KNeighborsClassifier(3),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier(),
        GaussianNB(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        BaggingClassifier()]

    # Logging for Visual Comparison
    log_cols = ["Classifier", "F-score", "Log Loss"]
    log = pd.DataFrame(columns=log_cols)

    from sklearn.metrics import accuracy_score, log_loss, f1_score, roc_curve, auc, average_precision_score, balanced_accuracy_score, brier_score_loss, precision_score, recall_score, roc_auc_score

    times = []

    for clf in classifiers:
        ts = time.time()
        clf.fit(X_train, Y_train)
        name = clf.__class__.__name__

        print("="*30)
        print(name)

        print('****Results****')
        test_predictions = clf.predict(X_test)

        # accuracy_score
        print('accuracy_score:')
        acc_score = accuracy_score(Y_test, test_predictions)
        print(format(acc_score))

        # balanced_acc_score
        print('balanced_acc_score:')
        balanced_acc_score = balanced_accuracy_score(Y_test, test_predictions)
        print(format(balanced_acc_score))

        # avg_precision_score
        # avg_precision_score = average_precision_score(Y_test, test_predictions)
        # print(avg_precision_score)

        # brier_score
        # f1_score
        f_score_none = f1_score(Y_test, test_predictions, average=None)
        if (classType == 'binary'):
            print('brier_score:')
            brier_score = brier_score_loss(Y_test, test_predictions)
            print(brier_score)
            print(format(brier_score))
            f_score = f1_score(Y_test, test_predictions)

        else:
            f_score = f_score_none

        print('f1_score:')
        print(f_score)

        #     f_score_samples = f1_score(
        #         Y_test, test_predictions, average='samples')
        #     print("F-score samples: {:.4%}".format(f_score_samples))

        # f_score_micro = f1_score(Y_test, test_predictions, average='micro')
        # f_score_macro = f1_score(Y_test, test_predictions, average='macro')
        # f_score_weighted = f1_score(
        #     Y_test, test_predictions, average='weighted')
        # print(format(f_score_none))
        # print(format(f_score_micro))
        # print(format(f_score_macro))
        # print(format(f_score_weighted))

        # Log Loss
        test_predictions_1 = clf.predict_proba(X_test)
        try:
            ll = log_loss(Y_test, test_predictions_1)
        except:
            ll = np.nan

        print('log_loss:')
        print(format(ll))

        # precision_score
        print('precision_score:')
        pres_score = precision_score(Y_test, test_predictions, average=None)
        print(format(pres_score))

        # recall_score
        print('recall_score:')
        rec_score = recall_score(Y_test, test_predictions, average=None)
        print(format(rec_score))

        if (classType == 'binary'):
            roc_auc__score = roc_auc_score(
                Y_test, test_predictions, average=None)
            print('roc_auc__score:')
            print(roc_auc__score)
            # auc_score
            # fpr, tpr, thresholds = roc_curve(
            # Y_test, test_predictions, pos_label=None)
            # auc_score = auc(fpr, tpr)
            # print(format(auc_score))

        ts_2 = time.time()
        print("Time: %f" % (ts_2 - ts))
        times.append(ts_2 - ts)

        log_entry = pd.DataFrame([[name, f_score*100, ll]], columns=log_cols)
        log = log.append(log_entry)

    print(times)
