import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
import sys
sys.path.append('../')
from scikit_clustering import k_means, get_optimal_cluster_count, k_means

iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target

optimal_count = get_optimal_cluster_count(data_frame=X_iris)

k_means(n_clust=optimal_count, data_frame=X_iris, true_labels=y_iris)
