import pandas as pd
from sklearn import datasets
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sys
sys.path.append('../')
from scikit_clustering import k_means, get_optimal_cluster_count, pca_transform, get_pca_comp_count, get_clustering_scores


X, y = make_blobs(n_samples=10000, centers=2, n_features=2,
                  random_state=0)
data = X
target = y

# dataset = datasets.load_iris()
# dataset = datasets.load_wine()
# data = dataset.data
# target = dataset.target

optimal_count = get_optimal_cluster_count(data_frame=data)
get_clustering_scores(n_clust=optimal_count,
                      data_frame=data, true_labels=target)
