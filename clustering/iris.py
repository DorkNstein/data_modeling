import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sys
sys.path.append('../')
from scikit_clustering import k_means, get_optimal_cluster_count, pca_transform, get_pca_comp_count, get_clustering_scores

# dataset = datasets.load_iris()
dataset = datasets.load_wine()
data = dataset.data
target = dataset.target

optimal_count = get_optimal_cluster_count(data_frame=data)
get_clustering_scores(n_clust=optimal_count,
                      data_frame=data, true_labels=target)
