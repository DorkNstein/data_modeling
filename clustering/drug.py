import pandas as pd
from sklearn import datasets
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import sys
sys.path.append('../')
from scikit_clustering import k_means, get_optimal_cluster_count, pca_transform, get_pca_comp_count, get_clustering_scores