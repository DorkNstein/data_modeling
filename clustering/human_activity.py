
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score, mutual_info_score, normalized_mutual_info_score
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from IPython.display import display
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import sys
sys.path.append('../')
from scikit_clustering import get_optimal_cluster_count, get_clustering_scores

# %matplotlib inline


np.random.seed(123)

Data = pd.read_csv('../input/clustering/simplifiedhuarus/train.csv')
Data.sample(5)

print('Shape of the data set: ' + str(Data.shape))
# save labels as string
Labels = Data['activity']
Data = Data.drop(['rn', 'activity'], axis=1)
Labels_keys = Labels.unique().tolist()
Labels = np.array(Labels)
print('Activity labels: ' + str(Labels_keys))

# check for missing values
Temp = pd.DataFrame(Data.isnull().sum())
Temp.columns = ['Sum']
print('Amount of rows with missing values: ' +
      str(len(Temp.index[Temp['Sum'] > 0])))
# normalize the dataset
scaler = StandardScaler()
Data = scaler.fit_transform(Data)

# check the optimal k value
optimal_count = get_optimal_cluster_count(data_frame=Data)

# change labels into binary: 0 - not moving, 1 - moving
Labels_binary = Labels.copy()
for i in range(len(Labels_binary)):
    if (Labels_binary[i] == 'STANDING' or Labels_binary[i] == 'SITTING' or Labels_binary[i] == 'LAYING'):
        Labels_binary[i] = 0
    else:
        Labels_binary[i] = 1
Labels_binary = np.array(Labels_binary.astype(int))

get_clustering_scores(n_clust=optimal_count, data_frame=Data,
        true_labels=Labels_binary)

# # check for optimal number of features
# pca = PCA(random_state=123)
# pca.fit(Data)
# features = range(pca.n_components_)
# print("features: {0}".format(pca.n_components_))
# print pca.explained_variance_ratio_

# # plt.figure(figsize=(8, 4))
# # plt.bar(features[:15], pca.explained_variance_[:15], color='lightskyblue')
# # plt.xlabel('PCA feature')
# # plt.ylabel('Variance')
# # plt.xticks(features[:15])
# # plt.show()

