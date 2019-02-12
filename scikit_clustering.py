from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, ward_tree, AgglomerativeClustering, DBSCAN, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score, mutual_info_score, normalized_mutual_info_score
import time
import pandas as pd


def get_kmeans_inertias(data_frame):
    ks = range(1, 10)
    inertias = []

    for k in ks:
        model = KMeans(n_clusters=k)
        model.fit(data_frame)
        inertias.append(model.inertia_)

    return inertias


def get_optimal_cluster_count(data_frame):

    inertias = get_kmeans_inertias(data_frame)

    for i, x in enumerate(inertias):
        if (i > 0):
            optimalDiff = (inertias[i - 1] - x) / \
                (x - inertias[len(inertias) - 1])
            if (optimalDiff < 1):
                return i
                break
            else:
                continue
        else:
            continue


def k_means(n_clust, data_frame, true_labels):
    ts = time.time()
    """
    Function k_means applies k - means clustering alrorithm on dataset and prints the crosstab of cluster and actual labels
    and clustering performance parameters.

    Input:
        n_clust - number of clusters(k value)
    data_frame - dataset we want to cluster
    true_labels - original labels

    Output:
        1 - crosstab of cluster and actual labels
    2 - performance table 
    """

    k_means = KMeans(n_clusters=n_clust, random_state=123, n_init=30)
    k_means.fit(data_frame)
    c_labels = k_means.labels_
    df = pd.DataFrame(
        {'clust_label': c_labels, 'orig_label': true_labels.tolist()})
    ct = pd.crosstab(df['clust_label'], df['orig_label'])
    y_clust = k_means.predict(data_frame)  # display(ct)
    print("K-MEANS with {0} clusters:".format(n_clust))
    scores(data_frame, true_labels, y_clust)
    ts_2 = time.time()
    print("Time: %f" % (ts_2 - ts))


def affinity_prop(n_clust, data_frame, true_labels):
    ts = time.time()
    clustering = AffinityPropagation().fit(data_frame)
    y_clust = clustering.predict(data_frame)

    print("Affinity_prop with {0} clusters:".format(n_clust))
    scores(data_frame, true_labels, y_clust)
    ts_2 = time.time()
    print("Time: %f" % (ts_2 - ts))


def mean_shift(n_clust, data_frame, true_labels):
    ts = time.time()
    clustering = MeanShift().fit(data_frame)
    y_clust = clustering.predict(data_frame)

    print("mean_shift with {0} clusters:".format(n_clust))
    scores(data_frame, true_labels, y_clust)
    ts_2 = time.time()
    print("Time: %f" % (ts_2 - ts))


def spectral_clustering(n_clust, data_frame, true_labels):
    ts = time.time()
    y_clust = SpectralClustering(n_clusters=n_clust, assign_labels="discretize",
                                 random_state=123).fit_predict(data_frame)
    # y_clust = clustering.predict(data_frame)

    print("spectral_clustering with {0} clusters:".format(n_clust))
    scores(data_frame, true_labels, y_clust)
    ts_2 = time.time()
    print("Time: %f" % (ts_2 - ts))


def ward_clustering(n_clust, data_frame, true_labels):
    ts = time.time()
    y_clust = ward_tree(
        X=data_frame, n_clusters=n_clust).fit_predict(data_frame)
    # y_clust = clustering.predict(data_frame)

    print("ward_clustering with {0} clusters:".format(n_clust))
    scores(data_frame, true_labels, y_clust)
    ts_2 = time.time()
    print("Time: %f" % (ts_2 - ts))


def agglomerative_clustering(n_clust, data_frame, true_labels):
    ts = time.time()
    y_clust = AgglomerativeClustering(
        n_clusters=n_clust).fit_predict(data_frame)
    # y_clust = clustering.predict(data_frame)

    print("agglomerative_clustering with {0} clusters:".format(n_clust))
    scores(data_frame, true_labels, y_clust)
    ts_2 = time.time()
    print("Time: %f" % (ts_2 - ts))


def dbscan_clustering(n_clust, data_frame, true_labels):
    ts = time.time()
    y_clust = DBSCAN().fit_predict(data_frame)
    # y_clust = clustering.predict(data_frame)

    print("dbscan_clustering with {0} clusters:".format(n_clust))
    scores(data_frame, true_labels, y_clust)
    ts_2 = time.time()
    print("Time: %f" % (ts_2 - ts))


def gaussian_mixture(n_clust, data_frame, true_labels):
    ts = time.time()
    y_clust = GaussianMixture().fit_predict(data_frame)
    # y_clust = clustering.predict(data_frame)

    print("gaussian_mixture with {0} clusters:".format(n_clust))
    scores(data_frame, true_labels, y_clust)
    ts_2 = time.time()
    print("Time: %f" % (ts_2 - ts))


def birch_clustering(n_clust, data_frame, true_labels):
    ts = time.time()
    clustering = Birch(n_clusters=n_clust).fit(data_frame)
    y_clust = clustering.predict(data_frame)
    print("birch_clustering with {0} clusters:".format(n_clust))
    scores(data_frame, true_labels, y_clust)
    ts_2 = time.time()
    print("Time: %f" % (ts_2 - ts))


def scores(data_frame, true_labels, y_clust):
    print('% 9s' % 'inertia  homo    compl   v-meas   ARI     AMI     silhouette')
    # print('%i' % (k_means.inertia_))
    print(adjusted_mutual_info_score(true_labels, y_clust))
    print(adjusted_rand_score(true_labels, y_clust))
    print(completeness_score(true_labels, y_clust))
    print(homogeneity_score(true_labels, y_clust))
    print(mutual_info_score(true_labels, y_clust))
    # print(v_measure_score(true_labels, y_clust))
    print(normalized_mutual_info_score(true_labels, y_clust))
    # print(silhouette_score(data_frame, y_clust, metric='euclidean'))


def get_clustering_scores(n_clust, data_frame, true_labels):
    pca_comp = get_pca_comp_count(data_frame)
    print("pca_comp: {0}".format(pca_comp))
    Data_reduced = pca_transform(data_frame, n_comp=pca_comp)

    print("\n")
    print("###### K Means ######")
    k_means(n_clust, data_frame, true_labels)
    k_means(n_clust, Data_reduced, true_labels)

    print("\n")
    print("###### Affinity Propogation ######")
    affinity_prop(n_clust, data_frame, true_labels)
    affinity_prop(n_clust, Data_reduced, true_labels)

    print("\n")
    print("###### MeanShift Bandwidth ######")
    mean_shift(n_clust, data_frame, true_labels)
    mean_shift(n_clust, Data_reduced, true_labels)

    print("\n")
    print("###### Spectral Clustering ######")
    # spectral_clustering(n_clust, data_frame, true_labels)
    # spectral_clustering(n_clust, Data_reduced, true_labels)
    # ward_clustering(n_clust, data_frame, true_labels)

    print("\n")
    print("###### agglomerative Clustering ######")
    agglomerative_clustering(n_clust, data_frame, true_labels)
    agglomerative_clustering(n_clust, Data_reduced, true_labels)

    print("\n")
    print("###### dbscan Clustering ######")
    dbscan_clustering(n_clust, data_frame, true_labels)
    dbscan_clustering(n_clust, Data_reduced, true_labels)
    # gaussian_mixture(n_clust, data_frame, true_labels)

    print("\n")
    print("###### Birch Clustering ######")
    birch_clustering(n_clust, data_frame, true_labels)
    birch_clustering(n_clust, Data_reduced, true_labels)


def get_optimal_pca_count(variance_ratios):
    for i, x in enumerate(variance_ratios):
        if (i > 0):
            optimalDiff = (variance_ratios[i - 1] - x) / \
                (variance_ratios[i - 1])
            if (optimalDiff > 0.7):
                return i
                break
            else:
                continue
        else:
            continue


# check for optimal number of features
def get_pca_comp_count(data_frame):
    pca = PCA(random_state=123)
    pca.fit(data_frame)
    # this tells us the extent to which each component explains the original dataset.
    # - See more at: https://shankarmsy.github.io/posts/pca-sklearn.html#sthash.t2njrxyW.dpuf
    return get_optimal_pca_count(pca.explained_variance_ratio_)


def pca_transform(data_frame, n_comp):
    pca = PCA(n_components=n_comp, random_state=123)
    global Data_reduced
    Data_reduced = pca.fit_transform(data_frame)
    print("\n")
    print('Shape of the new Data df: ' + str(Data_reduced.shape))
    return Data_reduced
