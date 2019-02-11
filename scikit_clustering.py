import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, silhouette_score, mutual_info_score, normalized_mutual_info_score


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

    print("\n")
    print("K-MEANS with {0} clusters:".format(n_clust))
    print('% 9s' % 'inertia  homo    compl   v-meas   ARI     AMI     silhouette')
    print('%i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f' %
        (k_means.inertia_,
        adjusted_mutual_info_score(true_labels, y_clust),
        adjusted_rand_score(true_labels, y_clust),
        completeness_score(true_labels, y_clust),
        homogeneity_score(true_labels, y_clust),
        mutual_info_score(true_labels, y_clust),
        # v_measure_score(true_labels, y_clust)# silhouette_score(data_frame, y_clust, metric = 'euclidean')
        normalized_mutual_info_score(true_labels, y_clust),
        ))
