from sklearn.cluster import AgglomerativeClustering, KMeans
import numpy as np
from scipy.spatial.distance import cdist


def eigen_clusters(num_clusters, eigenvectors):
    """
    :return: cluster labels for all states
    """
    t_eigenvectors = np.transpose(np.real(eigenvectors))  # get eigenvectors in correct shape, only take real components
    # clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage="ward").fit(t_eigenvectors)
    clustering = KMeans(n_clusters=num_clusters).fit(t_eigenvectors)
    clusters = clustering.labels_
    inertia = clustering.inertia_
    return clusters, inertia


