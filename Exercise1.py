import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Bisecting k-Means
def bkmeans (X, k, iter):
    # Initialize values and create arrays
    X = MinMaxScaler().fit_transform(X)
    divisions, cluster_number = 0, 0
    all_indices = np.zeros(len(X), dtype = 'int64')
    indice = np.where(all_indices == 0)
    largest_cluster = X.copy()
    
    while divisions < (k - 1):
        divisions += 1

        # Clustering k-Means on the largest cluster
        kmeans = KMeans(n_clusters = 2, n_init = iter).fit(largest_cluster)
        labels = kmeans.labels_
        
        # Assign new cluster numbers
        first_indices = np.where(labels == 1)
        second_indices = np.where(labels == 0)
        labels[second_indices] = cluster_number
        labels[first_indices] = divisions

        # Update cluster indices
        all_indices[indice] = labels
        
        # Find the largest cluster (with most indices)
        counter = np.bincount(all_indices)
        indice = np.where(np.argmax(counter) == all_indices)
        
        # Update largest cluster and update cluster number
        largest_cluster = X[indice]
        cluster_number = all_indices[indice][0]
    return all_indices
