import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets import load_digits, load_wine, load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import Exercise1 as e1
import Exercise2 as e2

# Load datasets
digits_in = load_digits()
digits_data = digits_in.data[:500]
digits_target = digits_in.target[:500]
wine = load_wine()
breast_cancer = load_breast_cancer()

cmap = ('Paired')
# Initialize PCA and t-SNE for 2 dimensional output, 
pca = PCA(n_components=2)
tsne = TSNE(n_components=2)

# 3.1 Comparison of DR techniques
def taskOne():
    # Sammon arguments: Dataset, iterations, threshold, learning rate - all of which are easily changeable
    # Perform Sammon mapping, PCA and t-SNE on the digits dataset
    sammon_digits = e2.sammon(digits_data, 500, 0.115, 0.7) # 0.12
    pca_digits = pca.fit_transform(digits_data)
    tsne_digits = tsne.fit_transform(digits_data)
    
    # Perform Sammon mapping, PCA and t-SNE on the wine dataset
    sammon_wine = e2.sammon(wine.data, 500, 0.062, 0.6)
    pca_wine = pca.fit_transform(wine.data)
    tsne_wine = tsne.fit_transform(wine.data)

    # Perform Sammon mapping, PCA and t-SNE on the breast cancer dataset
    sammon_breast_cancer = e2.sammon(breast_cancer.data, 500, 0.06, 0.2)
    pca_breast_cancer = pca.fit_transform(breast_cancer.data)
    tsne_breast_cancer = tsne.fit_transform(breast_cancer.data)

    # Create a scatterplot matrix and colormap
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))

    # Sammon Mapping
    ax[0, 0].scatter(sammon_digits[:, 0], sammon_digits[:, 1], c=digits_target, cmap=cmap)
    ax[0, 0].set_title("Sammon Mapping (Digits)")

    # PCA 
    ax[0, 1].scatter(pca_digits[:, 0], pca_digits[:, 1], c=digits_target, cmap=cmap)
    ax[0, 1].set_title("PCA (Digits)")

    # t-SNE
    ax[0, 2].scatter(tsne_digits[:, 0], tsne_digits[:, 1], c=digits_target, cmap=cmap)
    ax[0, 2].set_title("t-SNE (Digits)")

    # Wine dataset
    # Sammon Mapping
    ax[1, 0].scatter(sammon_wine[:, 0], sammon_wine[:, 1], c=wine.target, cmap=cmap)
    ax[1, 0].set_title("Sammon Mapping (Wine)")

    # PCA
    ax[1, 1].scatter(pca_wine[:, 0], pca_wine[:, 1], c=wine.target, cmap=cmap)
    ax[1, 1].set_title("PCA (Wine)")

    # t-SNE
    ax[1, 2].scatter(tsne_wine[:, 0], tsne_wine[:, 1], c=wine.target, cmap=cmap)
    ax[1, 2].set_title("t-SNE (Wine)")

    # Breast cancer
    # Sammon Mapping
    ax[2, 0].scatter(sammon_breast_cancer[:, 0], sammon_breast_cancer[:, 1], c=breast_cancer.target, cmap=cmap)
    ax[2, 0].set_title("Sammon Mapping (Breast cancer)")

    # PCA
    ax[2, 1].scatter(pca_breast_cancer[:, 0], pca_breast_cancer[:, 1], c=breast_cancer.target, cmap=cmap)
    ax[2, 1].set_title("PCA (Breast cancer)")

    # t-SNE
    ax[2, 2].scatter(tsne_breast_cancer[:, 0], tsne_breast_cancer[:, 1], c=breast_cancer.target, cmap=cmap)
    ax[2, 2].set_title("t-SNE (Breast cancer)")


# 3.2 Comparison of Clustering techniques
def taskTwo(k, iter):
    # DR on datasets
    X_digits = tsne.fit_transform(digits_data)
    X_wine = tsne.fit_transform(wine.data)
    X_breast_cancer = tsne.fit_transform(breast_cancer.data)

    # Bisecting k-Means
    bisecting_kmeans_digits = e1.bkmeans(X_digits, k, iter)
    bisecting_kmeans_wine = e1.bkmeans(X_wine, k, iter)
    bisecting_kmeans_breast_cancer = e1.bkmeans(X_breast_cancer, k, iter)

    # k-Means
    kmeans_digits = KMeans(n_clusters = k, n_init="auto").fit_predict(X_digits)
    kmeans_wine = KMeans(n_clusters = k, n_init="auto").fit_predict(X_wine)
    kmeans_breast_cancer = KMeans(n_clusters = k, n_init="auto").fit_predict(X_breast_cancer)

    # Hierarchical clustering (Agglomerative)
    agglomerative_clustering_digits = AgglomerativeClustering(n_clusters = k).fit_predict(X_digits)
    agglomerative_clustering_wine = AgglomerativeClustering(n_clusters = k).fit_predict(X_wine)
    agglomerative_clustering_breast_cancer = AgglomerativeClustering(n_clusters = k).fit_predict(X_breast_cancer)

    # Plotting
    fig, ax = plt.subplots(3, 3, figsize=(12, 12))

    # Bisecting k-Means
    ax[0, 0].scatter(X_digits[:, 0], X_digits[:, 1], c=bisecting_kmeans_digits, cmap=cmap)
    ax[0, 0].set_title("Bisecting k-Means (Digits)")

    # k-Means
    ax[0, 1].scatter(X_digits[:, 0], X_digits[:, 1], c=kmeans_digits, cmap=cmap)
    ax[0, 1].set_title("k-Means (Digits)")

    # Agglomerative clustering
    ax[0, 2].scatter(X_digits[:, 0], X_digits[:, 1], c=agglomerative_clustering_digits, cmap=cmap)
    ax[0, 2].set_title("Agglomerative clustering (Digits)")

    # Wine dataset
    # Bisecting k-Means
    ax[1, 0].scatter(X_wine[:, 0], X_wine[:, 1], c=bisecting_kmeans_wine, cmap=cmap)
    ax[1, 0].set_title("Bisecting k-Means (Wine)")

    # k-Means
    ax[1, 1].scatter(X_wine[:, 0], X_wine[:, 1], c=kmeans_wine, cmap=cmap)
    ax[1, 1].set_title("k-Means (Wine)")

    # Agglomerative clustering
    ax[1, 2].scatter(X_wine[:, 0], X_wine[:, 1], c=agglomerative_clustering_wine, cmap=cmap)
    ax[1, 2].set_title("Agglomerative clustering (Wine)")

    # Breast cancer dataset
    # Bisecting k-Means
    ax[2, 0].scatter(X_breast_cancer[:, 0], X_breast_cancer[:, 1], c=bisecting_kmeans_breast_cancer, cmap=cmap)
    ax[2, 0].set_title("Bisecting k-Means (Breast cancer)")

    # k-Means
    ax[2, 1].scatter(X_breast_cancer[:, 0], X_breast_cancer[:, 1], c=kmeans_breast_cancer, cmap=cmap)
    ax[2, 1].set_title("k-Means (Breast cancer)")

    # Agglomerative clustering
    ax[2, 2].scatter(X_breast_cancer[:, 0], X_breast_cancer[:, 1], c=agglomerative_clustering_breast_cancer, cmap=cmap)
    ax[2, 2].set_title("Agglomerative clustering (Breast cancer)")


# Select which task you want to perform (or both)
# Task one, select sammon arguments for each idividual dataset in the code above.
# Sammon arguments: Dataset, iterations, threshold, learning rate
taskOne()

# Task two, select variables as arguments for taskTwo, k = clusters, iter = iterations
taskTwo(k = 3, iter = 15)

plt.show()