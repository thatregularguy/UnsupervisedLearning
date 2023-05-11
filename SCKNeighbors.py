import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn.neighbors import NearestNeighbors

class SpectralClustering:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters

    def fit_predict(self, X):
        # Similarity matrix
        W = self.similarity_matrix(X)
        # Laplacian matrix
        L = self.laplacian_matrix(W)
        # Eigenvectors and eigenvalues
        eigvals, eigvecs = self.eigenvectors(L)
        # Lower-dimensional representation of the data
        embedding = eigvecs[:, :self.n_clusters]
        # K-means clustering to the embedding
        kmeans = KMeans(n_clusters=self.n_clusters)
        clusters = kmeans.fit_predict(embedding)
        return clusters

    def similarity_matrix(self, X):
        # KNN graph
        knn = NearestNeighbors(n_neighbors=10)
        knn.fit(X)
        dists, indices = knn.kneighbors(X)
        W = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            W[i, indices[i]] = 1
            W[indices[i], i] = 1
        return W

    def laplacian_matrix(self, W):
        # Unnormalized Laplacian matrix
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        return L

    def eigenvectors(self, L):
        # Eigenvectors and eigenvalues
        eigvals, eigvecs = np.linalg.eigh(L)
        idx = eigvals.argsort()
        eigvecs = eigvecs[:, idx]
        eigvals = eigvals[idx]
        return eigvals, eigvecs

# X, y_true = make_blobs(n_samples=300, centers=5, cluster_std=0.60, random_state=0)
X, y_true = make_moons(n_samples=1000, noise=0.04, random_state=42)
sc = SpectralClustering(n_clusters=2)
y_pred = sc.fit_predict(X)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.scatter(X[:, 0], X[:, 1], c=y_true)
ax1.set_title('True Clusters')
ax2.scatter(X[:, 0], X[:, 1], c=y_pred)
ax2.set_title('Spectral Clustering Results')
plt.show()