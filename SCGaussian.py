import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons

class SpectralClustering:
    def __init__(self, n_clusters, sigma):
        self.n_clusters = n_clusters
        self.sigma = sigma

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
        # Gaussian similarity matrix
        dist_sq = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
        W = np.exp(-dist_sq / (2 * self.sigma ** 2))
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

#X, y_true = make_blobs(n_samples=300, centers=5, cluster_std=0.60, random_state=0)
X, y_true = make_moons(n_samples=1000, noise=0.04, random_state=42)
sc = SpectralClustering(n_clusters=2, sigma=1)
y_pred = sc.fit_predict(X)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.scatter(X[:, 0], X[:, 1], c=y_true)
ax1.set_title('True Clusters')
ax2.scatter(X[:, 0], X[:, 1], c=y_pred)
ax2.set_title('Spectral Clustering Results')
plt.show()