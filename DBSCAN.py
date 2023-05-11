import numpy as np
from scipy.spatial.distance import cdist
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

class DBSCAN:
    def __init__(self, eps=0.5, min_pts=5):
        self.eps = eps
        self.min_pts = min_pts
        self.labels_ = None

    def fit(self, X):
        n = X.shape[0]
        visited = np.zeros(n, dtype=bool)
        self.labels_ = np.full(n, -1, dtype=int)
        cluster_id = 0
        for i in range(n):
            if not visited[i]:
                visited[i] = True
                neighbors = self.range_query(X, i)
                if len(neighbors) >= self.min_pts:
                    self.expand_cluster(X, visited, neighbors, cluster_id)
                    print("Labels updated:", set(self.labels_))
                    cluster_id += 1
        return self.labels_

    def range_query(self, X, i):
        dist = cdist(X[i:i + 1], X)
        return np.where(dist <= self.eps)[1]

    def expand_cluster(self, X, visited, neighbors, cluster_id):
        self.labels_[neighbors] = cluster_id
        while len(neighbors) > 0:
            i = neighbors[0]
            if not visited[i]:
                visited[i] = True
                neighbors2 = self.range_query(X, i)
                if len(neighbors2) >= self.min_pts:
                    neighbors = np.concatenate([neighbors[1:], neighbors2])
                    self.labels_[neighbors2] = cluster_id
            neighbors = neighbors[1:]

X, _ = make_moons(n_samples=1000, noise=0.08, random_state=42)
np.random.seed(42)
n_noise = 50
noise = np.random.rand(n_noise, 2)
X = np.concatenate((X, noise), axis=0)

dbscan = DBSCAN(eps=0.1, min_pts=5)
initial_labels = dbscan.labels_
labels = dbscan.fit(X)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print("Number of clusters found:", n_clusters)
colors = [plt.cm.jet(i / n_clusters) for i in range(n_clusters)]
plt.scatter(X[labels == -1, 0], X[labels == -1, 1], s=10, color='black')
for i in range(n_clusters):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], s=10, color=colors[i])
plt.show()