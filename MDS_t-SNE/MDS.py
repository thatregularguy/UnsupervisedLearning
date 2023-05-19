import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class MDS:
    def __init__(self, n_dims=2, max_iter=100, eta=0.1):
        self.n_dims = n_dims
        self.max_iter = max_iter
        self.eta = eta
        self.embedding = None

    def fit_transform(self, X):
        distances = self.pairwise_distances(X)
        n_samples = X.shape[0]

        self.embedding = np.random.rand(n_samples, self.n_dims) # Output matrix init
        # "Gradient descent" of output matrix
        for iter in range(self.max_iter):
            # Pairwise distances of output
            current_distances = self.pairwise_distances(self.embedding)

            # Sammon's mapping stress
            stress = self.sammon_stress(distances, current_distances)
            if iter % 50 == 0:
                print(f'Iter: {iter}, Stress: {stress}')

            # Gradient calculation
            gradient = self.compute_gradient(distances, current_distances)

            # Output matrix update
            self.embedding -= self.eta * gradient

        return self.embedding

    def pairwise_distances(self, X):
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))
        # Creating the symmetric matrix of pairwise distances
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                distances[i, j] = np.linalg.norm(X[i] - X[j])
                distances[j, i] = distances[i, j]

        return distances

    def sammon_stress(self, distances, current_distances):
        diff = distances - current_distances
        numerator = np.sum(diff ** 2) # Numerator of stress function
        denominator = np.sum(distances ** 2) # Denominator of stress function
        stress = numerator / denominator
        return stress

    def compute_gradient(self, distances, current_distances):
        # For the Sammon's mapping the gradient equals to -2 * (diff of dist's) / denominator
        n_samples = distances.shape[0]
        gradient = np.zeros((n_samples, self.n_dims))
        denominator = np.sum(distances ** 2)

        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    diff = distances[i, j] - current_distances[i, j]
                    factor = -2 * diff / denominator
                    gradient[i] += factor * (self.embedding[i] - self.embedding[j])
        return gradient

# Load the Iris dataset
data = load_iris()
input_data = data.data
sc = StandardScaler()
input_data = sc.fit_transform(input_data)

# MDS to 2d
mds_2d = MDS(n_dims=2, max_iter=3000, eta=5)
output_data_2d = mds_2d.fit_transform(input_data)
print('---------------------MDS done!---------------------')
# 2D plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(output_data_2d[:, 0], output_data_2d[:, 1], c=data.target)
plt.title("Transformed Data (MDS, 2D)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
# Annotations
for i, (x, y) in enumerate(zip(output_data_2d[:, 0], output_data_2d[:, 1])):
    plt.annotate(i, (x, y), textcoords="offset points", xytext=(0, 5), ha='center')

mds_3d = MDS(n_dims=3, max_iter=3000, eta=5)
output_data_3d = mds_3d.fit_transform(input_data)
print('---------------------MDS done!---------------------')
# 3D plot
ax = plt.subplot(1, 2, 2, projection='3d')
ax.scatter(output_data_3d[:, 0], output_data_3d[:, 1], output_data_3d[:, 2], c=data.target)
ax.set_title("Transformed Data (MDS, 3D)")
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
ax.set_zlabel("Dimension 3")
# Annotations
for i, (x, y, z) in enumerate(zip(output_data_3d[:, 0], output_data_3d[:, 1], output_data_3d[:, 2])):
    ax.text(x, y, z, str(i), color='k', fontsize=8)

plt.tight_layout()
plt.show()