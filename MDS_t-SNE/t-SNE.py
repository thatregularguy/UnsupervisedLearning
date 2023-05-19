import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

class TSNE:
    def __init__(self, perplexity=30, learning_rate=200, momentum=0.5, max_iter=5000, n_components=2):
        self.perplexity = perplexity
        self.lr = learning_rate
        self.momentum = momentum
        self.max_iter = max_iter
        self.n_components = n_components

    def fit_transform(self, X):
        P = self._compute_pairwise_similarities(X)  # Calculating the P{j|i}
        Y = np.random.normal(0, 1e-4, size=(X.shape[0], self.n_components))  # Initialization
        Y_prev = np.zeros_like(Y)  # Required for accumulation

        for i in tqdm(range(self.max_iter)):
            # Formula of update:
            # Y(t) = Y(t-1) - learning_rate * gradient + momentum(t) * (Y(t-1) - Y(t-2))
            dCdY = self._compute_gradient(P, Y)  # Gradient
            # Updating the output matrix
            Y_update = - self.lr * dCdY + self.momentum * (Y - Y_prev)
            Y = Y + Y_update
            Y = self._clip(Y)  # Clipping the values that can cause numerical errors
            Y_prev = Y - Y_update  # Saving the previous values of output
            C = self._compute_cost(P, Y)
            if i % 50 == 0:
                print("Iteration", i, "Cost", C)
        return Y

    def _compute_pairwise_similarities(self, X):
        dists = squareform(pdist(X, 'euclidean')) ** 2
        P = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            beta = 1 / (2 * self.perplexity ** 2)  # Using the perplexity as a sigma
            distances = np.exp(-beta * dists[i])  # Calculating the numerator
            P[i] = distances / np.sum(distances)  # Calculating the probabilities
        P = (P + P.T) / (2 * X.shape[0])  # Symmetrizing the matrix
        np.fill_diagonal(P, 0)  # Filling the diagonals with 0
        return P

    def _compute_Q(self, Y):
        dists = squareform(pdist(Y, 'euclidean'))
        Q = 1 / (1 + dists ** 2)  # Calculating the numerator
        np.fill_diagonal(Q, 0)  # Filling the diagonals with 0
        Q /= np.sum(Q)  # Calculating the Q matrix
        return Q

    def _compute_gradient(self, P, Y):
        # The formula of gradient:
        # dC/dY = 4 * sum((p{i|j}-q{i|j}) * (y(i)-y(j)) * ((1+||y(i)-y(j)||^2)^(-1)))
        Q = self._compute_Q(Y)
        # difference matrix
        Y_diff = Y[:, np.newaxis, :] - Y[np.newaxis, :, :]
        # squared distances
        dists_squared = np.sum(Y_diff ** 2, axis=-1)
        # t-SNE gradient
        factor = 4 * ((P - Q)[:, :, np.newaxis] * Y_diff)
        factor *= (dists_squared + 1)[:, :, np.newaxis]
        dCdY = np.sum(factor, axis=1)
        return dCdY

    def _clip(self, Y): # for numerical issues
        Y[Y < -1e15] = -1e15
        Y[Y > 1e15] = 1e15
        return Y

    def _compute_cost(self, P, Y):
        Q = self._compute_Q(Y)
        C = np.sum(P * np.log((P + 1e-12) / (Q + 1e-12)))
        return C

# Load the Iris dataset
iris = load_iris()
sc = StandardScaler()
X = sc.fit_transform(iris.data)
y = iris.target

# Initialize and fit t-SNE
# Some info from my tests:
# For 5000 iters and 3-dimensional ouput the optimal range of params:
# perp <= 40, lr >= 5, momentum < 1
tsne = TSNE(perplexity=40, learning_rate=2, momentum=0.9, max_iter=5000, n_components=3)
Y = tsne.fit_transform(X)

# Plot the 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=y)
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
plt.title('t-SNE Visualization (Iris Dataset)')
plt.show()