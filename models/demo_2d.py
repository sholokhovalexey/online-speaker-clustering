
import numpy as np
import matplotlib.pyplot as plt

import torch

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from models.gaussian import OnlineRecognizerGaussian


# generate 2D data
seed = 0
np.random.seed(seed)

n_classes = 3

n_samples = 200
X_all, y_all = make_blobs(
    n_features=2,
    n_samples=n_samples,
    centers=n_classes,
    cluster_std=1,
    random_state=seed,
)

X_enroll, X, y_enroll, y = train_test_split(
    X_all, y_all, test_size=0.95, random_state=seed
)

# Run 3 models with different within-class dispersion parameters
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

params = [(100, 0.01), (100, 0.1), (100, 1)]

for i, ax in enumerate(axes):

    b, w = params[i]

    model = OnlineRecognizerGaussian(b, w, dim=2)

    T = lambda X: torch.tensor(X).float()

    sigma_noise = torch.zeros(X.shape[0])

    with torch.no_grad():
        preds, mu, Sigma = model(T(X), T(X_enroll), T(y_enroll), sigma_noise, n_iters=2)
    preds, mu, Sigma = preds.numpy(), mu.numpy(), Sigma.numpy()

    y_pred = np.argmax(preds, axis=1)
    y_pred[y_pred == n_classes] = -1

    ax.scatter(X[:, 0], X[:, 1], c=y)
    ax.scatter(X[y != y_pred, 0], X[y != y_pred, 1], marker="s", color="k")
    ax.scatter(X_enroll[:, 0], X_enroll[:, 1], marker="v", c=y_enroll, edgecolor="r")
    ax.scatter(mu[:, 0], mu[:, 1], marker="o", color="g", s=100, edgecolor="k")


plt.subplots_adjust(wspace=0.1, hspace=0)
plt.show()

