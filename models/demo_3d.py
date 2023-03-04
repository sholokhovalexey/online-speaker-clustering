import numpy as np
import matplotlib.pyplot as plt

import torch

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

from models.vonmises import OnlineRecognizerMisesFisher


# generate 3D data
seed = 4
np.random.seed(seed)

n_classes = 3

n_samples = 500
X_all, y_all = make_blobs(
    n_features=3,
    n_samples=n_samples,
    centers=n_classes,
    cluster_std=2,
    random_state=seed,
)
mean = np.mean(X_all, axis=0, keepdims=True)
X_all = X_all - mean
X_all = X_all / np.linalg.norm(X_all, axis=1, keepdims=True)

X_enroll, X, y_enroll, y = train_test_split(
    X_all, y_all, test_size=0.95, random_state=seed
)

# Run 3 models with different within-class dispersion parameters
fig, axes = plt.subplots(1, 3, subplot_kw=dict(projection="3d"), figsize=(20, 5))

mu0 = np.array([1.0, 0, 0.0])
params = [(0.001, 10), (0.001, 100), (0.001, 1000)]

for i, ax in enumerate(axes):

    b, w = params[i]

    model = OnlineRecognizerMisesFisher(b, w, mu0)

    T = lambda X: torch.tensor(X).float()

    precision = torch.ones(X.shape[0]) * 999999

    with torch.no_grad():
        preds, mu, kappa = model(T(X), T(X_enroll), T(y_enroll), precision, n_iters=2)
    preds, mu, kappa = preds.numpy(), mu.numpy(), kappa.numpy()

    y_pred = np.argmax(preds, axis=1)
    y_pred[y_pred == n_classes] = -1

    #
    xdata = X[y != y_pred, 0]
    ydata = X[y != y_pred, 1]
    zdata = X[y != y_pred, 2]
    ax.scatter3D(xdata, ydata, zdata, marker="s", color="k")

    xdata = X[y == y_pred, 0]
    ydata = X[y == y_pred, 1]
    zdata = X[y == y_pred, 2]
    ax.scatter3D(xdata, ydata, zdata, c=y[y == y_pred])

    xdata = X_enroll[:, 0]
    ydata = X_enroll[:, 1]
    zdata = X_enroll[:, 2]
    ax.scatter3D(xdata, ydata, zdata, marker="v", c=y_enroll, edgecolor="r")


plt.subplots_adjust(wspace=0.1, hspace=0)
plt.show()

