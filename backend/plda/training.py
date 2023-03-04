import torch
import torch.nn.functional as F


def train_plda_sph_em(X, y, n_iter=10):
    """
    Train a sph-PLDA model via EM algorithm.

    Args:
        X (torch.Tensor): (N, d) a matrix of N d-dimensional observations.
        y (torch.Tensor): (N,) a vector of integer class labels.
        n_iter (int, optional): number of iterations of the EM algorithm. Defaults to 10.

    Returns:
        float, float: a pair of scalars, the between-class and the within-class variances.
    """
    device = X.device

    X_norm = F.normalize(X, dim=1)

    classes, yy = torch.unique(y, return_inverse=True)
    I = torch.eye(len(classes)).to(device)
    Y = I[yy]

    N, dim = X.shape

    # initialization
    sums = torch.mm(Y.t(), X_norm)
    counts = torch.sum(Y, dim=0)
    mu = sums / counts.view(-1, 1)
    Xc = X_norm - mu[yy]
    w = torch.mean(Xc ** 2)
    b = torch.mean(mu ** 2)

    for _ in range(n_iter):
        # E-step
        Sigma = 1 / (1 / b + counts * 1 / w)  # (n_classes,)
        mu = Sigma.view(-1, 1) * (1 / w * sums)  # (n_classes, dim)

        # M-step
        b = Sigma.mean() + (mu ** 2).mean()
        Xc = X_norm - torch.mm(Y, mu)
        w = (Xc ** 2).mean() + torch.sum(counts * Sigma) / N

    return b, w
