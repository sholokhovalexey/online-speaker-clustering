import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def mahalanobis_distance(X, Y, M):
    quad_term1 = torch.sum((X @ M) * X, dim=1, keepdim=True)
    quad_term2 = torch.sum((Y @ M) * Y, dim=1, keepdim=True).t()
    cross_term = 2 * (X @ M) @ Y.t()
    dist = quad_term1 + quad_term2 - cross_term
    return dist


def euclidean_distance(X, Y):
    quad_term1 = torch.sum(X * X, dim=-1, keepdim=True)
    quad_term2 = torch.sum(Y * Y, dim=-1, keepdim=True).transpose(-2, -1)
    cross_term = 2 * torch.matmul(X, Y.transpose(-2, -1))
    dist = quad_term1 + quad_term2 - cross_term
    return dist


def logmvnpdf(X, mu, Sigma):
    dim = X.shape[1]
    var = Sigma.view(1, -1)
    const = -0.5 * dim * math.log(2 * math.pi) - 0.5 * dim * torch.log(var)
    exp_term = (
        -0.5
        / var
        * (
            torch.sum(X ** 2, dim=1, keepdim=True)
            + torch.sum(mu ** 2, dim=1, keepdim=True).t()
            - 2 * torch.mm(X, mu.t())
        )
    )
    return const + exp_term


def logmvnpdf_noisy(X, mu, Sigma, variance_noise):
    dim = X.shape[1]
    var = Sigma.view(1, -1) + variance_noise.view(-1, 1)
    const = -0.5 * dim * math.log(2 * math.pi) - 0.5 * dim * torch.log(var)
    exp_term = (
        -0.5
        / var
        * (
            torch.sum(X ** 2, dim=1, keepdim=True)
            + torch.sum(mu ** 2, dim=1, keepdim=True).t()
            - 2 * torch.mm(X, mu.t())
        )
    )
    return const + exp_term



