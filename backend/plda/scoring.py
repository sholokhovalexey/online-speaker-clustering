import math
import torch
import torch.nn.functional as F


def get_centroid(X):
    centroid = torch.mean(X, dim=0, keepdim=True)
    count = torch.tensor(X.shape[0])
    return (centroid, count)


def score_plda_sph_centroids(data_enroll, data_test, b, w, all_pairs=True):
    """
    Compute the log-likelihood ratio (LLR) score for a spherical PLDA model.

    Args:
        data_enroll (tuple or torch.Tensor): a pair of tensors (centroids, counts) of shapes (N, d) and (N,), 
        or a single tensor of shape (N, d) representing centroids, assuming counts=torch.ones(N).
        data_test (tuple or torch.Tensor): a pair of tensors (centroids, counts) of shapes (M, d) and (M,), 
        or a single tensor of shape (M, d) representing centroids, assuming counts=torch.ones(M).
        b (torch.Tensor): scalar between-class variance.
        w (torch.Tensor): scalar within-class variance.
        all_pairs (bool, optional): if true, computes NxM scores for all possible pairs between N enrollment trial sides 
        and M test trial sides. If false, computes N=M scores between the corresponding trial sides. Defaults to True.

    Returns:
        torch.Tensor: a tensor of similarity scores of shape (N, 1) if all_pairs=False or (N, M) if all_pairs=True.
    """

    b_inv = 1 / b
    w_inv = 1 / w

    centroids_e, n_e = data_enroll

    if isinstance(data_enroll, (tuple, list)):
        centroids_e, n_e = data_enroll
    else:
        centroids_e = data_enroll
        n_e = 1

    if isinstance(data_test, (tuple, list)):
        centroids_t, n_t = data_test
    else:
        centroids_t = data_test
        n_t = 1

    if not all_pairs:
        assert (
            centroids_e.shape[0] == centroids_t.shape[0]
        ), "Number of enrollments and tests must be the same if all_pairs=False"

    dim = centroids_e.shape[1]

    if isinstance(n_e, int):
        n_e = torch.tensor(n_e)
    n_e = n_e.view(-1, 1)
    if isinstance(n_t, int):
        n_t = torch.tensor(n_t)
    n_t = n_t.view(-1, 1)

    a_e = n_e * centroids_e * w_inv
    a_t = n_t * centroids_t * w_inv

    if all_pairs:
        n_t = n_t.t()

    sigma_e_inv = b_inv + n_e * w_inv
    sigma_t_inv = b_inv + n_t * w_inv
    sigma_e = 1 / sigma_e_inv
    sigma_t = 1 / sigma_t_inv
    sigma_inv_sum = sigma_e_inv + sigma_t_inv - b_inv

    const = dim * (
        0.5 * torch.log(b)
        + 0.5 * torch.log(sigma_e_inv)
        + 0.5 * torch.log(sigma_t_inv)
        - 0.5 * torch.log(sigma_inv_sum)
    )

    a_e_sqr = torch.sum(a_e ** 2, dim=1, keepdim=True)
    a_t_sqr = torch.sum(a_t ** 2, dim=1, keepdim=True)
    if all_pairs:
        a_t_sqr = a_t_sqr.t()
    mu_quad_term = -0.5 * (a_e_sqr * sigma_e + a_t_sqr * sigma_t)
    a_sqr = a_e_sqr + a_t_sqr
    if all_pairs:
        a_sqr += 2 * a_e @ a_t.t()
    else:
        a_sqr += 2 * torch.sum(a_e * a_t, dim=1, keepdim=True)
    return 0.5 / sigma_inv_sum * a_sqr + mu_quad_term + const

