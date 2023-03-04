import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.special

# See https://backend.orbit.dtu.dk/ws/files/134946549/neco_a_01000.pdf and https://arxiv.org/pdf/2203.14893.pdf

# Check https://github.com/nicola-decao/s-vae-pytorch/blob/master/hyperspherical_vae/ops/ive.py
# See also https://github.com/nicola-decao/s-vae-pytorch/blob/master/hyperspherical_vae/distributions/von_mises_fisher.py


def is_small(z, v):
    # return z < 1e-1 # TODO: should grow with v
    return z < 10 ** (np.log10(v) - 1)  # will not work for v > 200


def ive(z, v):
    mask_small = is_small(z, v)
    mask_large = np.logical_not(mask_small)
    out = np.zeros_like(z)
    # approximate by the 1st term
    log_ive = (
        -scipy.special.gammaln(v + 1) + v * np.log(z[mask_small] / 2) - z[mask_small]
    )
    out[mask_small] = np.exp(log_ive)
    out[mask_large] = scipy.special.ive(v, z[mask_large])
    return out


def log_ive(z, v):
    mask_small = is_small(z, v)
    mask_large = np.logical_not(mask_small)
    out = np.zeros_like(z)
    out[mask_small] = (
        scipy.special.gammaln(v + 1) + v * np.log(z[mask_small] / 2) - z[mask_small]
    )
    out[mask_large] = np.log(ive(z[mask_large], v) + 1e-300)
    return out


def iv(z, v):
    return ive(z, v) / np.exp(-z)


def rho(kappa, dim):
    dim_half = 0.5 * dim
    mask_small = (kappa > 1e-3) * (kappa < 10)
    mask_large = kappa >= 10
    out = np.ones_like(kappa)
    out[mask_large] = ive(kappa[mask_large], dim_half) / (
        ive(kappa[mask_large], dim_half - 1) + 1e-300
    )
    out[mask_small] = iv(kappa[mask_small], dim_half) / (
        iv(kappa[mask_small], dim_half - 1) + 1e-300
    )
    return out


def log_vmf_normalizer(kappa, dim):
    # = log C_d(k)
    dim_half = 0.5 * dim
    log_ive_kappa = log_ive(kappa, dim_half - 1)
    log_iv_kappa = log_ive_kappa + kappa
    normalizer = (
        -dim_half * np.log(2 * np.pi) - log_iv_kappa + (dim_half - 1) * np.log(kappa)
    )
    return normalizer



