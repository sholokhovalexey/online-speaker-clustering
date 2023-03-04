import torch


def real2pos(x):
    return torch.exp(x)


def pos2real(x):
    return torch.log(x + 1e-16)
