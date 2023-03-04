import os
import numpy as np
import torch
from data_io.kaldi_io import read_plda
from scipy.linalg import eigh
import h5py


l2_norm = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)


def transform_embeddings(X, embeddings_name, model_dir=""):

    if embeddings_name in ["clova", "speechbrain"]:
        X = l2_norm(X)

    elif embeddings_name == "brno":
        xvec_transform = os.path.join(model_dir, "ResNet101_16kHz/transform.h5")

        with h5py.File(xvec_transform, "r") as f:
            mean1 = np.array(f["mean1"])
            mean2 = np.array(f["mean2"])
            lda = np.array(f["lda"])
            X = l2_norm(np.dot(l2_norm(X - mean1), lda) - mean2)
    else:
        raise NotImplementedError

    return X


def prepare_plda_kaldi(embeddings_name, model_dir=""):

    if embeddings_name == "brno":

        plda_file = os.path.join(model_dir, "ResNet101_16kHz/plda")

        kaldi_plda = read_plda(plda_file)
        plda_mu, plda_tr, plda_psi = [
            kaldi_plda[key] for key in ["mean", "transform", "psi"]
        ]

        W = np.linalg.inv(plda_tr.T.dot(plda_tr))
        B = np.linalg.inv((plda_tr.T / plda_psi).dot(plda_tr))
        acvar, wccn = eigh(B, W)
        plda_mu = plda_mu.ravel()
        plda_psi = acvar[::-1]
        plda_tr = wccn.T[::-1]

    else:
        raise NotImplementedError

    return plda_mu, plda_tr, plda_psi


def score_psda_many2many(x_e, x_t, psda):

    if x_e.shape[0] > 1:
        x_e = torch.sum(x_e, dim=0, keepdim=True)
    if x_t.shape[0] > 1:
        x_t = torch.sum(x_t, dim=0, keepdim=True)

    E = psda.prep(x_e.numpy())
    T = psda.prep(x_t.numpy())
    return torch.tensor(E.llr_vector(T))
