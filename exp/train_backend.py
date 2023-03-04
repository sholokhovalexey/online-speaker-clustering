import os
import sys

import numpy as np
import torch
import torch.nn.functional as F


def train_psda_em(X, y):

    X_norm = F.normalize(X, dim=1)

    classes, yy = torch.unique(y, return_inverse=True)
    I = torch.eye(len(classes)).to(X.device)
    Y = I[yy]

    counts = torch.sum(Y, dim=0, keepdim=True)
    means = torch.mm(Y.t(), X_norm) / counts.t()

    psda, obj = PSDA.em(means.numpy(), counts.numpy().ravel(), niters=10, quiet=True)
    return psda


root_dir = os.path.abspath(os.getcwd())
pretrained_dir = f"{root_dir}/pretrained"
embeddings_path = f"{root_dir}/cache/voxceleb"
sys.path.append(f"{root_dir}/external/PSDA")


for EMBEDDINGS_NAME in ["speechbrain", "clova", "brno"]:

    embeddings_type = "2sec"

    print("Embeddings:", EMBEDDINGS_NAME)

    # "Load embeddings"
    data_train = np.load(
        f"{embeddings_path}/emb_vox1_train_{EMBEDDINGS_NAME}_{embeddings_type}.npz"
    )

    get_spk_id = lambda utt: utt.split("-")[0]

    X = data_train["X"]
    utt_ids = data_train["ids"]

    X = torch.tensor(X).float()

    # Labels
    speakers_train, y = np.unique([get_spk_id(utt) for utt in utt_ids], return_inverse=True)
    y = torch.tensor(y)

    print(f"Training set with {len(X)} embeddings and {len(speakers_train)} classes")

    # Pre-processing
    CENTERING = False

    if CENTERING:
        mu = torch.mean(X, dim=0, keepdim=True)
        X = X - mu
        torch.save({"mu": mu}, f"{pretrained_dir}/{EMBEDDINGS_NAME}/transform.pt")

    # Length normalization
    X = F.normalize(X, dim=1)

    # PSDA
    from psda.psdamodel import PSDA

    psda = train_psda_em(X, y)
    print(psda)
    psda.save(f"{pretrained_dir}/{EMBEDDINGS_NAME}/psda_vox_2sec")

    # sph-PLDA
    from backend.plda import SphPLDA

    plda = SphPLDA()
    plda.fit(X, y)
    print(plda)
    plda.save(f"{pretrained_dir}/{EMBEDDINGS_NAME}/plda-sph_vox_2sec")

