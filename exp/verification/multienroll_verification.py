import os
import sys
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from prettytable import PrettyTable

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.cosine import score_cosine_embeddings_averaging
from backend.cosine import score_cosine_scores_averaging
from backend import score_psda_many2many
from evaluation.metrics import calculate_eer, calculate_mindcf


torch.set_grad_enabled(False)


def read_lines_file(file_path, sep=" ", merge=False):
    output = []
    with open(file_path, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if sep is None:
                output += [line]
            else:
                if merge:
                    output.extend(line.split(sep))
                else:
                    output.append(line.split(sep))
    return output


root_dir = os.path.abspath(os.getcwd())
pretrained_dir = f"{root_dir}/pretrained"
embeddings_path = f"{root_dir}/cache/voxceleb"
sys.path.append(f"{root_dir}/external/PSDA")


for EMBEDDINGS_NAME in ["clova", "speechbrain", "brno"]:

    embeddings_type = "2sec"

    print("Embeddings:", EMBEDDINGS_NAME)

    # Load embeddings
    data_test = np.load(
        f"{embeddings_path}/emb_vox1_test_{EMBEDDINGS_NAME}_{embeddings_type}.npz"
    )
    get_spk_id = lambda utt: utt.split("-")[0]

    X = data_test["X"]
    utt_ids = data_test["ids"]
    utt2idx = {utt: idx for idx, utt in enumerate(utt_ids)}

    X = torch.tensor(X).float()

    # Load trials
    utts_enroll = read_lines_file(
        f"{root_dir}/data/meta/voxceleb/multienroll/trials_enroll.txt"
    )
    utts_test = read_lines_file(
        f"{root_dir}/data/meta/voxceleb/multienroll/trials_test.txt"
    )
    labels = np.loadtxt(
        f"{root_dir}/data/meta/voxceleb/multienroll/trials_label.txt", dtype=int
    )

    cfg2trials = defaultdict(list)
    for utts_e, utts_t, label in zip(utts_enroll, utts_test, labels):
        idx_enr = torch.tensor([utt2idx[u] for u in utts_e])
        idx_test = torch.tensor([utt2idx[u] for u in utts_t])
        # trials += [(idx_enr, idx_test)]

        cfg = (len(idx_enr), len(idx_test))
        cfg2trials[cfg].append((idx_enr, idx_test, label))

    # Pre-processing
    CENTERING = False

    if CENTERING:
        data = torch.load(f"{pretrained_dir}/{EMBEDDINGS_NAME}/transform.pt")
        mu = data["mu"]
        X = X - mu

    # Length normalization
    X = F.normalize(X, dim=1)

    # PSDA
    from psda.psdamodel import PSDA

    psda = PSDA.load(f"{pretrained_dir}/{EMBEDDINGS_NAME}/psda_vox_2sec")
    score_psda = lambda x1, x2: score_psda_many2many(x1, x2, psda)
    print(psda)

    # sph-PLDA
    from backend.plda import SphPLDA

    plda = SphPLDA.load(f"{pretrained_dir}/{EMBEDDINGS_NAME}/plda-sph_vox_2sec")
    score_plda = lambda x1, x2: plda.score(x1, x2)
    print(plda)

    # Compute scores
    results_table = PrettyTable()
    results_table.title = EMBEDDINGS_NAME
    column_names = [f"({ne}, {nt})" for (ne, nt) in cfg2trials] + ["pooled"]
    row_names = ["CSEA", "CSSA", "sph-PLDA", "PSDA"]
    results_table.field_names = ["Scoring"] + column_names

    score_types = ["cos_emb_avg", "cos_sc_avg", "plda_sph", "psda"]
    # colors = ['b', 'r', 'g', 'm']

    idx_subset = [0, 1, 2, 3]
    row_names = [row_names[idx] for idx in idx_subset]
    score_types = [score_types[idx] for idx in idx_subset]
    # colors = [colors[idx] for idx in idx_subset]

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    for idx, score_type in enumerate(score_types):

        if score_type == "cos_emb_avg":
            similarity_score = score_cosine_embeddings_averaging
        elif score_type == "cos_sc_avg":
            similarity_score = score_cosine_scores_averaging
        elif score_type == "plda_sph":
            similarity_score = score_plda
        elif score_type == "psda":
            similarity_score = score_psda

        scores_pooled = []
        labels_pooled = []
        eers = []
        dcfs = []
        for (n_enrolls, n_tests), trials in cfg2trials.items():
            scores_part = []
            labels_part = []
            for (idx_enr, idx_test, label) in tqdm(trials):

                x_enr = X[idx_enr]
                x_test = X[idx_test]

                score = similarity_score(x_enr, x_test)
                scores_part += [score]
                labels_part += [label]

            scores_part = torch.tensor(scores_part).cpu().numpy()
            labels_part = np.array(labels_part)

            scores_pooled += [scores_part]
            labels_pooled += [labels_part]

            EER, _ = calculate_eer(scores_part, labels_part)
            EER = EER * 100
            # print(f"\t\tEER, {score_type}, ({n_enrolls}, {n_tests}): {EER:.2f} %")
            eers += [EER]

            minDCF, threshold = calculate_mindcf(scores_part, labels_part)
            dcfs += [minDCF]

        scores_pooled = np.concatenate(scores_pooled)
        labels_pooled = np.concatenate(labels_pooled)

        EER, threshold = calculate_eer(scores_pooled, labels_pooled)
        EER = EER * 100
        # print(f"\tEER, {score_type}, pooled: {EER:.2f} %")
        eers += [EER]

        minDCF, threshold = calculate_mindcf(scores_pooled, labels_pooled)
        dcfs += [minDCF]

        results_table.add_row(
            [row_names[idx]] + [f"{e:.2f} / {d:.3f}" for e, d in zip(eers, dcfs)]
        )

    print(results_table)

    # ax.legend(loc="upper right")
    # plt.grid(True)
    # plt.show()


# +-------------------------------------------------------------------------------------+
# |                                        clova                                        |
# +----------+--------------+--------------+--------------+--------------+--------------+
# | Scoring  |    (1, 1)    |    (3, 1)    |   (10, 1)    |    (3, 3)    |    pooled    |
# +----------+--------------+--------------+--------------+--------------+--------------+
# |   CSEA   | 4.24 / 0.464 | 1.60 / 0.204 | 0.86 / 0.126 | 0.28 / 0.051 | 3.06 / 0.304 |
# |   CSSA   | 4.24 / 0.464 | 1.77 / 0.279 | 1.08 / 0.202 | 0.60 / 0.154 | 1.95 / 0.307 |
# | sph-PLDA | 4.24 / 0.464 | 1.54 / 0.192 | 0.78 / 0.113 | 0.22 / 0.043 | 1.69 / 0.217 |
# |   PSDA   | 4.24 / 0.462 | 1.58 / 0.197 | 0.84 / 0.119 | 0.23 / 0.044 | 1.72 / 0.227 |
# +----------+--------------+--------------+--------------+--------------+--------------+

# +-------------------------------------------------------------------------------------+
# |                                     speechbrain                                     |
# +----------+--------------+--------------+--------------+--------------+--------------+
# | Scoring  |    (1, 1)    |    (3, 1)    |   (10, 1)    |    (3, 3)    |    pooled    |
# +----------+--------------+--------------+--------------+--------------+--------------+
# |   CSEA   | 4.72 / 0.402 | 1.54 / 0.151 | 0.79 / 0.074 | 0.15 / 0.018 | 2.70 / 0.200 |
# |   CSSA   | 4.72 / 0.402 | 1.63 / 0.179 | 0.93 / 0.098 | 0.30 / 0.036 | 1.92 / 0.220 |
# | sph-PLDA | 4.72 / 0.402 | 1.52 / 0.147 | 0.76 / 0.073 | 0.13 / 0.017 | 1.90 / 0.166 |
# |   PSDA   | 4.73 / 0.402 | 1.49 / 0.146 | 0.78 / 0.074 | 0.14 / 0.017 | 2.03 / 0.171 |
# +----------+--------------+--------------+--------------+--------------+--------------+

# +-------------------------------------------------------------------------------------+
# |                                         brno                                        |
# +----------+--------------+--------------+--------------+--------------+--------------+
# | Scoring  |    (1, 1)    |    (3, 1)    |   (10, 1)    |    (3, 3)    |    pooled    |
# +----------+--------------+--------------+--------------+--------------+--------------+
# |   CSEA   | 3.45 / 0.336 | 1.04 / 0.111 | 0.57 / 0.049 | 0.06 / 0.004 | 2.08 / 0.166 |
# |   CSSA   | 3.45 / 0.336 | 1.16 / 0.127 | 0.62 / 0.062 | 0.12 / 0.013 | 1.38 / 0.172 |
# | sph-PLDA | 3.45 / 0.336 | 1.03 / 0.108 | 0.54 / 0.049 | 0.05 / 0.004 | 1.30 / 0.127 |
# |   PSDA   | 3.47 / 0.337 | 1.02 / 0.109 | 0.58 / 0.051 | 0.04 / 0.004 | 1.32 / 0.128 |
# +----------+--------------+--------------+--------------+--------------+--------------+

