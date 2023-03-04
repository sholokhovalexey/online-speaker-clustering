import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F

from embeddings import prepare_model
from embeddings.extraction import extract_embeddings_path
import backend as backend
import evaluation.metrics as metrics

torch.set_grad_enabled(False)


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

root_dir = os.path.abspath(os.getcwd())
config_common = yaml.load(
    open(f"{root_dir}/config_common.yaml"), Loader=yaml.FullLoader
)
data_root_vox1_test = config_common["datasets"]["VoxCeleb1-test"]
cache_location = f"{root_dir}/cache/voxceleb"
os.makedirs(cache_location, exist_ok=True)

# https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt
trials_file = f"{root_dir}/data/meta/voxceleb/veri_test2.txt"


for EMBEDDINGS_NAME in ["speechbrain", "clova", "brno"]:

    print(f"Embeddings: {EMBEDDINGS_NAME}")

    # Extract embeddings from speech
    cache_emb_file = f"{cache_location}/emb_vox1_test_{EMBEDDINGS_NAME}.npz"

    batch_size = 1
    crop_size = None
    crop_center = True  # if True crop_size must be a number
    path2utt_vox_fn = lambda wav_path: file2utt_vox_fn("/".join(wav_path.split("/")[-3:]))
    utt2spk_vox_fn = lambda utt: utt.split("/")[0]
    file2utt_vox_fn = lambda wav: f"ID{wav[2:-4].replace('/', '-')}"

    if not os.path.exists(cache_emb_file):

        emb_model = prepare_model(EMBEDDINGS_NAME, device=DEVICE)

        embeddings, utt_ids = extract_embeddings_path(
            emb_model,
            data_root_vox1_test,
            DEVICE,
            path2utt_vox_fn,
            batch_size=batch_size,
            crop_size=crop_size,
            crop_center=crop_center,
        )

        utt_ids_new = []
        for utt in utt_ids:
            utt_new = file2utt_vox_fn(utt)
            utt_ids_new += [utt_new]

        np.savez(cache_emb_file, X=embeddings, ids=utt_ids_new)

    # Load the VoxCeleb1 protocol
    get_spk = lambda utt_id: utt_id.split("-")[0]
    data = pd.read_csv(trials_file, sep=" ", header=None)
    labels_trials = data[0].to_numpy()

    trials = []
    for utt1, utt2 in zip(data[1], data[2]):
        pair = (file2utt_vox_fn(utt1), file2utt_vox_fn(utt2))
        trials += [pair]

    # Load and preprocess embeddings
    data = np.load(cache_emb_file)
    X_test = data["X"]
    ids_test = data["ids"]
    utt2emb_test = {utt: emb for (utt, emb) in zip(ids_test, X_test)}

    similarity_score = backend.cosine_similarity

    # Compute scores
    scores = []
    mask = []
    for u1, u2 in tqdm(trials):
        e1 = utt2emb_test[u1].reshape(1, -1)  # + 1e-6
        e2 = utt2emb_test[u2].reshape(1, -1)  # + 1e-6
        e1 = torch.tensor(e1)
        e2 = torch.tensor(e2)

        score = similarity_score(e1, e2).numpy()

        if np.isnan(score):
            print("NaN score (will be ignored):", u1, u2)
            mask += [0]
        else:
            mask += [1]
        scores += [score]
    scores = np.array(scores).ravel()
    mask = np.array(mask) > 0.5

    # Compute equal error rate (EER)
    eer, _ = metrics.calculate_eer(scores[mask], labels_trials[mask], pos_label=1)
    print(f"EER: {eer*100:.2f}%",)

    # "brno": 0.65%
    # "speechbrain": 0.90%
    # "clova": 1.19%
