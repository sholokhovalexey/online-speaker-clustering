import os
import numpy as np
import torch
import yaml
import argparse

from embeddings import prepare_model
from embeddings.extraction import extract_embeddings_path

torch.set_grad_enabled(False)


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

root_dir = os.path.abspath(os.getcwd())
config_common = yaml.load(
    open(f"{root_dir}/config_common.yaml"), Loader=yaml.FullLoader
)

data_root_vox1_dev = config_common["datasets"]["VoxCeleb1-dev"]
data_root_vox1_test = config_common["datasets"]["VoxCeleb1-test"]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--emb", type=str, required=True, choices=["speechbrain", "clova", "brno"]
)
parser.add_argument("--save_dir", type=str)
parser.add_argument("--batch_size", type=int, default=128)
args = parser.parse_args()

EMBEDDINGS_NAME = args.emb
save_location = args.save_dir
save_location = save_location if save_location else f"{root_dir}/cache/voxceleb"
os.makedirs(save_location, exist_ok=True)

print(f"Embeddings: {EMBEDDINGS_NAME}")
print(f"Device: {DEVICE}")

emb_model = prepare_model(EMBEDDINGS_NAME, device=DEVICE)


# VoxCeleb1
batch_size = args.batch_size
crop_size = 32000
crop_center = True  # if True crop_size must be a number
file2utt_vox_fn = lambda wav: f"ID{wav[2:-4].replace('/', '-')}"
path2utt_vox_fn = lambda wav_path: file2utt_vox_fn("/".join(wav_path.split("/")[-3:]))
# utt2spk_vox_fn = lambda utt: utt.split("/")[0]


# VoxCeleb1 test set
print("VoxCeleb1 test set")
embeddings, utt_ids = extract_embeddings_path(
    emb_model,
    data_root_vox1_test,
    DEVICE,
    path2utt_vox_fn,
    batch_size=1,
    crop_size=None,
)

# labels = [utt2spk_vox_fn(utt) for utt in utt_ids]
print(len(embeddings), len(utt_ids))
np.savez(f"{save_location}/emb_vox1_test_{EMBEDDINGS_NAME}", X=embeddings, ids=utt_ids)

# VoxCeleb1 test set, 2 seconds crops
print("VoxCeleb1 test set, 2 seconds crops")
embeddings, utt_ids = extract_embeddings_path(
    emb_model,
    data_root_vox1_test,
    DEVICE,
    path2utt_vox_fn,
    batch_size=batch_size,
    crop_size=crop_size,
    crop_center=crop_center,
)

# labels = [utt2spk_vox_fn(utt) for utt in utt_ids]
print(len(embeddings), len(utt_ids))
np.savez(
    f"{save_location}/emb_vox1_test_{EMBEDDINGS_NAME}_2sec", X=embeddings, ids=utt_ids
)

# VoxCeleb1 dev set, 2 seconds crops
print("VoxCeleb1 dev set, 2 seconds crops")
embeddings, utt_ids = extract_embeddings_path(
    emb_model,
    data_root_vox1_dev,
    DEVICE,
    path2utt_vox_fn,
    batch_size=batch_size,
    crop_size=crop_size,
    crop_center=crop_center,
)

# labels = [utt2spk_vox_fn(utt) for utt in utt_ids]
print(len(embeddings), len(utt_ids))
np.savez(
    f"{save_location}/emb_vox1_train_{EMBEDDINGS_NAME}_2sec", X=embeddings, ids=utt_ids
)

