import os
import glob2
import yaml
from pyannote.database.util import load_rttm


class Dataset:
    def __init__(self, name, uri2path, annotations={}):
        self.name = name
        self.uri2path = uri2path
        self.annotations = annotations

    def path(self, uri):
        return self.uri2path[uri]

    def labels(self, label_type):
        return self.annotations[label_type]


# TODO: follow the original directory structure
def make_dataset(dataset_name):

    config_common = yaml.load(open("config_common.yaml"), Loader=yaml.FullLoader)

    if dataset_name == "aishell4":

        data_root = config_common["datasets"]["AISHELL-4"]
        audio_dir = f"{data_root}/test/wav"
        rttm_dir = f"data/meta/AISHELL-4/test/rttm"

        wav_list = glob2.glob(f"{audio_dir}/*.flac")
        uri2path = {os.path.splitext(os.path.basename(wav))[0]: wav for wav in wav_list}

        uri2ann_ref = {}
        rttm_list = glob2.glob(f"{rttm_dir}/*.rttm")
        for rttm_path in rttm_list:
            uri2ann_ref.update(load_rttm(rttm_path))

    elif dataset_name in ["AMI_test", "AMI_dev"]:

        data_split = dataset_name.split("_")[1]
        data_root = config_common["datasets"]["AMI"]
        rttm_dir = f"data/meta/AMI-diarization-setup/only_words/rttms/{data_split}"

        uri2ann_ref = {}
        rttm_list = glob2.glob(f"{rttm_dir}/*.rttm")
        for rttm_path in rttm_list:
            uri2ann_ref.update(load_rttm(rttm_path))

        uri2path = {}
        for uri in uri2ann_ref:
            uri2path[uri] = f"{data_root}/{uri}/audio/{uri}.Mix-Headset.wav"

    elif dataset_name in ["voxconverse_test", "voxconverse_dev"]:

        data_split = dataset_name.split("_")[1]
        data_root = config_common["datasets"]["VoxConverse"]
        audio_dir = f"{data_root}/{data_split}/audio"
        rttm_dir = f"data/meta/voxconverse/{data_split}"

        wav_list = glob2.glob(f"{audio_dir}/*.wav")
        uri2path = {os.path.splitext(os.path.basename(wav))[0]: wav for wav in wav_list}

        uri2ann_ref = {}
        rttm_list = glob2.glob(f"{rttm_dir}/*.rttm")
        for rttm_path in rttm_list:
            uri2ann_ref.update(load_rttm(rttm_path))

    elif dataset_name in ["Eval_ali_far", "Test_ali_far"]:

        data_split = dataset_name.split("_")[0]
        data_root = config_common["datasets"]["AliMeeting"]
        audio_dir = f"{data_root}/{data_split}_Ali/{data_split}_Ali_far/audio_dir"
        rttm_dir = f"data/meta/AliMeeting/{data_split}_Ali_far/rttm"

        wav_list = glob2.glob(f"{audio_dir}/*.wav")
        uri2path = {}
        for wav in wav_list:
            file_id = os.path.splitext(os.path.basename(wav))[0]
            parts = file_id.split("_")
            uri = "_".join(parts[:2])
            uri2path[uri] = wav

        uri2ann_ref = {}
        rttm_list = glob2.glob(f"{rttm_dir}/*.rttm")
        for rttm_path in rttm_list:
            uri2ann_ref.update(load_rttm(rttm_path))

    else:
        print(f"Dataset '{dataset_name}' not found")
        return None

    # TODO:
    if len(uri2path) == 0:
        print("No audio data found, cache will be used")
        for uri in uri2ann_ref:
            uri2path[uri] = "Path does not exist"

    return Dataset(dataset_name, uri2path, {"speakers": uri2ann_ref})

