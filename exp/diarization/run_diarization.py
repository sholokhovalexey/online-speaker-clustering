import os
import sys
from tqdm import tqdm
import pickle
import argparse
import shutil

import numpy as np
from prettytable import PrettyTable

import torch
import torch.nn.functional as F

from clustering import (
    OnlineClusteringCentroids,
    OnlineClusteringMemory,
    OnlineClusteringGaussian,
    OnlineClusteringMises,
)

from pyannote.core import Annotation
from pyannote.audio import Model
from pyannote.audio.pipelines import (
    VoiceActivityDetection,
    OverlappedSpeechDetection,
)
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

from embeddings import prepare_model
from embeddings.extraction import extract_embeddings_wav
from utils.segmentation import split_segments, split_overlap_part

from data_io import load_audio
from data.datasets import make_dataset
import backend

from parameters_AMI_dev import get_default_params as get_default_params_AMI
from parameters_voxconverse_dev import (
    get_default_params as get_default_params_voxconverse,
)


torch.set_grad_enabled(False)


root_dir = os.path.abspath(os.getcwd())
pretrained_dir = f"{root_dir}/pretrained"
cache_location = f"{root_dir}/cache/diarization"
sys.path.append(f"{root_dir}/external/PSDA")
sys.path.append(f"{root_dir}/external/dscore")

from external.dscore.scorelib.score import score
from external.dscore.scorelib.turn import merge_turns, trim_turns
from external.dscore.scorelib.uem import gen_uem
from external.dscore.score import load_rttms


min_duration = 0.5  # sec


def prepare_embeddings(
    embeddings_name,
    dataset_name,
    clustering,
    win,
    hop,
    ignore_overlap=False,
    vad_oracle=True,
    device=0,
):

    np.random.seed(0)

    EMBEDDINGS_NAME = embeddings_name
    CLUSTERING = clustering
    DATASET_NAME = dataset_name
    WIN_SIZE = win
    STEP_SIZE = hop
    IGNORE_OVERLAP = ignore_overlap
    VAD_ORACLE = vad_oracle
    DEVICE = f"cuda:{device}"

    print(DATASET_NAME, EMBEDDINGS_NAME, CLUSTERING)
    dataset = make_dataset(DATASET_NAME)

    assert len(dataset.labels("speakers")) > 0, "No reference annotation exists"

    if not VAD_ORACLE:

        # Parameters: https://huggingface.co/pyannote/segmentation#reproducible-research

        HYPER_PARAMETERS = {
            # onset/offset activation thresholds
            "onset": 0.5,
            "offset": 0.5,
            # remove speech regions shorter than that many seconds.
            "min_duration_on": 0.1,
            # fill non-speech regions shorter than that many seconds.
            "min_duration_off": 0.1,
        }

        vad_osd_joint = Model.from_pretrained(
            f"{root_dir}/pretrained/frontend/pytorch_model.bin"
        )

        vad_model = VoiceActivityDetection(segmentation=vad_osd_joint)
        vad_model.instantiate(HYPER_PARAMETERS)

        osd_model = OverlappedSpeechDetection(segmentation=vad_osd_joint)
        osd_model.instantiate(HYPER_PARAMETERS)

    # Voice activity and overlapped speech detection

    uri2vad = {}
    uri2osd = {}
    for uri, wav_path in dataset.uri2path.items():

        if VAD_ORACLE:
            vad = dataset.labels("speakers")[uri].get_timeline().support()
            osd = vad.get_overlap()
        else:
            ann_vad = vad_model(wav_path)
            vad = ann_vad.get_timeline().support()
            ann_osd = osd_model(wav_path)
            osd = ann_osd.get_timeline().support()

        uri2vad[uri] = vad
        uri2osd[uri] = osd
        # TODO: cache annotations

    # Embeddings extraction

    cache_id = f"{DATASET_NAME}_emb-{EMBEDDINGS_NAME}_vad-{VAD_ORACLE}_win-{WIN_SIZE:.1f}_step-{STEP_SIZE:.1f}_no-ovlp-{IGNORE_OVERLAP}"
    cache_path = f"{cache_location}/{cache_id}.p"

    try:
        print(cache_path)
        with open(cache_path, "rb") as f:
            uri2data = pickle.load(f)
    except:

        emb_model = prepare_model(EMBEDDINGS_NAME, device=DEVICE)

        print(f"Not found in cache: {cache_path}")

        uri2data = {}
        for uri, wav_path in tqdm(list(dataset.uri2path.items())):

            vad_timeline = uri2vad[uri]

            if IGNORE_OVERLAP:
                # exclude segments with overlapped speech
                osd_timeline = uri2osd[uri]
                vad_timeline = vad_timeline.extrude(osd_timeline).support()

            waveform, _ = load_audio(wav_path)

            segments = split_segments(
                vad_timeline, WIN_SIZE, STEP_SIZE, min_duration=min_duration
            )
            embeddings = extract_embeddings_wav(
                emb_model, waveform, DEVICE, segments, batch_size=64
            )

            uri2data[uri] = (embeddings, segments)

        os.makedirs(cache_location, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(uri2data, f)

    return {"uri2data": uri2data, "uri2ann_ref": dataset.labels("speakers")}


def run_clustering(
    hyperparams,
    EMBEDDINGS_NAME,
    DATASET_NAME,
    CLUSTERING,
    uri2data,
    uri2ann_ref,
    versbose=True,
):

    uri2ann_hyp = {}
    for uri in tqdm(uri2data, disable=not versbose):

        embeddings, segments = uri2data[uri]

        # idx_subset = np.array([i for i, seg in enumerate(segments) if seg.end - seg.start > min_duration])
        # embeddings = embeddings[idx_subset]
        # segments = [segments[i] for i in idx_subset]

        CENTERING = False

        if CENTERING:
            data = torch.load(f"{pretrained_dir}/{EMBEDDINGS_NAME}/transform.pt")
            mu = data["mu"].numpy()
            embeddings = embeddings - mu

        l2_norm = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)
        embeddings = l2_norm(embeddings)

        embeddings = embeddings.astype(np.float32)

        dim = embeddings.shape[-1]

        if CLUSTERING == "online_csea":

            similarity_score = backend.cosine_similarity

            if DATASET_NAME.startswith("AMI"):
                params = get_default_params_AMI(EMBEDDINGS_NAME, "online_csea")
            elif DATASET_NAME.startswith("voxconverse"):
                params = get_default_params_voxconverse(EMBEDDINGS_NAME, "online_csea")
            for param_name in hyperparams:
                params[param_name] = hyperparams[param_name]
            threshold = params["threshold"]

            online_clustering = OnlineClusteringCentroids(similarity_score, threshold)
            preds = online_clustering(torch.tensor(embeddings))
            labels = np.unique(preds, return_inverse=True)[1]

        elif CLUSTERING == "online_cssa":

            similarity_score = backend.cosine_similarity

            if DATASET_NAME.startswith("AMI"):
                params = get_default_params_AMI(EMBEDDINGS_NAME, "online_cssa")
            elif DATASET_NAME.startswith("voxconverse"):
                params = get_default_params_voxconverse(EMBEDDINGS_NAME, "online_cssa")
            for param_name in hyperparams:
                params[param_name] = hyperparams[param_name]
            threshold = params["threshold"]

            online_clustering = OnlineClusteringMemory(similarity_score, threshold)
            preds = online_clustering(torch.tensor(embeddings))
            labels = np.unique(preds, return_inverse=True)[1]

        elif CLUSTERING == "online_plda":

            from backend.plda import SphPLDA

            plda = SphPLDA.load(f"{pretrained_dir}/{EMBEDDINGS_NAME}/plda-sph_vox_2sec")

            b = plda.b
            w = plda.w

            if DATASET_NAME.startswith("AMI"):
                params = get_default_params_AMI(EMBEDDINGS_NAME, "online_plda")
            elif DATASET_NAME.startswith("voxconverse"):
                params = get_default_params_voxconverse(EMBEDDINGS_NAME, "online_plda")
            for param_name in hyperparams:
                params[param_name] = hyperparams[param_name]
            threshold = params["threshold"]

            similarity_score = lambda x, y: plda.score(x, y)
            online_clustering = OnlineClusteringMemory(
                similarity_score, threshold, average_scores=False
            )

            preds = online_clustering(torch.tensor(embeddings))
            labels = np.unique(preds, return_inverse=True)[1]

        elif CLUSTERING == "online_psda":

            # PSDA
            from psda.psdamodel import PSDA

            psda = PSDA.load(f"{pretrained_dir}/{EMBEDDINGS_NAME}/psda_vox_2sec")

            b = psda.b
            w = psda.w
            mu = psda.mu

            if DATASET_NAME.startswith("AMI"):
                params = get_default_params_AMI(EMBEDDINGS_NAME, "online_psda")
            elif DATASET_NAME.startswith("voxconverse"):
                params = get_default_params_voxconverse(EMBEDDINGS_NAME, "online_psda")
            for param_name in hyperparams:
                params[param_name] = hyperparams[param_name]
            threshold = params["threshold"]

            similarity_score = lambda x1, x2: backend.score_psda_many2many(x1, x2, psda)
            online_clustering = OnlineClusteringMemory(
                similarity_score, threshold, average_scores=False
            )

            preds = online_clustering(torch.tensor(embeddings))
            labels = np.unique(preds, return_inverse=True)[1]

        elif CLUSTERING == "online_vb_plda":

            from backend.plda import SphPLDA

            plda = SphPLDA.load(f"{pretrained_dir}/{EMBEDDINGS_NAME}/plda-sph_vox_2sec")

            b = plda.b
            w = plda.w

            if DATASET_NAME.startswith("AMI"):
                params = get_default_params_AMI(EMBEDDINGS_NAME, "online_vb_plda")
            elif DATASET_NAME.startswith("voxconverse"):
                params = get_default_params_voxconverse(EMBEDDINGS_NAME, "online_vb_plda")
            for param_name in hyperparams:
                params[param_name] = hyperparams[param_name]
            threshold = params["threshold"]

            online_clustering = OnlineClusteringGaussian(b, w, dim, threshold)
            preds = online_clustering(torch.tensor(embeddings))
            labels = np.unique(preds, return_inverse=True)[1]

        elif CLUSTERING == "online_vb_psda":

            # PSDA
            from psda.psdamodel import PSDA

            psda = PSDA.load(f"{pretrained_dir}/{EMBEDDINGS_NAME}/psda_vox_2sec")

            b = psda.b
            w = psda.w
            mu = psda.mu

            if DATASET_NAME.startswith("AMI"):
                params = get_default_params_AMI(EMBEDDINGS_NAME, "online_vb_psda")
            elif DATASET_NAME.startswith("voxconverse"):
                params = get_default_params_voxconverse(EMBEDDINGS_NAME, "online_vb_psda")
            for param_name in hyperparams:
                params[param_name] = hyperparams[param_name]
            threshold = params["threshold"]

            online_clustering = OnlineClusteringMises(b, w, mu, threshold)
            preds = online_clustering(torch.tensor(embeddings))
            labels = np.unique(preds, return_inverse=True)[1]

        else:
            print(f"Algorithm {CLUSTERING} was not found!")
            exit()

        # print(sorted(np.unique(labels, return_counts=True)[1]))

        ann_hyp = Annotation(uri=uri)
        label_previous = labels[0]
        for segment, label in zip(segments, labels):
            if segment.end - segment.start > min_duration:
                ann_hyp[segment, "_"] = str(label)
            else:
                ann_hyp[segment, "_"] = str(label_previous)
            label_previous = label

        ann_hyp = split_overlap_part(ann_hyp.support())

        uri2ann_hyp[uri] = ann_hyp

    # Post-processing

    # Nothing for now

    # Performance metrics
    return {"uri2ann_ref": uri2ann_ref, "uri2ann_hyp": uri2ann_hyp}


def calculate_metrics(
    collar, skip_overlap, uri2ann_ref, uri2ann_hyp, path_to_rttm, scoring_lib="pyannote"
):
    # scoring_lib : "pyannote" or "dscore"
    # Diarization Error Rate (DER)
    # NOTE: FA and Miss will be zero if skip_overlap=True and the Oracle VAD is used, that is, DER = Confusion

    assert scoring_lib in ["pyannote", "dscore"]
    if scoring_lib == "pyannote":
        # double the collar - https://github.com/pyannote/pyannote-metrics/issues/33
        collar = 2 * collar
        der_metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
        for uri, ann_hyp in uri2ann_hyp.items():
            ann_ref = uri2ann_ref[uri]
            der_metric(ann_ref, ann_hyp)

        report = der_metric.report(display=False)
        DER = report["diarization error rate"]["%"]["TOTAL"]
        # print(f"DER: {DER}%")

        # Jaccard Error Rate (JER)
        jer_metric = JaccardErrorRate(collar=collar, skip_overlap=skip_overlap)
        for uri, ann_hyp in uri2ann_hyp.items():
            ann_ref = uri2ann_ref[uri]
            jer_metric(ann_ref, ann_hyp)

        report = jer_metric.report(display=False)
        JER = report["jaccard error rate"]["%"]["TOTAL"]

    elif scoring_lib == "dscore":
        dir_with_rttms = f"./cache/RTTM/{path_to_rttm}"
        if os.path.exists(dir_with_rttms):
            shutil.rmtree(dir_with_rttms)
        os.makedirs(dir_with_rttms, exist_ok=True)
        for part in ["hyp", "ref"]:
            os.makedirs(os.path.join(dir_with_rttms, part), exist_ok=True)
        ref_rttms = []
        hyp_rttms = []
        for uri, ann_hyp in uri2ann_hyp.items():
            ann_ref = uri2ann_ref[uri]
            ref_rttm = os.path.join(dir_with_rttms, "ref", uri + ".rttm")
            with open(ref_rttm, "w") as f:
                ann_ref.write_rttm(f)
            ref_rttms.append(ref_rttm)

            hyp_rttm = os.path.join(dir_with_rttms, "hyp", uri + ".rttm")
            with open(hyp_rttm, "w") as f:
                ann_hyp.write_rttm(f)
            hyp_rttms.append(hyp_rttm)
        ref_turns, _ = load_rttms(ref_rttms)
        hyp_turns, _ = load_rttms(hyp_rttms)

        uem = gen_uem(ref_turns, hyp_turns)

        ref_turns = trim_turns(ref_turns, uem)
        hyp_turns = trim_turns(hyp_turns, uem)
        ref_turns = merge_turns(ref_turns)
        hyp_turns = merge_turns(hyp_turns)

        file_scores, global_scores = score(
            ref_turns,
            hyp_turns,
            uem,
            step=0.010,
            jer_min_ref_dur=0.0,
            collar=collar,
            ignore_overlaps=skip_overlap,
        )
        DER = global_scores.der
        JER = global_scores.jer

    # print(f"JER: {JER}%")
    return {"DER": DER, "JER": JER}


def run_evaluation(args):
    np.random.seed(0)

    # Arguments parsing
    EMBEDDINGS_NAME = args.emb
    CLUSTERING = args.alg
    DATASET_NAME = args.data
    WIN_SIZE = args.win
    STEP_SIZE = args.hop
    IGNORE_OVERLAP = args.ignore_overlap
    VAD_ORACLE = args.vad == "oracle"
    SCORING_LIB = args.scoring_lib
    OPT_LIB = args.opt_lib
    DEVICE = args.device

    # Embeddings preparation
    res_dict = prepare_embeddings(
        EMBEDDINGS_NAME,
        DATASET_NAME,
        CLUSTERING,
        WIN_SIZE,
        STEP_SIZE,
        IGNORE_OVERLAP,
        VAD_ORACLE,
        DEVICE,
    )
    uri2data = res_dict["uri2data"]
    uri2ann_ref = res_dict["uri2ann_ref"]

    # Clustering
    cache_id = f"{DATASET_NAME}_emb-{EMBEDDINGS_NAME}_vad-{VAD_ORACLE}_win-{WIN_SIZE:.1f}_step-{STEP_SIZE:.1f}_no-ovlp-{IGNORE_OVERLAP}"
    cache_params_path = os.path.join(
        cache_location,
        OPT_LIB,
        f"{cache_id.replace('test', 'dev')}_alg-{CLUSTERING}_params.p",
    )

    # TODO: we use the dev set for hyperparameter optimization
    try:
        with open(cache_params_path, "rb") as f:
            tuned_hyperparams = pickle.load(f)
        print("The tuned hyperparams have been set from cache")
    except:
        tuned_hyperparams = {}
        print(
            f"The tuned hyperparams have not been found in cache. The hyperparams from exp/diarization/parameters_{DATASET_NAME.split('_')[0]}_dev.py will be used."
        )
    res_dict = run_clustering(
        tuned_hyperparams,
        EMBEDDINGS_NAME,
        DATASET_NAME,
        CLUSTERING,
        uri2data,
        uri2ann_ref,
    )
    uri2ann_hyp = res_dict["uri2ann_hyp"]

    # Calculate metrics and print the results
    results_table = PrettyTable()
    results_table.title = DATASET_NAME
    results_table.field_names = ["Collar", "Skip overlap", "DER, %", "JER, %"]
    for (collar, skip_overlap) in [(0, False), (0, True), (0.25, True)]:
        res_dict = calculate_metrics(
            collar,
            skip_overlap,
            uri2ann_ref=uri2ann_ref,
            uri2ann_hyp=uri2ann_hyp,
            path_to_rttm=f"{cache_id}_{CLUSTERING}",
            scoring_lib=SCORING_LIB,
        )
        DER, JER = res_dict["DER"], res_dict["JER"]
        results_table.add_row([collar, skip_overlap, f"{DER:.2f}", f"{JER:.2f}"])
    print(results_table)


def default_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emb", 
        type=str, 
        required=True, 
        help="embeddings name"
    )
    parser.add_argument(
        "--alg", 
        type=str, 
        required=True, 
        help="clustering algorithm"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        required=True, 
        help="dataset name"
    )
    parser.add_argument(
        "-d", "--device", 
        type=int, 
        default=0, 
        help="cuda device id"
    )
    parser.add_argument(
        "--win", 
        type=float, 
        default=2.0, 
        help="length of the sliding window"
    )
    parser.add_argument(
        "--hop", 
        type=float, 
        default=1.0, 
        help="hop of the sliding window"
    )
    parser.add_argument(
        "--ignore_overlap",
        action="store_true",
        help="exclude segments with overlapped speech",
    )
    parser.add_argument(
        "--vad",
        type=str,
        choices=["oracle", "pyannote"],
        default="oracle",
        help="VAD name",
    )
    parser.add_argument(
        "-sl",
        "--scoring_lib",
        type=str,
        choices=["pyannote", "dscore"],
        default="dscore",
        help="lib for computing performance metrics",
    )
    parser.add_argument(
        "--opt_lib",
        type=str,
        choices=["optuna", "skopt"],
        default="skopt",
        help="lib for hyperparameter search",
    )
    return parser


if __name__ == "__main__":

    parser = default_argparser()
    args = parser.parse_args()

    run_evaluation(args)
