import os
import pickle
import argparse
from functools import partial
import numpy as np
import torch

from run_diarization import (
    prepare_embeddings,
    run_clustering,
    calculate_metrics,
    default_argparser,
)

import optuna
from optuna.samplers import TPESampler
from skopt import gp_minimize
from skopt.space import Real


torch.set_grad_enabled(False)

root_dir = os.path.abspath(os.getcwd())
cache_location = f"{root_dir}/cache/diarization"

min_duration = 0.5  # sec


def hyperparams_tuning_step_optuna(
    trial,
    hyperparam_name2idx,
    hyperparam_name2type,
    space,
    EMBEDDINGS_NAME,
    CLUSTERING,
    uri2data,
    uri2ann_ref,
    path_to_rttm,
    versbose=False,
):
    hyperparams_dict = {}
    for hyperparam_name in hyperparam_name2type:
        bounds = space[hyperparam_name2idx[hyperparam_name]].bounds
        hyperparams_dict.update(
            {
                hyperparam_name: trial.suggest_float(
                    hyperparam_name, bounds[0], bounds[1]
                )
            }
        )
    # print(hyperparams_dict)
    np.random.seed(0)

    res_dict = run_clustering(
        hyperparams_dict,
        EMBEDDINGS_NAME,
        CLUSTERING,
        uri2data,
        uri2ann_ref,
        versbose=versbose,
    )
    uri2ann_hyp = res_dict["uri2ann_hyp"]

    collar, skip_overlap = 0.25, True
    DER = calculate_metrics(
        collar=collar,
        skip_overlap=skip_overlap,
        uri2ann_ref=uri2ann_ref,
        uri2ann_hyp=uri2ann_hyp,
        path_to_rttm=path_to_rttm,
    )["DER"]
    # print(f"DER: {DER}%")
    return DER


def hyperparams_tuning_step_skopt(
    hyperparams,
    hyperparam_name2idx,
    EMBEDDINGS_NAME,
    CLUSTERING,
    uri2data,
    uri2ann_ref,
    path_to_rttm,
    versbose=False,
):
    hyperparams_dict = {
        hyperparam_name: hyperparams[hyperparam_name2idx[hyperparam_name]]
        for hyperparam_name in hyperparam_name2idx
    }
    # print(hyperparams_dict)
    np.random.seed(0)

    res_dict = run_clustering(
        hyperparams_dict,
        EMBEDDINGS_NAME,
        CLUSTERING,
        uri2data,
        uri2ann_ref,
        versbose=versbose,
    )
    uri2ann_hyp = res_dict["uri2ann_hyp"]

    collar, skip_overlap = 0.25, True
    DER = calculate_metrics(
        collar=collar,
        skip_overlap=skip_overlap,
        uri2ann_ref=uri2ann_ref,
        uri2ann_hyp=uri2ann_hyp,
        path_to_rttm=path_to_rttm,
    )["DER"]
    # print(f"DER: {DER}%")
    return DER


if __name__ == "__main__":

    parser = default_argparser()
    parser.add_argument("--n_jobs", type=int, default=4)
    args = parser.parse_args()

    EMBEDDINGS_NAME = args.emb
    CLUSTERING = args.alg
    DATASET_NAME = args.data
    WIN_SIZE = args.win
    STEP_SIZE = args.hop
    IGNORE_OVERLAP = args.ignore_overlap
    VAD_ORACLE = args.vad == "oracle"
    DEVICE = f"cuda:{args.device}"
    OPT_LIB = args.opt_lib
    N_JOBS = args.n_jobs

    cache_id = f"{DATASET_NAME}_emb-{EMBEDDINGS_NAME}_vad-{VAD_ORACLE}_win-{WIN_SIZE:.1f}_step-{STEP_SIZE:.1f}_no-ovlp-{IGNORE_OVERLAP}"

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

    # Hyperparameter tuning
    if CLUSTERING in ["online_csea", "online_cssa"]:
        space = [Real(-0.1, 0.7, "uniform", name="threshold")]
    elif CLUSTERING in [
        "online_plda",
        "online_psda",
        "online_vb_plda",
        "online_vb_psda",
    ]:
        space = [Real(-100, 50, "uniform", name="threshold")]
    hyperparam_name2idx = {elem.name: idx for idx, elem in enumerate(space)}
    hyperparam_name2type = {elem.name: elem.dtype for elem in space}

    n_calls = 50

    # TODO: unified print for both optimizers
    if OPT_LIB == "skopt":
        fmin_objective = partial(
            hyperparams_tuning_step_skopt,
            hyperparam_name2idx=hyperparam_name2idx,
            EMBEDDINGS_NAME=EMBEDDINGS_NAME,
            CLUSTERING=CLUSTERING,
            uri2data=uri2data,
            uri2ann_ref=uri2ann_ref,
            path_to_rttm=f"{cache_id}_{CLUSTERING}",
        )
        result = gp_minimize(
            fmin_objective,
            space,
            n_calls=n_calls,
            random_state=0,
            n_jobs=N_JOBS,
            verbose=False,
        )
        print("Best:")
        print(result["fun"])
        print(result["x"])
        result = result["x"]
        tuned_hyperparams = {
            hyperparam_name: result[hyperparam_name2idx[hyperparam_name]]
            for hyperparam_name in hyperparam_name2idx
        }
    elif OPT_LIB == "optuna":
        fmin_objective = partial(
            hyperparams_tuning_step_optuna,
            hyperparam_name2idx=hyperparam_name2idx,
            hyperparam_name2type=hyperparam_name2type,
            space=space,
            EMBEDDINGS_NAME=EMBEDDINGS_NAME,
            CLUSTERING=CLUSTERING,
            uri2data=uri2data,
            uri2ann_ref=uri2ann_ref,
            path_to_rttm=f"{cache_id}_{CLUSTERING}",
        )
        sampler = TPESampler(seed=10)
        # optuna.logging.set_verbosity(optuna.logging.INFO)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            fmin_objective, n_trials=n_calls, n_jobs=N_JOBS, show_progress_bar=True
        )
        tuned_hyperparams = study.best_params
        print("Best:")
        print(study.best_value)
        print(study.best_params)
    else:
        raise (NotImplementedError)

    os.makedirs(os.path.join(cache_location, OPT_LIB), exist_ok=True)
    cache_params_path = os.path.join(
        cache_location, OPT_LIB, f"{cache_id}_alg-{CLUSTERING}_params.p"
    )
    with open(cache_params_path, "wb") as f:
        pickle.dump(tuned_hyperparams, f)
