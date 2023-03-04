
CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH="${PYTHONPATH}:${CDIR}";

n_jobs=4

python exp/diarization/eval_hyperopt.py --emb speechbrain --alg online_csea --data AMI_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb speechbrain --alg online_cssa --data AMI_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb speechbrain --alg online_plda --data AMI_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb speechbrain --alg online_psda --data AMI_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb speechbrain --alg online_vb_plda --data AMI_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb speechbrain --alg online_vb_psda --data AMI_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;

python exp/diarization/eval_hyperopt.py --emb clova --alg online_csea --data AMI_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb clova --alg online_cssa --data AMI_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb clova --alg online_plda --data AMI_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb clova --alg online_psda --data AMI_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb clova --alg online_vb_plda --data AMI_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb clova --alg online_vb_psda --data AMI_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;

python exp/diarization/eval_hyperopt.py --emb brno --alg online_csea --data AMI_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb brno --alg online_cssa --data AMI_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb brno --alg online_plda --data AMI_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb brno --alg online_psda --data AMI_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb brno --alg online_vb_plda --data AMI_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb brno --alg online_vb_psda --data AMI_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;


python exp/diarization/eval_hyperopt.py --emb speechbrain --alg online_csea --data voxconverse_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb speechbrain --alg online_cssa --data voxconverse_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb speechbrain --alg online_plda --data voxconverse_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb speechbrain --alg online_psda --data voxconverse_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb speechbrain --alg online_vb_plda --data voxconverse_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb speechbrain --alg online_vb_psda --data voxconverse_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;

python exp/diarization/eval_hyperopt.py --emb clova --alg online_csea --data voxconverse_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb clova --alg online_cssa --data voxconverse_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb clova --alg online_plda --data voxconverse_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb clova --alg online_psda --data voxconverse_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb clova --alg online_vb_plda --data voxconverse_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb clova --alg online_vb_psda --data voxconverse_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;

python exp/diarization/eval_hyperopt.py --emb brno --alg online_csea --data voxconverse_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb brno --alg online_cssa --data voxconverse_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb brno --alg online_plda --data voxconverse_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb brno --alg online_psda --data voxconverse_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb brno --alg online_vb_plda --data voxconverse_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
python exp/diarization/eval_hyperopt.py --emb brno --alg online_vb_psda --data voxconverse_dev --win 2.0 --hop 1.0 --opt_lib skopt --n_jobs $n_jobs;
