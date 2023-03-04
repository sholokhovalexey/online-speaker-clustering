
CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export PYTHONPATH="${PYTHONPATH}:${CDIR}"; 


python exp/diarization/run_diarization.py --emb clova --alg online_csea --data AMI_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb clova --alg online_cssa --data AMI_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb clova --alg online_plda --data AMI_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb clova --alg online_psda --data AMI_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb clova --alg online_vb_plda --data AMI_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb clova --alg online_vb_psda --data AMI_test --win 2.0 --hop 1.0;


python exp/diarization/run_diarization.py --emb speechbrain --alg online_csea --data AMI_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb speechbrain --alg online_cssa --data AMI_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb speechbrain --alg online_plda --data AMI_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb speechbrain --alg online_psda --data AMI_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb speechbrain --alg online_vb_plda --data AMI_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb speechbrain --alg online_vb_psda --data AMI_test --win 2.0 --hop 1.0;


python exp/diarization/run_diarization.py --emb brno --alg online_csea --data AMI_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb brno --alg online_cssa --data AMI_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb brno --alg online_plda --data AMI_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb brno --alg online_psda --data AMI_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb brno --alg online_vb_plda --data AMI_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb brno --alg online_vb_psda --data AMI_test --win 2.0 --hop 1.0;


python exp/diarization/run_diarization.py --emb clova --alg online_csea --data voxconverse_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb clova --alg online_cssa --data voxconverse_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb clova --alg online_plda --data voxconverse_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb clova --alg online_psda --data voxconverse_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb clova --alg online_vb_plda --data voxconverse_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb clova --alg online_vb_psda --data voxconverse_test --win 2.0 --hop 1.0;


python exp/diarization/run_diarization.py --emb speechbrain --alg online_csea --data voxconverse_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb speechbrain --alg online_cssa --data voxconverse_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb speechbrain --alg online_plda --data voxconverse_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb speechbrain --alg online_psda --data voxconverse_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb speechbrain --alg online_vb_plda --data voxconverse_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb speechbrain --alg online_vb_psda --data voxconverse_test --win 2.0 --hop 1.0;


python exp/diarization/run_diarization.py --emb brno --alg online_csea --data voxconverse_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb brno --alg online_cssa --data voxconverse_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb brno --alg online_plda --data voxconverse_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb brno --alg online_psda --data voxconverse_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb brno --alg online_vb_plda --data voxconverse_test --win 2.0 --hop 1.0;
python exp/diarization/run_diarization.py --emb brno --alg online_vb_psda --data voxconverse_test --win 2.0 --hop 1.0;

