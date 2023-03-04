
### VAD/OSD
#rm -rf pretrained/frontend
#mkdir -p pretrained/frontend
#wget https://huggingface.co/pyannote/segmentation/resolve/main/pytorch_model.bin -P pretrained/frontend

# initialize submodules
git submodule init
git submodule update


### clova
git clone https://github.com/clovaai/voxceleb_trainer.git embeddings/clova/voxceleb_trainer
rm -rf pretrained/clova
mkdir -p pretrained/clova
wget http://www.robots.ox.ac.uk/~joon/data/baseline_v2_ap.model -P pretrained/clova
# remove the nested git repository
rm -rf embeddings/clova/voxceleb_trainer/.git* 
# fix the import errors
cat embeddings/clova/voxceleb_trainer/models/ResNetSE34V2.py | sed -e "s/from models.ResNetBlocks import/from ..models.ResNetBlocks import/" > embeddings/clova/voxceleb_trainer/models/ResNetSE34V2.txt
mv embeddings/clova/voxceleb_trainer/models/ResNetSE34V2.txt embeddings/clova/voxceleb_trainer/models/ResNetSE34V2.py
cat embeddings/clova/voxceleb_trainer/models/ResNetSE34V2.py | sed -e "s/from utils import PreEmphasis/from ..utils import PreEmphasis/" > embeddings/clova/voxceleb_trainer/models/ResNetSE34V2.txt
mv embeddings/clova/voxceleb_trainer/models/ResNetSE34V2.txt embeddings/clova/voxceleb_trainer/models/ResNetSE34V2.py


### speechbrain
rm -rf pretrained/speechbrain
mkdir -p pretrained/speechbrain
wget -O pretrained/speechbrain/hyperparams.yaml https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/hyperparams.yaml
wget -O pretrained/speechbrain/embedding_model.ckpt https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/embedding_model.ckpt
wget -O pretrained/speechbrain/classifier.ckpt https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/classifier.ckpt
wget -O pretrained/speechbrain/mean_var_norm_emb.ckpt https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/mean_var_norm_emb.ckpt
wget -O pretrained/speechbrain/label_encoder.ckpt https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/resolve/main/label_encoder.txt

### brno
rm -rf pretrained/brno
mkdir -p pretrained/brno
#wget https://data-tx.oss-cn-hangzhou.aliyuncs.com/AISHELL-4-Code/sd-part.zip
#unzip sd-part.zip -d pretrained/brno
VBx_path="pretrained/brno/VBx"
git clone https://github.com/BUTSpeechFIT/VBx.git $VBx_path
cat $VBx_path/VBx/models/ResNet101_16kHz/nnet/raw_81.pth.zip* > $VBx_path/VBx/models/ResNet101_16kHz/nnet/raw_81.pth.zip
unzip $VBx_path/VBx/models/ResNet101_16kHz/nnet/raw_81.pth.zip -d $VBx_path/VBx/models/ResNet101_16kHz/nnet
mv $VBx_path/VBx/models/ResNet101_16kHz pretrained/brno 
rm -rf $VBx_path
