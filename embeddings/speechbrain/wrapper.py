import os
from speechbrain.pretrained import EncoderClassifier

from pretrained import model_path_speechbrain


class EmbedderSpeechBrain:
    def __init__(self, model_path, device):
        # super().__init__()

        # load from the Huggingface hub
        # self.classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
        #                                                 run_opts={"device": device})

        # load from a local file
        savedir = os.path.dirname(model_path)
        source = os.path.dirname(model_path)
        self.classifier = EncoderClassifier.from_hparams(
            source=source,
            savedir=savedir,
            hparams_file="hyperparams.yaml",
            run_opts={"device": device},
        )

    def __call__(self, wav):
        batch_size = wav.shape[0]
        x = self.classifier.encode_batch(wav).view(batch_size, -1)
        return x

    def to(self, dst):
        # print("Do nothing")
        pass


def prepare_model_speechbrain(model_path=model_path_speechbrain, device="cuda:0"):
    return EmbedderSpeechBrain(model_path, device)
