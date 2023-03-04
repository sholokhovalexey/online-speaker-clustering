import numpy as np
# import onnxruntime
import torch

from .features import (
    povey_window,
    mel_fbank_mx,
    fbank_htk,
    cmvn_floating_kaldi,
    add_dither,
)
from .resnet import ResNet101

from pretrained import model_path_brno


class EmbedderBUT:
    def __init__(self, weights_path, device, backend):
        self.backend = backend
        self.device = device
        if backend == "pytorch":
            feat_dim = 64
            embed_dim = 256
            model = ResNet101(feat_dim=feat_dim, embed_dim=embed_dim)
            model = model.to(device)
            checkpoint = torch.load(weights_path, map_location=device)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            model.eval()
        elif backend == "onnx":
            raise NotImplementedError
            # model = onnxruntime.InferenceSession(
            #     weights_path, providers=["CUDAExecutionProvider"]
            # )
            # self.input_name = model.get_inputs()[0].name
            # self.label_name = model.get_outputs()[0].name

        self.model = model

    def __call__(self, wav_batch):
        batch_size = wav_batch.shape[0]

        # extract features
        samplerate = 16000
        noverlap = 240
        winlen = 400
        window = povey_window(winlen)
        fbank_mx = mel_fbank_mx(
            winlen, samplerate, NUMCHANS=64, LOFREQ=20.0, HIFREQ=7600, htk_bug=False
        )

        wav_batch = wav_batch.cpu().numpy()  # .ravel()

        features = []
        for wav in wav_batch:

            if np.max(np.abs(wav)) <= 1:
                wav = wav * 2 ** 15

            np.random.seed(3)  # for reproducibility
            wav = add_dither((wav * 2 ** 15).astype(int))

            fea = fbank_htk(
                wav, window, noverlap, fbank_mx, USEPOWER=True, ZMEANSOURCE=True
            )
            LC = 150
            RC = 149
            fea = cmvn_floating_kaldi(fea, LC, RC, norm_vars=False).astype(np.float32)
            features += [fea.T[None, :, :]]  # (batch, n_features, n_frames)

        features = np.concatenate(features)

        if self.backend == "pytorch":
            x = get_embedding_batch(
                features, self.model, self.device, backend="pytorch"
            )
        elif self.backend == "onnx":
            # NOTE: batch_size=1 only implemented
            raise NotImplementedError
            # x = get_embedding_single(
            #     features[0],
            #     self.model,
            #     self.device,
            #     label_name=self.label_name,
            #     input_name=self.input_name,
            #     backend="onnx",
            # )
        return x.reshape(batch_size, -1)


def get_embedding_single(
    fea, model, device, label_name=None, input_name=None, backend="pytorch"
):
    if backend == "pytorch":
        data = torch.from_numpy(fea).to(device)
        data = data[None, :, :]
        data = torch.transpose(data, 1, 2)  # (batch, n_features, n_frames)
        spk_embeds = model(data)
        return spk_embeds.data  # .cpu().numpy()[0]
    elif backend == "onnx":
        raise NotImplementedError
        # emb = model.run(
        #     [label_name],
        #     {input_name: fea.astype(np.float32).transpose()[np.newaxis, :, :]},
        # )[0].squeeze()
        # return torch.tensor(emb)


def get_embedding_batch(
    fea, model, device, label_name=None, input_name=None, backend="pytorch"
):
    if backend == "pytorch":
        data = torch.from_numpy(fea).to(device)
        spk_embeds = model(data)
        return spk_embeds.data  # .cpu().numpy()[0]
    elif backend == "onnx":
        raise NotImplementedError


def prepare_model_brno(model_path=model_path_brno, device="cuda:0", backend="pytorch"):
    return EmbedderBUT(model_path, device, backend)

