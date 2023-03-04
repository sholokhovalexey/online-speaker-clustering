import torch
import torchaudio


def load_audio(path, sum_channels=True):
    signal, sr = torchaudio.load(path)
    n_channels = signal.shape[0]
    if n_channels > 1:
        # signal = signal[0:1]
        if sum_channels:
            signal = torch.mean(signal, dim=0, keepdim=True)
    if torch.max(torch.abs(signal)) <= 1:
        signal = signal * (2 ** 15 - 1)
    return signal, sr
