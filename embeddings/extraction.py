import glob2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data_io import load_audio

from collections import namedtuple

Segment = namedtuple("Segment", ["start", "end"])
# from pyannote.core import Segment


def invert_permutation(p):
    """Return an array s with which np.array_equal(arr[p][s], arr) is True.
    The array_like argument p must be some permutation of 0, 1, ..., len(p)-1.
    Source: https://stackoverflow.com/questions/11649577/how-to-invert-a-permutation-array-in-numpy
    """
    p = np.asanyarray(p)  # in case p is a tuple, etc.
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s


def extract_embeddings_wav(
    embedder,
    audio,
    device="cuda:0",
    segments=(),
    batch_size=1,
    sample_rate=16000,
    verbose=False,
):

    try:
        embedder.eval()
    except:
        pass

    if isinstance(audio, str):
        waveform, _ = load_audio(audio)
    elif torch.is_tensor(audio):
        waveform = audio
    else:
        waveform = torch.tensor(audio).view(1, -1)

    wav_length = waveform.shape[-1]

    if len(segments) == 0:
        segments_all = [Segment(0.0, wav_length / sample_rate)]
    else:
        segments_all = (
            segments.copy()
        )  # to avoid modifying segments by in-place operations

    # make sure that segments boundaries are within the actual waveform
    for i in range(len(segments_all)):
        segments_all[i] = Segment(
            segments_all[i].start, min(segments_all[i].end, wav_length / sample_rate)
        )  # in-place

    # find unique segment lengths and group the segments with the same length for batch processing
    if batch_size > 1:
        lengths = [
            int(min(sample_rate * seg.end, wav_length) - sample_rate * seg.start)
            for seg in segments_all
        ]
        lengths_unique, idx_inverse = np.unique(lengths, return_inverse=True)
        segments_sets = []
        for k in range(len(lengths_unique)):
            idx = np.nonzero(idx_inverse == k)[0]
            segments_set = []
            for i in idx:
                segments_set += [(segments_all[i], i)]
            segments_sets += [segments_set]
    else:
        segments_set = [(seg, i) for (i, seg) in enumerate(segments_all)]
        segments_sets = [segments_set]

    embeddings = []
    indices = []
    for segments_set in segments_sets:
        if verbose:
            t = tqdm(segments_set)
        else:
            t = segments_set
        wavs_batch = []
        for i, (segment, idx) in enumerate(t):
            indices += [idx]
            start = segment.start
            end = segment.end
            duration = end - start

            sample_start = int(sample_rate * start)
            # sample_end = int(sample_rate * end)
            num_samples = int(sample_rate * duration)  # variable duration
            # num_samples = int(sample_rate * win_size)

            wav = waveform[
                ..., sample_start : min(sample_start + num_samples, wav_length)
            ]
            # wav = waveform[..., sample_start: min(sample_end, wav_length)]
            wav = torch.as_tensor(wav).view(1, -1)
            wavs_batch += [wav]

            if (i + 1) % batch_size == 0 or (i + 1) == len(segments_set):

                # crop waves in the batch to the minimum length
                lengths_batch = [wav.shape[-1] for wav in wavs_batch]
                n_min = min(lengths_batch)
                n_max = max(lengths_batch)
                if n_min != n_max:
                    # print(f"Warning: min and max lengths in the batch are not equal: {n_min} != {n_max}, cropping.")
                    wavs_batch = [wav[..., :n_min] for wav in wavs_batch]

                with torch.no_grad():
                    emb_batch = embedder(torch.cat(wavs_batch).to(device)).cpu()

                embeddings += [emb_batch]
                wavs_batch = []

    embeddings = torch.cat(embeddings).cpu().numpy()
    indices = np.array(indices)
    embeddings = embeddings[invert_permutation(indices)]
    return embeddings


def extract_embeddings_path(
    embedder,
    data_root,
    device="cuda:0",
    path2utt_fn=lambda p: p,
    batch_size=1,
    crop_size=None,
    crop_center=True,
    file_ext="wav",
    subset=None,
):

    try:
        embedder = embedder.eval()
    except:
        pass

    wav_files = glob2.glob(f"{data_root}/**/*.{file_ext}")

    if subset is not None:
        wav_files = [wav for wav in wav_files if path2utt_fn(wav) in subset]
        print("Subset:", len(subset))
        print("Found:", len(wav_files))

    if crop_size is None:
        assert batch_size == 1

    embeddings = []
    utt_ids = []

    wavs_batch = []
    for i, wav_path in enumerate(tqdm(wav_files)):

        utt_id = path2utt_fn(wav_path)
        utt_ids += [utt_id]

        waveform, _ = load_audio(wav_path)
        length = waveform.shape[-1]
        length_half = int(0.5 * length)

        if crop_size is not None:
            if crop_center:
                crop_size_half = int(0.5 * crop_size)
                start = max(0, length_half - crop_size_half)
                waveform_crop = waveform[..., start : start + crop_size]
            else:
                waveform_crop = waveform[..., :crop_size]

            # pad if the waveform is too short
            if crop_size > length:
                pad_size = (0, crop_size - length)
                waveform_crop = F.pad(waveform_crop, pad_size)
        else:
            waveform_crop = waveform

        wavs_batch += [waveform_crop]

        if (i + 1) % batch_size == 0 or (i + 1) == len(wav_files):
            with torch.no_grad():
                bs = len(wavs_batch)
                emb_batch = embedder(torch.cat(wavs_batch).to(device)).cpu()
                emb_batch = emb_batch.view(
                    bs, -1
                )  # ad-hoc for speechbrain, no need anymore ?

            embeddings += [emb_batch]
            wavs_batch = []

    embeddings = torch.cat(embeddings).cpu().numpy()
    return embeddings, utt_ids
