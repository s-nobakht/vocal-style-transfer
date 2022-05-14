import math

import torch as th
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchaudio import sox_effects


def collate_fn(batch):
    waveform = th.stack([x[0] for x in batch])
    waveform_mask = pad_sequence([x[1] for x in batch], batch_first=True)

    return waveform, waveform_mask


def dummy_collate_fn(batch):
    return batch


def process_sample(idx, waveform, sample_rate, effects, window_size, root):
    waveform, _ = sox_effects.apply_effects_tensor(waveform,
                                                   sample_rate=sample_rate,
                                                   effects=effects,
                                                   channels_first=True)
    waveform = waveform.squeeze()
    for j in range(math.ceil(waveform.numel() / window_size)):
        start = j * window_size
        finish = min((j + 1) * window_size, waveform.numel())
        waveform_j = waveform[start:finish]
        waveform_j_len = waveform_j.numel()
        waveform_j = F.pad(waveform_j, [0, window_size - waveform_j_len])
        th.save(waveform_j.clone(), f'{root}/{idx}_{j}_{waveform_j_len}.pt')
