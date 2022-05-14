import math

import torch as th
import torch.nn.functional as F
import torchcrepe
from torchaudio.compliance import kaldi

LD_RANGE = 120.0


def _fft_frequencies(sr=22050, n_fft=2048, device='cpu'):
    return th.linspace(0, float(sr) / 2, int(1 + n_fft // 2), device=device)


def _a_weighting(frequencies, min_db=-80.0):
    # Pre-compute squared frequency
    f_sq = frequencies.pow(2)

    const = th.tensor([12200, 20.6, 107.7, 737.9], device=f_sq.device).pow(2)

    weights = 2.0 + 20.0 * (th.log10(const[0]) + 4 * th.log10(frequencies)
                            - th.log10(f_sq + const[0])
                            - th.log10(f_sq + const[1])
                            - 0.5 * th.log10(f_sq + const[2])
                            - 0.5 * th.log10(f_sq + const[3]))

    if min_db is not None:
        weights = th.clamp(weights, min=min_db)

    return weights


def compute_loudness(waveform,
                     sample_rate=16000,
                     hop_length=64,
                     n_fft=2048,
                     range_db=LD_RANGE,
                     ref_db=20.7):

    spectra = th.stft(waveform,
                      n_fft=n_fft,
                      hop_length=hop_length,
                      win_length=n_fft,
                      center=False,
                      return_complex=True).permute(0, 2, 1)

    # Compute power.
    amplitude = spectra.abs().nan_to_num()
    amin = 1e-20
    power_db = 20.0 * th.log10(th.clamp(amplitude, min=amin))

    # Perceptual weighting.
    frequencies = _fft_frequencies(sample_rate, n_fft, device=power_db.device)
    a_weighting = _a_weighting(frequencies).reshape(1, 1, -1)
    loudness = power_db + a_weighting

    # Set dynamic range.
    loudness -= ref_db
    loudness = th.clamp(loudness, min=-range_db)

    # Average over frequency bins.
    loudness = loudness.mean(dim=-1)

    # Compute expected length of loudness vector.
    n_secs = waveform.size(-1) / float(sample_rate)  # `n_secs` can have milliseconds
    expected_len = int(n_secs * (sample_rate / hop_length))

    if loudness.size(1) < expected_len:
        n_padding = expected_len - loudness.size(1)
        loudness = F.pad(loudness, [0, n_padding, 0, 0], mode='constant', value=-range_db)
    elif loudness.size(1) > expected_len:
        loudness = loudness[..., :expected_len]

    loudness = F.interpolate(loudness.unsqueeze(1), waveform.size(1), mode='linear').squeeze()

    return loudness


def compute_sine_exc(waveform, sample_rate=16000):
    f0 = torchcrepe.predict(waveform,
                            sample_rate=sample_rate,
                            model='tiny',
                            decoder=torchcrepe.decode.weighted_argmax,
                            device=waveform.device).nan_to_num()

    # Upsample funamental frequncies.
    sr = f0.size(1) / (waveform.size(1) / sample_rate)
    f0 = kaldi.resample_waveform(f0, orig_freq=sr, new_freq=sample_rate)

    # Generate random noise.
    nt = th.normal(0, 0.003 ** 2, size=f0.size(), device=f0.device)

    # Compute sinusoidal wave
    e = f0 * (2.0 * math.pi)
    e = e / float(sample_rate)
    e = e.cumsum(0)
    e += 2 * math.pi * th.rand(1, device=e.device) - math.pi
    e = e.sin()
    e *= 0.1

    # Compute sine-excitation
    e_zero = 100 * nt

    mask = f0 > 20
    e = e * mask
    e_zero = e_zero * ~mask

    return e + e_zero, e, e_zero, mask
