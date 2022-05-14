import os

import torch as th
import torchaudio
import torch.nn.functional as F

from src.nn.transforms import Resampler
from src.data.utils import calculate_loudness, calculate_sine_excitation
from src.models import FastSVCInferenceModule

resampler = Resampler()


def _audio():
    waveform_ref, sample_rate_ref = torchaudio.load('./data/nus-smc-corpus_48/ADIZ/sing/18.wav')
    torchaudio.save('test_inference_ref.wav', waveform_ref, sample_rate=sample_rate_ref)

    waveform_ref, sample_rate_ref = resampler(waveform_ref, sample_rate_ref)
    torchaudio.save('test_inference_ref_resampled.wav', waveform_ref, sample_rate=sample_rate_ref)

    waveform_user, sample_rate_user = torchaudio.load('./data/nus-smc-corpus_48/ADIZ/sing/18.wav')
    torchaudio.save('test_inference_user.wav', waveform_user, sample_rate=sample_rate_user)

    waveform_user, sample_rate_user = resampler(waveform_user, sample_rate_user)
    torchaudio.save('test_inference_user_resampled.wav', waveform_user, sample_rate=sample_rate_user)

    waveform_ref = F.pad(waveform_ref, [0, 16000 - waveform_ref.size(1) % 16000, 0, 0]).squeeze()
    waveform_ref_len = th.tensor(waveform_ref.numel()).long()

    sine_excitation = th.from_numpy(calculate_sine_excitation(waveform_ref.numpy())[0]).reshape(-1, 1, 16000).float()
    loudness = th.from_numpy(calculate_loudness(waveform_ref.numpy())).float().reshape(-1, 1, 16000)
    loudness = ((loudness - loudness.mean(dim=2, keepdim=True)) / loudness.std(dim=2, keepdim=True))

    return waveform_user, waveform_ref_len, sine_excitation, loudness


@th.no_grad()
def main():
    checkpoint = os.getenv('checkpoint', None)

    waveform_user, waveform_ref_len, sine_excitation, loudness = _audio()

    model = FastSVCInferenceModule()
    model.eval()

    if checkpoint is not None:
        checkpoint = th.load(f'./checkpoints/fastsvc/{checkpoint}.ckpt', map_location='cpu')
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):  # NOTE: Remove discriminator.
            if k.startswith('discriminator.'):
                del state_dict[k]
        model.encoder.load_state_dict({k[len('encoder.'):]: v
                                       for k, v in state_dict.items() if k.startswith('encoder.')})
        model.generator.load_state_dict({k[len('generator.'):]: v
                                         for k, v in state_dict.items() if k.startswith('generator.')})

    jit_trace = th.jit.trace(model, (waveform_user, waveform_ref_len, sine_excitation, loudness))
    jit_trace.save('./models/inference.pt')

    waveform = model(waveform_user, waveform_ref_len, sine_excitation, loudness)
    torchaudio.save('test_inference_reconstructed.wav', waveform, sample_rate=16000)


if __name__ == '__main__':
    main()
