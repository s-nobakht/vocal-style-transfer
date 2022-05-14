import math
from collections import OrderedDict

import torch as th
from torch import nn
from transformers import Wav2Vec2Model, Wav2Vec2Config

from .utils import compute_loudness, compute_sine_exc
from ..nn import MelGANDiscriminator, WaveGradGenerator
from ..nn.utils import zero_mean_unit_var_norm


class Wav2MelInferenceModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Wav2Vec2Model.from_pretrained(self.hparams.encoder)

        self.adjust = nn.Sequential(OrderedDict([
            ('layer_norm', nn.LayerNorm(768)),
            ('projection', nn.Linear(768, 400)),
            ('dropout', nn.Dropout(0.1)),
        ]))

        self.generator = WaveGradGenerator()

    def forward(self, waveform_user, waveform_ref, waveform_ref_len):
        if waveform_ref_len.item() > waveform_user.size(1):
            waveform_user = waveform_user.repeat(1, math.ceil(waveform_ref_len.item() / waveform_user.size(1)))
        waveform_user = waveform_user[:, :waveform_ref_len.item()].reshape(-1, 16000)

        sine_exc, _, _, _ = compute_sine_exc(waveform_ref)
        sine_exc = zero_mean_unit_var_norm(sine_exc, dim=1, keepdim=True)

        loudness = compute_loudness(waveform_ref)
        loudness = zero_mean_unit_var_norm(loudness, dim=1, keepdim=True)

        x = self.encoder(waveform_user, attention_mask=waveform_mask).last_hidden_state

        x = self.adjust(x)
        x_hat = self.generator(x, sine_exc, loudness)

        return waveform
