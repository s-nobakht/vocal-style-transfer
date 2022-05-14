import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        nn.init.zeros_(self.bias)


class FiLM(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_conv = nn.Conv1d(input_size, input_size, 3, padding=1)
        self.output_conv = nn.Conv1d(input_size, output_size * 2, 3, padding=1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_conv.weight)
        nn.init.xavier_uniform_(self.output_conv.weight)
        nn.init.zeros_(self.input_conv.bias)
        nn.init.zeros_(self.output_conv.bias)

    def forward(self, x):
        x = self.input_conv(x)
        x = F.leaky_relu(x, 0.2)
        shift, scale = torch.chunk(self.output_conv(x), 2, dim=1)
        return shift, scale


class UBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor, dilation):
        super().__init__()
        assert isinstance(dilation, (list, tuple))
        assert len(dilation) == 4

        self.factor = factor
        self.block1 = Conv1d(input_size, hidden_size, 1)
        self.block2 = nn.ModuleList([
            Conv1d(input_size, hidden_size, 3, dilation=dilation[0], padding=dilation[0]),
            Conv1d(hidden_size, hidden_size, 3, dilation=dilation[1], padding=dilation[1])
        ])
        self.block3 = nn.ModuleList([
            Conv1d(hidden_size, hidden_size, 3, dilation=dilation[2], padding=dilation[2]),
            Conv1d(hidden_size, hidden_size, 3, dilation=dilation[3], padding=dilation[3])
        ])

    def forward(self, x, film_shift, film_scale):
        block1 = F.interpolate(x, size=x.size(-1) * self.factor)
        block1 = self.block1(block1)

        block2 = F.leaky_relu(x, 0.2)
        block2 = F.interpolate(block2, size=x.size(-1) * self.factor)
        block2 = self.block2[0](block2)
        block2 = film_shift + film_scale * block2
        block2 = F.leaky_relu(block2, 0.2)
        block2 = self.block2[1](block2)

        x = block1 + block2

        block3 = film_shift + film_scale * x
        block3 = F.leaky_relu(block3, 0.2)
        block3 = self.block3[0](block3)
        block3 = film_shift + film_scale * block3
        block3 = F.leaky_relu(block3, 0.2)
        block3 = self.block3[1](block3)

        x = x + block3
        return x


class DBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor):
        super().__init__()
        self.factor = factor
        self.residual_dense = Conv1d(input_size, hidden_size, 1)
        self.conv = nn.ModuleList([
            Conv1d(input_size, hidden_size, 3, dilation=1, padding=1),
            Conv1d(hidden_size, hidden_size, 3, dilation=2, padding=2),
            Conv1d(hidden_size, hidden_size, 3, dilation=4, padding=4),
        ])

    def forward(self, x):
        size = x.size(-1) // self.factor

        residual = self.residual_dense(x)
        residual = F.interpolate(residual, size=size)

        x = F.interpolate(x, size=size)
        for layer in self.conv:
            x = F.leaky_relu(x, 0.2)
            x = layer(x)

        return x + residual


class WaveGradGenerator(nn.Module):
    @staticmethod
    def _down_branch():
        downsample = nn.ModuleList([
            Conv1d(1, 16, 5, padding=2),
            DBlock(16, 64, 2),
            DBlock(64, 64, 2),
            DBlock(64, 128, 2),
        ])
        film = nn.ModuleList([
            FiLM(16, 64),
            FiLM(64, 64),
            FiLM(64, 128),
            FiLM(128, 256),
        ])
        return downsample, film

    def __init__(self):
        super().__init__()
        self.downsample_sine_exc, self.film_sine_exc = self._down_branch()
        self.downsample_loudness, self.film_loudness = self._down_branch()
        self.upsample = nn.ModuleList([
            UBlock(256, 256, 5, [1, 2, 1, 2]),
            UBlock(256, 128, 2, [1, 2, 4, 8]),
            UBlock(128, 64, 2, [1, 2, 4, 8]),
            UBlock(64, 64, 2, [1, 2, 4, 8]),
        ])
        self.first_conv = Conv1d(49, 256, 3, padding=1)
        self.last_conv = Conv1d(64, 1, 3, padding=1)

    @staticmethod
    def _down_forward(waveform, film_waveform, downsample_waveform):
        x = waveform.unsqueeze(1)
        downsampled = []
        for film, layer in zip(film_waveform, downsample_waveform):
            x = layer(x)
            downsampled.append(film(x))
        return downsampled

    def forward(self, wave2vec2, sine_exc, loudness):
        downsampled_sine_exc = self._down_forward(sine_exc, self.film_sine_exc, self.downsample_sine_exc)
        downsampled_loudness = self._down_forward(loudness, self.film_loudness, self.downsample_loudness)

        x = self.first_conv(wave2vec2)
        for layer, ((film_shift_sine_exc, film_scale_sine_exc), (film_shift_loudness, film_scale_loudness)) \
                in zip(self.upsample, zip(reversed(downsampled_sine_exc), reversed(downsampled_loudness))):
            x = layer(x, film_shift_sine_exc + film_shift_loudness, film_scale_sine_exc + film_scale_loudness)
        x = self.last_conv(x)
        return x
