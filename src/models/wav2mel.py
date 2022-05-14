from argparse import ArgumentParser
from collections import OrderedDict

import pytorch_lightning as pl
import torch as th
from torch import nn
from auraloss.freq import MultiResolutionSTFTLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from transformers import Wav2Vec2Model

from .utils import compute_loudness, compute_sine_exc
from ..nn import MelGANDiscriminator, WaveGradGenerator
from ..nn.utils import zero_mean_unit_var_norm


class Wav2MelGradModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.automatic_optimization = False

        self.hparams = hparams

        self.encoder = Wav2Vec2Model.from_pretrained(self.hparams.encoder)
        self._freeze_encoder()

        # NOTE: Adjust the size of input.
        self.adjust = nn.Sequential(OrderedDict([
            ('layer_norm', nn.LayerNorm(768)),
            ('projection', nn.Linear(768, 400)),
            ('dropout', nn.Dropout(0.1)),
        ]))

        self.generator = WaveGradGenerator()
        self.discriminator = MelGANDiscriminator()

        self.multi_resoultion_stft_loss = MultiResolutionSTFTLoss(fft_sizes=[2048, 1024, 512, 256, 128, 64],
                                                                  hop_sizes=[256, 128, 32, 16, 8, 4],
                                                                  win_lengths=[1024, 512, 256, 128, 32, 16])

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.encoder.eval()

    def _loss_g(self, real, fake):
        x_fake_g = self.discriminator(fake)
        x_real_g = self.discriminator(real)

        adv_loss = 0.0
        for (feats_fake, score_fake), (feats_real, _) in zip(x_fake_g, x_real_g):
            adv_loss += th.mean(th.sum(th.pow(score_fake - 1.0, 2), dim=[1, 2]))
            for feat_f, feat_r in zip(feats_fake, feats_real):
                adv_loss += th.mean(th.abs(feat_f - feat_r))
        stft_loss = self.multi_resoultion_stft_loss(fake, real)
        loss_g = stft_loss + self.hparams.alpha * adv_loss

        self.log(f'loss_g_adv', adv_loss, on_step=True, prog_bar=False)
        self.log(f'loss_g_stft', stft_loss, on_step=True, prog_bar=False)
        self.log(f'loss_g', loss_g, on_step=True, prog_bar=True)

        return loss_g

    def _loss_d(self, real, fake):
        x_fake_d = self.discriminator(fake.detach())
        x_real_d = self.discriminator(real)

        loss_d = 0.0
        for (_, score_fake), (_, score_real) in zip(x_fake_d, x_real_d):
            loss_d += th.mean(th.sum(th.pow(score_real - 1.0, 2), dim=[1, 2]))
            loss_d += th.mean(th.sum(th.pow(score_fake, 2), dim=[1, 2]))

        self.log(f'loss_d', loss_d, on_step=True, prog_bar=True)

        return loss_d

    def _optim(self, optim, loss, parameters):
        optim.zero_grad()
        self.manual_backward(loss)
        nn.utils.clip_grad_norm_(parameters, 1.0)  # NOTE: temporary as lightning has problems for the moment.
        optim.step()

    def shared_step(self, batch):
        waveform, waveform_mask = batch
        with th.no_grad():
            if self.hparams.normalize:
                waveform = zero_mean_unit_var_norm(waveform)

            sine_exc, _, _, _ = compute_sine_exc(waveform)
            sine_exc = zero_mean_unit_var_norm(sine_exc, dim=1, keepdim=True)

            loudness = compute_loudness(waveform)
            loudness = zero_mean_unit_var_norm(loudness, dim=1, keepdim=True)

            attention_mask = waveform_mask if self.hparams.attention_mask else None
            x = self.encoder(waveform, attention_mask=attention_mask).last_hidden_state

        x = self.adjust(x)
        x_hat = self.generator(x, loudness, loudness)

        if self.training:
            optim_g, optim_d = self.optimizers(use_pl_optimizer=True)

        waveform = waveform.unsqueeze(1)

        loss_g = self._loss_g(waveform, x_hat)
        if self.training:
            self._optim(optim_g, loss_g, self.generator.parameters())

        if self.global_step >= 100_000:
            loss_d = self._loss_d(waveform, x_hat)
            if self.training:
                self._optim(optim_d, loss_d, self.discriminator.parameters())

    def training_step(self, batch, _, optimizer_idx):
        return self.shared_step(batch)

    def validation_step(self, batch, optimizer_idx):
        return self.shared_step(batch)

    def test_step(self, batch, optimizer_idx):
        return self.shared_step(batch)

    def configure_optimizers(self):
        optim_g = Adam(self.generator.parameters(), lr=0.001)
        scheduler_g = StepLR(optim_g, step_size=100_000, gamma=0.5)

        optim_d = Adam(self.discriminator.parameters(), lr=0.001)
        scheduler_d = StepLR(optim_d, step_size=100_000, gamma=0.5)

        return [optim_g, optim_d], [scheduler_g, scheduler_d]

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser, ], add_help=False)

        parser.add_argument('-a', '--alpha', default=2.5, type=float)
        parser.add_argument('-e', '--encoder', default='facebook/wav2vec2-base-960h', type=str)
        parser.add_argument('-n', '--normalize', action='store_true')
        parser.add_argument('-am', '--attention_mask', action='store_true')

        return parser
