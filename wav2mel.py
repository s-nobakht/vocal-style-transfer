import os
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TestTubeLogger

from src.data.datamodule import Wave2MelDataModule
from src.models import Wav2MelGradModule


def _trainer(hparams: Namespace) -> Trainer:
    logger = TestTubeLogger('./logs', name='wav2mel')
    checkpoint_callback = ModelCheckpoint('./checkpoints/wav2mel',
                                          monitor='loss_g_epoch',
                                          verbose=True,
                                          save_top_k=1,
                                          mode='min',
                                          prefix=str(os.getenv('SLURM_JOB_ID', '')))
    callbacks = [LearningRateMonitor(), ]
    if hparams.patience >= 0:
        callbacks.append(EarlyStopping('loss_g_epoch',
                                       mode='min',
                                       patience=hparams.patience,
                                       verbose=True))
    trainer = Trainer.from_argparse_args(hparams,
                                         logger=logger,
                                         checkpoint_callback=checkpoint_callback,
                                         callbacks=callbacks)
    return trainer


def _hparams() -> Namespace:
    parser = ArgumentParser(add_help=False)

    parser.add_argument('-s', '--seed', default=-1, type=int)
    parser.add_argument('-p', '--patience', default=-1, type=int)

    parser = Trainer.add_argparse_args(parser)
    parser = Wave2MelDataModule.add_argparse_args(parser)
    parser = Wav2MelGradModule.add_argparse_args(parser)

    hparams = parser.parse_args()

    return hparams


def main() -> None:
    hparams = _hparams()

    if hparams.seed >= 0:
        pl.seed_everything(hparams.seed)

    trainer = _trainer(hparams)

    datamodule = Wave2MelDataModule(hparams)
    model = Wav2MelGradModule(hparams)
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, test_dataloaders=datamodule.test_dataloader())


if __name__ == '__main__':
    main()
