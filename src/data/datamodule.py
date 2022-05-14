from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from .datasets import Wav2MelDataset
from .utils import collate_fn


class Wave2MelDataModule(pl.LightningDataModule):
    def __init__(self, hparams: Namespace) -> None:
        super().__init__()
        self.hparams = hparams

    def setup(self, _) -> None:
        dataset = Wav2MelDataset(self.hparams.root)
        train_size = int(len(dataset) * 0.9)
        val_size = (len(dataset) - train_size) // 2
        test_size = len(dataset) - train_size - val_size
        self.train, self.val, self.test = random_split(dataset, [train_size, val_size, test_size])

    def prepare_data(self) -> None:
        try:
            Wav2MelDataset(self.hparams.root, download=True, num_processes=self.hparams.num_workers)
        except Exception:
            pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train,
                          batch_size=self.hparams.batch_size,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          collate_fn=collate_fn,
                          pin_memory=True,
                          drop_last=False)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          collate_fn=collate_fn,
                          pin_memory=True,
                          drop_last=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          collate_fn=collate_fn,
                          pin_memory=True,
                          drop_last=False)

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser, ], add_help=False)

        parser.add_argument('-bs', '--batch_size', default=32, type=int)
        parser.add_argument('-w', '--num_workers', default=4, type=int)
        parser.add_argument('-r', '--root', default='./data', type=str)

        return parser
