import glob
import os
import pickle
from multiprocessing import Pool

import torch as th
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchaudio.datasets import LJSPEECH, VCTK_092
from tqdm import tqdm

from .nus import NUS_48E
from ..utils import dummy_collate_fn, process_sample


class Wav2MelDataset(Dataset):
    def __init__(self,
                 root: str,
                 window_size: int = 16000,
                 download: bool = False,
                 num_processes: int = 1) -> None:

        self._root = f'{root}/Wav2Mel/{window_size}'
        index_path = f'{self._root}/index.pkl'

        if download and not os.path.exists(self._root):
            dataset = ConcatDataset([LJSPEECH(root, download=download),
                                     VCTK_092(root, download=download),
                                     NUS_48E(root, download=download)])
            dataloader = DataLoader(dataset,
                                    batch_size=num_processes,
                                    num_workers=num_processes,
                                    collate_fn=dummy_collate_fn,
                                    pin_memory=False,
                                    drop_last=False)

            os.makedirs(self._root, exist_ok=True)

            effects = [['remix', '1'], ['rate', '16000']]
            with Pool(num_processes) as p:
                for i, batch in enumerate(tqdm(dataloader)):
                    args = [
                        (j, waveform, sample_rate, effects, window_size, self._root)
                        for j, (waveform, sample_rate, *_) in enumerate(batch, i * num_processes)
                    ]
                    p.starmap(process_sample, args)

            index = []
            for path in sorted(glob.glob(f'{self._root}/*.pt')):
                fn = path.split('/')[-1]
                index.append((fn, fn.split('_')[-1][:-3]))

            with open(index_path, 'wb') as f:
                pickle.dump(index, f)

        with open(index_path, 'rb') as f:
            self._index = pickle.load(f)

    def __getitem__(self, idx):
        fn, waveform_len = self._index[idx]
        waveform = th.load(f'{self._root}/{fn}')
        waveform_mask = th.ones(int(waveform_len))
        return waveform, waveform_mask

    def __len__(self) -> int:
        return len(self._index)
