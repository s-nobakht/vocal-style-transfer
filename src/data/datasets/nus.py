import glob
import os
from pathlib import Path
from typing import Tuple, Union

import gdown
import soundfile as sf
import torch as th
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import extract_archive


URL = 'https://drive.google.com/uc?id=1RZymzC0cM6g2QzfmYZtLddrbg8Sh5AGj'


class NUS_48E(Dataset):
    def __init__(self,
                 root: Union[str, Path],
                 url: str = URL,
                 download: bool = False) -> None:

        archive = os.path.join(root, 'nus-smc-corpus_48.zip')

        self._path = os.path.join(root, 'nus-smc-corpus_48')

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    gdown.download(url, archive, quiet=False)
                extract_archive(archive, root)

        self._flist = sorted(glob.glob(f'{self._path}/**/**/*.wav'))

    def __getitem__(self, n: int) -> Tuple[Tensor, int]:
        waveform, sample_rate = sf.read(self._flist[n])
        if waveform.ndim == 1:
            waveform = waveform.reshape(-1, 1)
        waveform = th.from_numpy(waveform).t().float()
        return (
            waveform,
            sample_rate,
        )

    def __len__(self) -> int:
        return len(self._flist)
