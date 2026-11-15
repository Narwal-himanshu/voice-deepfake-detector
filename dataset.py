"""PyTorch Dataset: loads audio, makes it 16 kHz mono, fixed 4 sec."""
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from rawboost_aug import rawboost

SR = 16000
MAX_LEN = 4 * SR  # 64000 samples

def fix_length(x: np.ndarray, n: int = MAX_LEN) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    reps = int(np.ceil(n / len(x)))       # repeat short clips to fill 4 sec
    return np.tile(x, reps)[:n]

class AudioDataset(Dataset):
    def __init__(self, csv_path: str, algo: int = 0, prob: float = 0.5):
        """algo>0 turns on RawBoost augmentation (use ONLY for training data).
        prob = fraction of clips that get augmented."""
        self.df = pd.read_csv(csv_path)
        self.algo, self.prob = algo, prob

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav, sr = sf.read(row["path"], dtype="float32")
        if wav.ndim > 1:                   # stereo -> mono
            wav = wav.mean(axis=1)
        if sr != SR:
            wav = torchaudio.functional.resample(torch.from_numpy(wav), sr, SR).numpy()
        wav = fix_length(wav)
        if self.algo > 0 and np.random.rand() < self.prob:
            wav = rawboost(wav, self.algo)
        return torch.from_numpy(wav), torch.tensor(row["label"], dtype=torch.float32)
