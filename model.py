"""Baseline: log-mel spectrogram + small CNN (the 'old-style' baseline model)."""
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm

def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class LogMel(nn.Module):
    """waveform (B, T) -> normalised log-mel image (B, 1, 64, frames). Runs on CPU."""
    def __init__(self, sr=16000, n_mels=64):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=512, win_length=400, hop_length=160, n_mels=n_mels)

    def forward(self, wav):
        x = torch.log(self.mel(wav) + 1e-6)
        x = (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-5)
        return x.unsqueeze(1)

def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(_block(1, 16), _block(16, 32), _block(32, 64), _block(64, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):                       # returns logits (B,)
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(x).squeeze(1)

@torch.no_grad()
def predict(model, feat, loader, device):
    """returns (labels, fake-probabilities)"""
    model.eval()
    labels, scores = [], []
    for wav, y in tqdm(loader, leave=False):
        x = feat(wav).to(device)
        scores += torch.sigmoid(model(x)).cpu().tolist()
        labels += y.tolist()
    return labels, scores
