"""Run first: python check_setup.py"""
import torch, torchaudio, soundfile
print("torch:", torch.__version__)
print("torchaudio:", torchaudio.__version__)
print("MPS (Apple GPU) available:", torch.backends.mps.is_available())
dev = "mps" if torch.backends.mps.is_available() else "cpu"
x = torch.randn(2, 3, device=dev)
print("Test tensor on", dev, "->", x.shape)
print("Setup OK")
