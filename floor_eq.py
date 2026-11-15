"""Simple audio features + noise-floor equalisation (removes the 'clean recording' shortcut)."""
import numpy as np
import soundfile as sf
from scipy.signal import welch

SR = 16000

def load4(path):
    x, sr = sf.read(path, frames=4 * SR, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    assert sr == SR, f"{path}: sample rate {sr}"
    return x

def _frame_db(x, n=400):
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    fr = x[: len(x) // n * n].reshape(-1, n)
    return 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-10)

def noise_floor(x):
    return float(np.percentile(_frame_db(x), 10))

def equalize_floor(x, target_db, rng):
    """If the quiet parts are quieter than target_db, add white noise so they sit at target_db."""
    if noise_floor(x) >= target_db:
        return x
    noise = rng.randn(len(x)).astype(np.float32) * (10 ** (target_db / 20))
    return x + noise

def feats(x):
    fe = _frame_db(x)
    f, p = welch(x, fs=SR, nperseg=512)
    tot = p.sum() + 1e-10
    cum = np.cumsum(p) / tot
    return {
        "floor": float(np.percentile(fe, 10)),
        "silence": float(np.mean(fe < fe.max() - 40)),
        "rms": float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-10)),
        "peak": float(20 * np.log10(np.max(np.abs(x)) + 1e-10)),
        "rolloff": float(f[min(np.searchsorted(cum, 0.95), len(f) - 1)]),
        "hf_4_8k": float(p[f >= 4000].sum() / tot),
    }


def randomize_style(x, rng, floor_lo=-50.0, floor_hi=-40.0, gain_db=10.0):
    """Hide recording-style shortcuts: random loudness, then raise the noise floor to a RANDOM level."""
    x = x * (10 ** (rng.uniform(-gain_db, gain_db) / 20))
    return equalize_floor(x, rng.uniform(floor_lo, floor_hi), rng)
