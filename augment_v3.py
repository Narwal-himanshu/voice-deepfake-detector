"""V3 waveform augmentation.

Goals:
1. Break dataset shortcuts that made v1/v2 useless on MLAAD:
   - loudness  -> RMS-normalise every clip to a fixed level
   - noise floor (MLAAD real -95 dB vs fake -64 dB) -> raise the floor of every
     training clip to a random target, so the floor carries no class information
   - silence ratio -> random crop offsets + RawBoost noise
2. Learn general spoofing cues instead of one recording style:
   - RawBoost with a RANDOM algorithm per clip (not a single fixed one)
   - additive noise at a random SNR (not a fixed absolute floor)
   - light synthetic reverb and telephone/codec simulation

Everything is numpy/scipy and seeded from a np.random.Generator, so it is safe
to use inside DataLoader workers.
"""
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve, resample_poly

from rawboost_aug import rawboost

SR = 16000

DEFAULT_TRAIN_AUG = dict(
    normalize_rms=True,
    target_rms_db=-23.0,
    floor_prob=0.5,
    floor_lo=-62.0,
    floor_hi=-45.0,
    rawboost_prob=0.7,
    rawboost_algos=(4, 5, 8),
    noise_prob=0.25,
    snr_lo=10.0,
    snr_hi=35.0,
    reverb_prob=0.15,
    codec_prob=0.20,
)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-10))


def normalize_rms(x, target_db=-23.0, peak_limit=0.99):
    g = 10 ** (target_db / 20.0) / (rms(x) + 1e-10)
    peak = float(np.max(np.abs(x)) + 1e-10)
    g = min(g, peak_limit / peak)
    return (x * g).astype(np.float32)


def frame_db(x, n=400):
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    n = max(n, 1)
    fr = x[: len(x) // n * n].reshape(-1, n)
    return 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-10)


def noise_floor_db(x):
    return float(np.percentile(frame_db(x), 10))


def equalize_floor(x, rng, lo=-62.0, hi=-45.0, prob=0.5):
    """Raise the quiet parts to a RANDOM target floor (same range for both classes)."""
    if rng.random() > prob:
        return x
    target = float(rng.uniform(lo, hi))
    if noise_floor_db(x) >= target:
        return x
    noise = rng.standard_normal(len(x)).astype(np.float32)
    noise /= np.sqrt(np.mean(noise ** 2)) + 1e-10
    return (x + noise * (10 ** (target / 20.0))).astype(np.float32)


def add_noise_snr(x, rng, snr_lo=10.0, snr_hi=35.0, color="white"):
    snr = float(rng.uniform(snr_lo, snr_hi))
    noise = rng.standard_normal(len(x)).astype(np.float32)
    if color == "pink":
        sos = butter(2, 3000, btype="lowpass", fs=SR, output="sos")
        noise = sosfilt(sos, noise).astype(np.float32)
    sig_p = np.mean(np.square(x)) + 1e-10
    noise_p = np.mean(np.square(noise)) + 1e-10
    noise *= np.sqrt(sig_p / (noise_p * (10 ** (snr / 10.0))))
    return (x + noise).astype(np.float32)


def synthetic_reverb(x, rng, t60_lo=0.10, t60_hi=0.55, wet_lo=0.10, wet_hi=0.45):
    t60 = float(rng.uniform(t60_lo, t60_hi))
    n_rir = max(8, int(t60 * SR * 1.5))
    t = np.arange(n_rir) / SR
    rir = rng.standard_normal(n_rir) * np.exp(-6.9 * t / max(t60, 1e-3))
    rir[0] = 1.0
    rir /= np.sqrt(np.sum(rir ** 2)) + 1e-10
    y = fftconvolve(x, rir.astype(np.float32), mode="full")[: len(x)].astype(np.float32)
    wet = float(rng.uniform(wet_lo, wet_hi))
    return ((1 - wet) * x + wet * y).astype(np.float32)


def telephone_codec(x, rng):
    kind = int(rng.integers(0, 3))
    if kind == 0:  # mu-law companding
        mu = 255.0
        y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
        return y.astype(np.float32)
    if kind == 1:  # 8 kHz round trip
        y = resample_poly(x, 8000, SR)
        return resample_poly(y, SR, 8000).astype(np.float32)
    sos = butter(4, [300, 3400], btype="bandpass", fs=SR, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def random_crop(x, rng, n):
    """Random 4 s crop (training). Falls back to tiling for very short clips."""
    if len(x) >= n:
        start = int(rng.integers(0, len(x) - n + 1))
        return x[start:start + n]
    if len(x) == 0:
        return np.zeros(n, dtype=np.float32)
    return np.tile(x, int(np.ceil(n / len(x))))[:n]


def center_crop(x, n):
    if len(x) >= n:
        start = (len(x) - n) // 2
        return x[start:start + n]
    if len(x) == 0:
        return np.zeros(n, dtype=np.float32)
    return np.tile(x, int(np.ceil(n / len(x))))[:n]


def apply_rawboost(x, rng, algos=(4, 5, 8)):
    algo = int(algos[int(rng.integers(0, len(algos)))])
    np.random.seed(int(rng.integers(0, 2 ** 32 - 1)))  # RawBoost uses the global RNG
    return rawboost(x, algo)


def augment_train(x, rng, cfg=None):
    c = dict(DEFAULT_TRAIN_AUG)
    if cfg:
        c.update(cfg)

    x = np.asarray(x, dtype=np.float32)
    if c.get("normalize_rms", True):
        x = normalize_rms(x, c["target_rms_db"])

    x = equalize_floor(x, rng, c["floor_lo"], c["floor_hi"], c["floor_prob"])

    if rng.random() < c["rawboost_prob"]:
        x = apply_rawboost(x, rng, tuple(c["rawboost_algos"]))

    if rng.random() < c["noise_prob"]:
        x = add_noise_snr(x, rng, c["snr_lo"], c["snr_hi"])

    if rng.random() < c["reverb_prob"]:
        x = synthetic_reverb(x, rng)

    if rng.random() < c["codec_prob"]:
        x = telephone_codec(x, rng)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(x)) + 1e-10)
    if peak > 1.0:
        x = x / peak
    return x.astype(np.float32)
