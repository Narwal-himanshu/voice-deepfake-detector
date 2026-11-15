"""
Wrapper around the official RawBoost.py (Tak et al., ICASSP 2022,
github.com/TakHemlata/SSL_Anti-spoofing). Parameters = official defaults.
algo: 0 none | 1 LnL | 2 ISD | 3 SSI | 4 (1+2+3) | 5 (1+2) | 6 (1+3) | 7 (2+3) | 8 parallel(1,2)
"""
import numpy as np
from RawBoost import LnL_convolutive_noise, ISD_additive_noise, SSI_additive_noise, normWav

P = dict(nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000, minCoeff=10, maxCoeff=100,
         minG=0, maxG=0, minBias=5, maxBias=20, N_f=5, P=10, g_sd=2, SNRmin=10, SNRmax=40)
FS = 16000

def _lnl(x):
    return LnL_convolutive_noise(x, P["N_f"], P["nBands"], P["minF"], P["maxF"], P["minBW"], P["maxBW"],
                                 P["minCoeff"], P["maxCoeff"], P["minG"], P["maxG"], P["minBias"], P["maxBias"], FS)

def _isd(x):
    return ISD_additive_noise(x, P["P"], P["g_sd"])

def _ssi(x):
    return SSI_additive_noise(x, P["SNRmin"], P["SNRmax"], P["nBands"], P["minF"], P["maxF"], P["minBW"],
                              P["maxBW"], P["minCoeff"], P["maxCoeff"], P["minG"], P["maxG"], FS)

def rawboost(x, algo):
    x = np.asarray(x, dtype=np.float64)
    if algo == 0:
        return x.astype(np.float32)
    if algo == 1:   y = _lnl(x)
    elif algo == 2: y = _isd(x)
    elif algo == 3: y = _ssi(x)
    elif algo == 4: y = _ssi(_isd(_lnl(x)))
    elif algo == 5: y = _isd(_lnl(x))
    elif algo == 6: y = _ssi(_lnl(x))
    elif algo == 7: y = _ssi(_isd(x))
    elif algo == 8: y = normWav(_lnl(x) + _isd(x), 0)
    else: raise ValueError("algo must be 0..8")
    y = np.nan_to_num(np.asarray(y, dtype=np.float64).reshape(-1))
    return y.astype(np.float32)
