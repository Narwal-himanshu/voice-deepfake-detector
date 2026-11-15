"""Local API for the voice deepfake detector (supports v1 and v3 checkpoints).

Run:
  python3 -m uvicorn app:app --port 8000     then open http://localhost:8000

Optional environment variables:
  CKPT=checkpoints/xlsr_v3.pt           model checkpoint (v1 or v3)
  CALIB=eval/calibration_xlsr_v3.json   temperature + thresholds from v3
  DEVICE=mps|cuda|cpu                   inference device (default: cpu)
  LOW / HIGH                            manual thresholds (override calibration)
  WINDOW=4.0 HOP=2.0 MAX_WINDOWS=8      multi-window inference for long clips
  MIN_RMS_DB=-55 REL_RMS_DB=25          skip silent/quiet windows
  REAL_MAX=0.5 FAKE_MIN=0.8             demo band: real below, fake above
  DEMO_BAND=0                           set to 0 to use calibrated thresholds only
"""
import io
import os
from math import gcd

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

SR = 16000
WINDOW = int(float(os.environ.get("WINDOW", "4.0")) * SR)
HOP = max(1, int(float(os.environ.get("HOP", "2.0")) * SR))
MAX_WINDOWS = int(os.environ.get("MAX_WINDOWS", "8"))
MIN_RMS_DB = float(os.environ.get("MIN_RMS_DB", "-55"))
REL_RMS_DB = float(os.environ.get("REL_RMS_DB", "25"))
REAL_MAX = float(os.environ.get("REAL_MAX", "0.5"))
FAKE_MIN = float(os.environ.get("FAKE_MIN", "0.8"))
DEMO_BAND = os.environ.get("DEMO_BAND", "1") != "0"

CKPT = os.environ.get("CKPT", "checkpoints/xlsr_aasist_rb4.pt")
CALIB = os.environ.get("CALIB", "")
DEVICE = os.environ.get("DEVICE", "cpu")
MAX_BYTES = 20 * 1024 * 1024

LOW = float(os.environ.get("LOW", "1.2e-4"))
HIGH = float(os.environ.get("HIGH", "0.5"))
TEMPERATURE = 1.0


def fix_length(x, n=WINDOW):
    if len(x) >= n:
        return x[:n]
    if len(x) == 0:
        return np.zeros(n, dtype=np.float32)
    return np.tile(x, int(np.ceil(n / len(x))))[:n]


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model():
    obj = load_checkpoint(CKPT)
    keys = list(obj.keys()) if isinstance(obj, dict) else []
    arch = obj.get("arch") if isinstance(obj, dict) else None
    is_v3 = arch == "xlsr_aasist_v3" or any("layer_logits" in str(k) for k in keys)
    if is_v3:
        from model_v3 import load_model_v3
        return load_model_v3(CKPT, device=DEVICE).eval(), "v3"

    from transformers import Wav2Vec2Config, Wav2Vec2Model
    from xlsr_aasist import XLSR_AASIST

    cfg = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-xls-r-300m",
                                         layerdrop=0.0, apply_spec_augment=False)
    model = XLSR_AASIST(ssl=Wav2Vec2Model(cfg))
    state = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    model.load_state_dict(state)
    return model.to(DEVICE).eval(), "v1"


print("Loading model from", CKPT, "on", DEVICE, "...")
model, MODEL_VERSION = load_model()
if CALIB and os.path.exists(CALIB):
    import json
    with open(CALIB) as f:
        cal = json.load(f)
    TEMPERATURE = float(cal.get("temperature", 1.0))
    LOW = float(cal.get("real_thr", LOW))
    HIGH = float(cal.get("fake_thr", HIGH))
    if DEMO_BAND and MODEL_VERSION == "v3":
        LOW = min(LOW, REAL_MAX)
        HIGH = max(HIGH, FAKE_MIN)
        print(f"Demo band: LOW={LOW:.4f} HIGH={HIGH:.4f}")
    print(f"Calibration: T={TEMPERATURE:.3f} LOW={LOW:.4f} HIGH={HIGH:.4f}")
print("Model ready:", MODEL_VERSION)

app = FastAPI(title="Voice Deepfake Detector")


@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE, "model": MODEL_VERSION,
            "real_thr": LOW, "fake_thr": HIGH}


def window_starts(n):
    if n <= WINDOW:
        return [0]
    starts = list(range(0, n - WINDOW + 1, HOP))
    if starts[-1] != n - WINDOW:
        starts.append(n - WINDOW)
    if len(starts) > MAX_WINDOWS:
        idx = np.linspace(0, len(starts) - 1, MAX_WINDOWS).round().astype(int)
        starts = [starts[i] for i in sorted(set(idx.tolist()))]
    return starts


def aggregate(logits):
    z = np.asarray(logits, dtype=np.float64) / max(TEMPERATURE, 1e-6)
    mode = os.environ.get("AGG", "logit_mean")
    if mode == "max":
        return float(1.0 / (1.0 + np.exp(-z.max())))
    if mode == "prob":
        return float(np.mean(1.0 / (1.0 + np.exp(-z))))
    return float(1.0 / (1.0 + np.exp(-z.mean())))


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    data = file.file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File too large (max 20 MB).")
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32")
    except Exception:
        raise HTTPException(400, "Could not read this audio. Use WAV or FLAC.")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if len(x) < 0.5 * sr:
        raise HTTPException(400, "Audio is too short (need at least 0.5 second).")
    if sr != SR:
        g = gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype("float32")

    wins = []
    for s in window_starts(len(x)):
        w = fix_length(x[s:s + WINDOW])
        rms_db = 20 * np.log10(np.sqrt(np.mean(np.square(w))) + 1e-10)
        wins.append((w, rms_db))

    loudest = max(db for _, db in wins)
    logits = []
    with torch.no_grad():
        for w, rms_db in wins:
            if rms_db < MIN_RMS_DB or rms_db < loudest - REL_RMS_DB:
                continue
            t = torch.from_numpy(np.ascontiguousarray(w)).unsqueeze(0).to(DEVICE)
            logits.append(model(t).float().item())

    if not logits:
        raise HTTPException(400, "No speech detected. Record at least 4 seconds and start speaking right away.")
    score = aggregate(logits)
    if score >= HIGH:
        result = "fake"
    elif score < LOW:
        result = "real"
    else:
        result = "uncertain"
    return {"result": result, "score": score, "uncertain": result == "uncertain",
            "n_windows": len(logits), "model": MODEL_VERSION,
            "real_thr": LOW, "fake_thr": HIGH}


