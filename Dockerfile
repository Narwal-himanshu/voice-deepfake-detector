FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CKPT=/app/checkpoints/xlsr_v3.pt \
    CALIB=/app/eval/calibration_xlsr_v3.json \
    DEVICE=cpu \
    DEMO_BAND=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app.py index.html model_v3.py ./
COPY eval/calibration_xlsr_v3.json ./eval/calibration_xlsr_v3.json

RUN mkdir -p /app/checkpoints \
    && curl -fL --retry 5 \
        https://github.com/Narwal-himanshu/voice-deepfake-detector/releases/download/v3.0/xlsr_v3.pt \
        -o /app/checkpoints/xlsr_v3.pt

EXPOSE 10000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}"]
