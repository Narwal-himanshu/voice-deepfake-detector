# Model Card: XLS-R + AASIST-style voice deepfake detector

## Summary
Binary classifier on 4-second, 16 kHz mono audio. Output = probability the clip is fake (label 1). Research / educational demo. Two checkpoints are supported: **v1** (last XLS-R layer) and **v3** (all-layer fusion + multi-dataset training). v1 results come first; v3 results are at the bottom.

## Intended use
Learning and demonstration of anti-spoofing models. **Not** for deciding whether a real person's voice is genuine, for legal or forensic use, fraud decisions, or moderation.

## Architecture and training
- Front-end: `facebook/wav2vec2-xls-r-300m`, fine-tuned (CNN feature encoder frozen). 311.6M trainable parameters in total.
- Back-end: simplified AASIST-style graph attention (own re-implementation, not official code).
- Loss BCE; AdamW; lr 1e-5 (XLS-R) / 1e-4 (back-end), cosine schedule with 5% warm-up; batch 16; 8 epochs; mixed precision; seed 1.
- Augmentation: RawBoost algo 4 (LnL + ISD + SSI), applied to 70% of training clips.
- Data: 4000-file balanced subset of ASVspoof 2019 LA train. Checkpoint chosen by lowest dev EER on 1000 LA dev files.

## Results: v1 (single run)

| Test set | n | EER % | AUC | F1 @0.5 |
|---|---|---|---|---|
| LA dev | 1000 | 0.60 | 1.000 | 0.971 |
| LA eval | 2000 | 1.00 | 0.997 | 0.766 |
| MLAAD all | 4000 | 46.60 | 0.523 | 0.087 |
| MLAAD matched | 450 | 45.78 | 0.536 | 0.172 |
| MLAAD German | 2000 | 47.50 | 0.515 | 0.089 |
| MLAAD English | 2000 | 45.60 | 0.531 | 0.086 |

Comparison models (same data, same metrics):

| Model | LA eval EER / AUC | MLAAD all EER / AUC | MLAAD matched EER / AUC |
|---|---|---|---|
| Small CNN | 5.30 / 0.986 | 63.0 / 0.335 | 56.2 / 0.392 |
| Small CNN + RawBoost algo 5 | 6.10 / 0.981 | 61.4 / 0.356 | not measured |
| Small CNN + RawBoost algo 4 | 7.00 / 0.980 | 34.1 / 0.716 | 45.8 / 0.549 |

EER is threshold-free. F1 at 0.5 is low for XLS-R on LA eval because the model is under-confident on many fakes (see below), so EER and F1 tell different stories.

## v1 behaviour at threshold 0.5 (LA eval, 2000 files)
Real clips correct: 99.7%. Fake clips caught: 62.3%.

| Attack | n | % caught | median score |
|---|---|---|---|
| A07 | 70 | 78.6 | 0.9931 |
| A08 | 78 | 64.1 | 0.8979 |
| A09 | 74 | 100.0 | 0.9997 |
| A10 | 84 | 2.4 | 0.0003 |
| A11 | 85 | 22.4 | 0.0164 |
| A12 | 88 | 38.6 | 0.0323 |
| A13 | 84 | 100.0 | 0.9988 |
| A14 | 76 | 56.6 | 0.8528 |
| A15 | 84 | 14.3 | 0.0013 |
| A16 | 62 | 87.1 | 0.9994 |
| A17 | 64 | 93.8 | 0.9982 |
| A18 | 79 | 89.9 | 0.9979 |
| A19 | 72 | 90.3 | 0.9983 |

A cut-off of 1.18e-4, chosen on one random half of LA eval, gave 99% real and 99% fake correct on the other half (the reverse direction was not run). But my own real voice scored 1e-3 to 4e-4 and AI voices around 3e-4 to 3e-3, i.e. all above that cut-off. So a low cut-off only fits ASVspoof-style recordings.

## v1 demo verdicts
score < 1.2e-4 -> real; score >= 0.5 -> fake; otherwise "uncertain". The lower cut-off came from LA eval (see above), so it is not a clean held-out choice.

## Known problems and risks
- **v1 does not generalize to modern TTS** (MLAAD): AUC about 0.52-0.54 means no useful signal there. On MLAAD the median fake score is near 0 for almost every TTS system (real clips: 0.7% flagged, mean score 0.01). Only a few systems were partly flagged at 0.5: griffin_lim 20%, RVC 25%, suno_bark 25%, e2-tts 25%, suno_bark-small 44% (16 to 83 clips each; small samples).
- **v3 is much stronger but not shortcut-free:** ASV eval 0.25% EER / 0.9999 AUC; MLAAD unseen-system 2.60% / 0.998; matched 8.00% / 0.953. On the unseen set a 6-feature recording-style classifier still reaches AUC 0.952 (noise floor alone 0.932), so the unseen number is partly confounded. On matched the shortcut drops to 0.631 while v3 reaches 0.953, but matched shares 37 fake systems with training. There is no clean matched+unseen test yet.
- **Dataset shortcuts found:** MLAAD noise floor separates real from fake (AUC 0.916); in ASVspoof, real clips contain more silence than fakes (silence-ratio AUC 0.295, i.e. 0.705 reversed). Models can use such shortcuts instead of real fake-voice cues.
- Only subsets; one seed. v3 is a single run; matched has 225 pairs; the v3 calibration is sharp (T=0.362) and its uncertain band collapsed. The demo uses a cautious `0.5–0.8` band, so borderline live-mic audio is reported as NOT SURE instead of real/fake.
- Training/eval use a 4-second center crop; the demo analyzes up to eight 4-second windows over the full clip and skips near-silent windows. Clips shorter than 4 s are repeated to fill. Pure/near-silent audio is rejected.
- Not tested on live microphone/laptop recordings, compressed or phone audio, music, other languages, children's or accented voices. False "fake" or "real" answers on such audio are possible and unmeasured; a real voice can be flagged fake because the microphone/room is out-of-domain.
- Using audio of real people carries privacy risks; datasets used: ASVspoof 2019 LA, MLAAD-tiny (CC BY-NC 4.0, non-commercial use only).

## v3 pipeline (trained)

v1 results are above; v3 is a separate, stronger checkpoint. v3 was trained for 10 epochs on a Colab T4 (the session ended before epochs 11-12; best checkpoint = epoch 10) and evaluated locally on Apple M2 MPS.

- Front-end: all 25 XLS-R hidden states fused with learnable softmax weights (not just the last layer), then the same AASIST-style back-end.
- Training data: 13,200 balanced clips from ASVspoof 2019 LA and MLAAD-tiny, with MLAAD TTS systems disjoint between train/dev/test.
- Shortcut controls: RMS to -23 dB, random noise-floor equalisation, noise at 10-35 dB SNR, reverb, telephone codec, RawBoost algos 4/5/8.
- Loss: focal (gamma 2, alpha 0.25). Checkpoint selected by mean AUC over ASV dev and MLAAD dev.
- Calibration: temperature scaling plus real/fake thresholds fitted on the dev mix; `T=0.3622`, `real_thr = fake_thr = 0.5394` (uncertain band collapsed). The demo uses a cautious `0.5–0.8` band for live microphone safety (`REAL_MAX=0.5 FAKE_MIN=0.8`).

### v3 results (one seed)

| Test set | n | EER % | AUC | balanced acc | real acc | fake acc |
|---|---:|---:|---:|---:|---:|---:|
| ASVspoof LA eval | 2000 | 0.25 | 0.9999 | 0.9965 | 0.998 | 0.995 |
| MLAAD all | 4000 | 1.67 | 0.9984 | 0.9828 | 0.996 | 0.9695 |
| MLAAD matched | 450 | 8.00 | 0.9526 | 0.9222 | 0.964 | 0.880 |
| MLAAD unseen-system | 2000 | 2.60 | 0.9977 | 0.9760 | 0.999 | 0.953 |

### v3 shortcut audit

| Test set | noise-floor AUC | 6-feature GBM AUC | v3 AUC |
|---|---:|---:|---:|
| MLAAD unseen | 0.932 | 0.952 | 0.998 |
| MLAAD matched | 0.501 | 0.631 | 0.953 |

The unseen set is system-disjoint, but its recording-style shortcut remains strong (GBM 0.952), so v3's 2.60% EER is partly confounded. The matched set removes the noise-floor shortcut (0.501) and the GBM drops to 0.631 while v3 reaches 0.953 AUC; however, matched shares 37 fake systems with training. A clean matched **and** unseen-system test is still missing.

Reproduce with `colab_v3.ipynb` (T4 GPU) for training and `evaluate_v3.py` for the full local eval. Validate the pipeline locally with `python3 selftest_v3.py`.

These are one-seed results on subsets. Do not claim real-world or forensic generalization; next steps are an external dataset, ASVspoof 5, multiple seeds, and a matched unseen-system test.

## Citations
- ASVspoof 2019 LA dataset.
- MLAAD: `mueller91/MLAAD-tiny` on Hugging Face.
- Tak et al., RawBoost, ICASSP 2022; SSL_Anti-spoofing repo (TakHemlata).
- AASIST (Jung et al.) and XLS-R (Babu et al.) as inspiration / front-end; this repo's back-end is a simplified re-implementation.
