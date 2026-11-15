# Voice Deepfake Detector — Plain-English Guide

This guide explains the project to someone who has never seen it before: what it does, how it was built, what the results mean, what went wrong, and what every file is for.

---

## 1. One-minute summary

I built a system that listens to a short voice clip and estimates whether it sounds **real** or **AI-generated**.

- The final model is called **v3**.
- On the benchmark data it does very well: **0.25% EER** on ASVspoof and **2.60% EER** on a set of AI voices from TTS systems it never saw during training.
- It is **not** a production or forensic tool. A live microphone recording is very different from benchmark audio, so the demo now says **NOT SURE** for borderline cases instead of pretending to be certain.
- The project went through three attempts: **v1** (worked in-domain but failed on modern AI voices), **v2** (a failed experiment), and **v3** (the current model).

The honest claim is:

> “This is a strong research/demo model on the test sets I used. It is not proof that a real person’s voice is fake, and it is not reliable on arbitrary microphone recordings.”

---

## 2. What problem is this?

A “deepfake voice” is audio generated or modified by AI to sound like a person. The detector tries to answer one question:

> Given a 4-second voice clip, is it a real human recording or a synthetic/spoofed one?

The model outputs a **score from 0 to 1**:

- `0.0` = looks very real
- `1.0` = looks very fake
- `0.5` = completely uncertain

In the app, the score is turned into `real`, `fake`, or `NOT SURE` using thresholds.

---

## 3. What the model actually does (simple version)

1. **Listen to the waveform.** Audio is converted to 16 kHz mono, and a 4-second piece is used.
2. **Extract speech features.** A large pre-trained speech model called **XLS-R 300M** turns the waveform into rich representations of the voice.
3. **Look at many layers.** In v3, the model combines information from all 25 hidden layers of XLS-R instead of only the last one.
4. **Classify.** An **AASIST-style** network looks at spectral and temporal patterns and produces a fake/real score.
5. **Calibrate.** A temperature and thresholds turn raw scores into usable decisions.
6. **Demo.** A FastAPI server and a browser page let a user upload or record a clip and see the result.

“AASIST-style” means I wrote a simplified version inspired by the AASIST anti-spoofing architecture. It is **not** the official AASIST code.

---

## 4. The three versions: v1, v2, v3

These are **experiment iterations**, not software releases.

### v1 — first working model

- XLS-R 300M, using only the last layer.
- AASIST-style back-end.
- RawBoost augmentation.
- Trained on ASVspoof 2019 LA.

**What happened:** It worked well on ASVspoof, but failed on modern TTS audio from MLAAD.

### v2 — failed experiment

- Tried adding random loudness and noise-floor changes to make the model ignore recording-style shortcuts.
- Trained on ASVspoof + MLAAD.

**What happened:** It did not learn. Dev EER stayed around 49% (basically guessing), and all scores looked almost identical. No v2 numbers or checkpoint were kept.

**Lesson:** Throwing augmentation at the problem was not enough. The architecture, data splitting, and training setup also had to change.

### v3 — current model

- Combines **all 25 XLS-R hidden layers** with learnable weights.
- Trained on **13,200 balanced clips** from ASVspoof 2019 LA + MLAAD-tiny.
- MLAAD TTS systems are split so the test systems are never seen in training.
- Shortcut-safe augmentation: loudness normalisation, random noise floor, background noise, reverb, telephone codec, RawBoost.
- Focal loss, checkpoint chosen by **both** ASV dev and MLAAD dev AUC.
- Trained for 10 epochs on a Colab T4 GPU.

**What happened:** Large improvement on both domains. This is the model used by the demo.

---

## 5. The data

### ASVspoof 2019 LA

- An older anti-spoofing challenge dataset.
- Real recordings and several older spoofing attacks.
- English, roughly 4-second clips.
- Useful for learning basic fake-voice cues.

### MLAAD-tiny

- A newer dataset of real and AI-generated speech.
- Multiple TTS systems (different AI voice generators).
- English and German.
- Much harder than ASVspoof because the fake voices are modern and varied.

### Why the data split matters

- **Train:** examples the model learns from.
- **Dev:** a practice test used to pick the best epoch and calibrate thresholds.
- **Test:** the final exam. The model never trains on these clips.

For MLAAD, I split by **TTS system**:

- Systems used for training.
- Held-out systems for dev.
- Completely unseen systems for the final unseen-system test.

This checks whether the model can catch a **new** AI voice generator it has never seen.

### Matched and unseen tests

- **MLAAD all:** all test clips.
- **MLAAD matched:** 225 real/fake pairs chosen so their background noise floors are similar. This removes an easy shortcut.
- **MLAAD unseen-system:** TTS systems that were not in training.

There is no perfect test that is both matched **and** unseen-system. That is a limitation.

---

## 6. How I made it, step by step

1. Downloaded ASVspoof 2019 LA and MLAAD-tiny.
2. Converted audio to 16 kHz mono and built CSV lists of `path`, `label`, `attack`, `source`.
3. Built train/dev/test splits, with MLAAD split by TTS system.
4. Trained small CNN baselines to get a reference point.
5. Trained v1 (XLS-R + AASIST) on ASVspoof and evaluated it.
6. Discovered v1 failed on MLAAD and found dataset shortcuts.
7. Tried v2 augmentation; it failed to learn.
8. Built v3 with all-layer fusion, multi-dataset training, stronger augmentation, focal loss, and better checkpoint selection.
9. Trained v3 on Colab, then evaluated the checkpoint locally on an Apple M2 GPU.
10. Built a FastAPI demo with upload/record support.
11. Found and fixed real-world demo problems: silence was scored as fake, the browser sent only the first 4 seconds, and v3 was loaded in training mode by mistake.
12. Added a cautious `NOT SURE` band for live microphone audio.

---

## 7. Challenges

### Challenge 1: Domain shift

The model learns from benchmark recordings. A live laptop microphone has a different room, microphone, background noise, and codec. This is called **domain shift**. A real voice can look “fake” to the model simply because the recording style is unfamiliar.

**What I did:** Added silence/quiet-window detection, a `NOT SURE` band, and honest UI warnings. The real fix would be personal calibration or more diverse training data.

### Challenge 2: Dataset shortcuts

The model can cheat. In MLAAD, real and fake clips had different background noise floors. A tiny classifier using only six simple numbers (noise floor, loudness, silence, etc.) could separate real from fake with **AUC 0.952** on the unseen systems. That means part of the “good” result may come from recording style, not from detecting AI voices.

**What I did:** Added random noise-floor equalisation and other augmentations during training, built a matched test set, and reported a shortcut audit next to the model results.

### Challenge 3: v2 did not learn

Adding random noise made training collapse. Dev EER stayed near 49%. I stopped v2 and redesigned the approach instead of tuning a broken run.

### Challenge 4: Compute and time

XLS-R is large. Training ran on a free Colab T4 GPU. The session disconnected after 10 epochs, before the planned 12. The best checkpoint was epoch 10, so I used that.

### Challenge 5: Calibration

A model can rank clips well but still give bad probabilities. EER and AUC measure ranking; the demo needs a threshold. I fitted temperature scaling and thresholds on the dev mix. The calibrated real/fake thresholds collapsed to the same value, so the demo adds a wider `NOT SURE` band for live audio.

### Challenge 6: Silence is not “real”

Pure silence was scored as **fake 0.999** by the model. That is a dangerous demo behavior. I fixed it by rejecting near-silent clips and skipping quiet windows.

### Challenge 7: One seed, small subsets

The results are from one training run and subsets of the datasets, not the full official evaluation. Small differences should not be over-interpreted.

---

## 8. Results and what the metrics mean

### v3 results (one seed)

| Test set | Clips | EER % | AUC | Balanced accuracy | Real accuracy | Fake accuracy |
|---|---:|---:|---:|---:|---:|---:|
| ASVspoof LA eval | 2,000 | 0.25 | 0.9999 | 0.9965 | 0.998 | 0.995 |
| MLAAD all | 4,000 | 1.67 | 0.9984 | 0.9828 | 0.996 | 0.9695 |
| MLAAD matched | 450 | 8.00 | 0.9526 | 0.9222 | 0.964 | 0.880 |
| MLAAD unseen-system | 2,000 | 2.60 | 0.9977 | 0.9760 | 0.999 | 0.953 |

### v1 results (for comparison)

| Test set | Clips | EER % | AUC |
|---|---:|---:|---:|
| ASVspoof LA eval | 2,000 | 1.00 | 0.997 |
| MLAAD all | 4,000 | 46.60 | 0.523 |
| MLAAD matched | 450 | 45.78 | 0.536 |

At a fixed threshold of 0.5, v1 caught only **62.3%** of ASVspoof fakes (real clips: 99.7% correct). That is why v1 looked good on some metrics but bad in practice.

### What is EER?

**EER = Equal Error Rate.** Imagine moving the real/fake cutoff. At one point, the percentage of real clips wrongly called fake equals the percentage of fake clips wrongly called real. That percentage is the EER.

- **0%** = perfect.
- **50%** = random guessing.
- Lower is better.

`0.25% EER` means: at the equal-error point, only 0.25% of real clips are wrongly rejected and 0.25% of fake clips are wrongly accepted.

### What is AUC?

**AUC = Area Under the ROC Curve.** It measures how well the model ranks fakes above reals across all possible thresholds.

- **0.5** = coin flip.
- **1.0** = perfect ranking.
- Higher is better.

`AUC 0.9999` means almost every fake receives a higher score than almost every real clip.

AUC can be high even when the default threshold is bad. That is why we also report accuracy at a chosen threshold.

### What is balanced accuracy?

The average of real accuracy and fake accuracy:

```text
balanced accuracy = (real accuracy + fake accuracy) / 2
```

It stops a model from looking good just because one class is easier.

### What is real/fake accuracy?

- **Real accuracy:** of all real clips, how many were called real.
- **Fake accuracy:** of all fake clips, how many were called fake.

### What is F1?

F1 combines **precision** and **recall**:

- Precision: of the clips called fake, how many really were fake?
- Recall: of all fake clips, how many did we catch?

F1 is useful when you care about both false alarms and missed fakes. v1 had a low F1 at threshold 0.5 even though its EER was good, because many fake scores sat below 0.5. The ranking was good; the default cutoff was not.

### What is calibration and temperature?

A raw model score is not automatically a real probability. **Temperature scaling** adjusts the scores so they behave better as probabilities. The fitted value was `T = 0.3622`.

The calibrated real/fake cutoff was `0.5394`. The calibrated uncertain band was empty, so the demo uses a cautious band:

- `< 0.5` → **real**
- `0.5 – 0.8` → **NOT SURE**
- `>= 0.8` → **fake**

This is a safety policy for live microphone audio, not a new scientific result. You can disable it with `DEMO_BAND=0`.

### Shortcut audit

| Test set | Noise-floor AUC | 6-feature classifier AUC | v3 AUC |
|---|---:|---:|---:|
| MLAAD unseen | 0.932 | 0.952 | 0.998 |
| MLAAD matched | 0.501 | 0.631 | 0.953 |

- On the **unseen** set, even a simple recording-style classifier gets 0.952 AUC. So v3’s 2.60% EER is partly helped by a shortcut.
- On the **matched** set, the noise-floor shortcut is removed (0.501). The simple classifier drops to 0.631, while v3 still reaches 0.953. That is evidence v3 learned something beyond noise floor, but the matched set is small and shares 37 fake systems with training.

---

## 9. Limits and honest interpretation

- **One seed.** The v3 numbers are from one training run.
- **Subsets.** These are subsets of ASVspoof and MLAAD, not the full official benchmarks.
- **No external dataset yet.** No ASVspoof 5, no other-language test, no phone-call test.
- **Shortcut confound.** The unseen MLAAD test still has a strong recording-style shortcut.
- **Matched set is not clean.** It removes the noise-floor shortcut but shares 37 fake systems with training.
- **Live microphone audio is out-of-domain.** A real voice can be flagged fake. The demo now reports borderline cases as `NOT SURE`.
- **Silence is not valid input.** The demo rejects near-silent clips.
- **Not forensic.** Do not use this to accuse a person or as legal evidence.
- **Licensing.** MLAAD-tiny is CC BY-NC 4.0 (non-commercial).

### What the model is good at

- Older ASVspoof-style spoofing.
- Many modern TTS systems in MLAAD.
- Ranking fakes above reals on the internal test subsets.

### What it is not good at

- Arbitrary live microphone recordings.
- Quiet, noisy, compressed, or reverberant audio.
- Some attacks, especially `griffin_lim` (only 52.9% caught at the calibrated threshold).
- Proving that a specific real person’s voice is AI-generated.

---

## 10. What each file and folder is for

### Main demo files

| File | What it does |
|---|---|
| `app.py` | The FastAPI server. Loads the model, cleans audio, splits it into windows, skips silence, and returns real/fake/NOT SURE. |
| `index.html` | The browser page. Upload or record audio, convert it to WAV, and show the result. |
| `checkpoints/xlsr_v3.pt` | The v3 model weights (about 1.2 GB). Required for the demo. |
| `eval/calibration_xlsr_v3.json` | The v3 temperature and thresholds. Required for the demo. |

### v3 training and evaluation

| File | What it does |
|---|---|
| `model_v3.py` | Defines the v3 model: XLS-R + all-layer fusion + AASIST-style back-end. |
| `train_v3.py` | Trains v3. Builds loaders, applies augmentation, runs epochs, saves checkpoints. |
| `evaluate_v3.py` | Evaluates a checkpoint. Computes EER/AUC/accuracy, calibration, per-attack results, and shortcut control. |
| `data_v3.py` | Builds the v3 CSV splits and the PyTorch dataset. |
| `augment_v3.py` | Training augmentations: loudness, noise floor, noise, reverb, codec, RawBoost, crops. |
| `losses_v3.py` | Focal loss and hybrid loss. |
| `calibration_v3.py` | Temperature scaling and threshold fitting. |
| `selftest_v3.py` | Offline tests for the data/augmentation/model/calibration pipeline. |
| `pack_v3.py` | Creates a small code zip for Colab. |
| `colab_v3.ipynb` | The Colab notebook used to train and evaluate v3. |
| `floor_eq.py` | Noise-floor features used by the shortcut audit; also used in the v2 experiment. |

### v1 and older experiment files

| File | What it does |
|---|---|
| `xlsr_aasist.py` | The v1 XLS-R + AASIST model and dataset. `app.py` uses it for the v1 checkpoint. |
| `dataset.py` | The CNN baseline dataset (log-mel features). |
| `model.py` | The CNN baseline model. |
| `train.py` | CNN baseline training. |
| `evaluate.py` | CNN baseline evaluation. |
| `metrics.py` | EER/AUC/metric helpers. |
| `RawBoost.py` | Official RawBoost augmentation code. |
| `rawboost_aug.py` | Wrapper around RawBoost, used by the training pipelines. |
| `prepare_data.py` | Builds ASVspoof CSVs from raw audio. |
| `prepare_mlaad.py` | Builds MLAAD CSVs from raw audio. |
| `make_matched_mlaad.py` | Builds the matched MLAAD test set. |
| `analyze.py` | Audio and score analysis for v1. |
| `diagnose_domain.py` | Domain/shortcut diagnostics. |
| `make_split_v2.py` | v2 system-disjoint split experiment. Historical. |
| `pack_for_colab.py` | Builds `colab_pack.zip` from data + audio. |
| `pack_v2.py` | Builds `colab_pack_v2.zip` for the new MLAAD split. |
| `xlsr_aasist_colab.ipynb` | v1 Colab training notebook. |
| `check_setup.py` | Small environment check. |

### Data and results folders

| Path | What it contains |
|---|---|
| `data_v3/v3_train.csv` | Training list. |
| `data_v3/v3_dev*.csv` | Dev lists for epoch selection and calibration. |
| `data_v3/v3_test_asv.csv` | ASVspoof test list. |
| `data_v3/v3_test_mlaad_all.csv` | Full MLAAD test list. |
| `data_v3/v3_test_mlaad_matched.csv` | Matched MLAAD test list. |
| `data_v3/v3_test_mlaad_unseen.csv` | Unseen-system MLAAD test list. |
| `eval/eval_xlsr_v3.csv` | The final v3 results table. |
| `eval/scores_xlsr_v3_*.csv` | Per-file scores and labels for each test set. |
| `checkpoints/` | Saved model weights. `xlsr_v3.pt` is the main one. |
| `colab_pack/` | Packed ASVspoof + MLAAD audio and CSVs for Colab/evaluation. |
| `colab_pack_v2/` | Packed MLAAD train2 + unseen-system audio and CSVs. |
| `MLAAD-tiny/` | Raw MLAAD-tiny dataset. |
| `data/` | Older v1/v2 CSV splits. |
| `results/` | Older v1 results and diagnostics. |
| `README.md` | The main project overview and quickstart. |
| `MODEL_CARD.md` | The formal model card: metrics, risks, intended use. |
| `requirements.txt` | Python packages needed. |
| `eer_chart.png` | EER chart shown in the README. |
| `PROJECT_EXPLAINED.md` | This guide. |

---

## 11. How to run it

### Run the demo

```bash
cd voice-deepfake-detector
# download xlsr_v3.pt from the latest release into checkpoints/
CKPT=checkpoints/xlsr_v3.pt \
CALIB=eval/calibration_xlsr_v3.json \
DEVICE=mps \
python3 -m uvicorn app:app --port 8000
```

Then open `http://localhost:8000`.

Use a quiet room, record at least 4 seconds, and start speaking immediately. Silent clips are rejected.

### Re-run the full evaluation

```bash
python3 evaluate_v3.py --ckpt checkpoints/xlsr_v3.pt \
  --calibrate-csv data_v3/v3_dev.csv \
  --csv asv_eval=data_v3/v3_test_asv.csv \
  --csv mlaad_unseen=data_v3/v3_test_mlaad_unseen.csv \
  --csv mlaad_all=data_v3/v3_test_mlaad_all.csv \
  --csv mlaad_matched=data_v3/v3_test_mlaad_matched.csv \
  --batch 16 --workers 2 --out eval
```

### Rebuild the v3 data splits

```bash
python3 data_v3.py --root colab_pack --root colab_pack_v2
```

### Train again

Open `colab_v3.ipynb` on Google Colab with a T4 GPU. The notebook expects the data packs on Google Drive.

---

## 12. Glossary

| Term | Simple meaning |
|---|---|
| **Real** | A genuine human recording. |
| **Fake / spoof** | Audio generated or modified by AI. |
| **TTS** | Text-to-speech: AI that turns text into a voice. |
| **Waveform** | The raw audio signal. |
| **16 kHz mono** | 16,000 samples per second, one channel. A standard speech format. |
| **Checkpoint** | A saved copy of the model’s learned weights. |
| **Training** | Showing the model many labelled examples so it learns. |
| **Dev set** | Practice test used to choose the best epoch and thresholds. |
| **Test set** | Final exam the model never trains on. |
| **Epoch** | One full pass through the training data. |
| **Batch** | A small group of clips processed together. |
| **Threshold** | The score cutoff for calling something fake. |
| **EER** | Equal Error Rate. Lower is better; 0% is perfect, 50% is guessing. |
| **AUC** | Ranking quality. 0.5 is guessing, 1.0 is perfect. |
| **Balanced accuracy** | Average of real accuracy and fake accuracy. |
| **F1** | A combined score for precision and recall. |
| **Calibration** | Adjusting scores/thresholds so decisions are more sensible. |
| **Temperature** | A calibration number; smaller values make scores more extreme. |
| **Domain shift** | Training audio and real-world audio have different recording conditions. |
| **Shortcut** | The model uses an irrelevant clue (like background noise) instead of the voice itself. |
| **Unseen system** | A TTS generator that was not used during training. |
| **Matched set** | Real/fake pairs chosen to have similar background noise, removing that shortcut. |
| **Seed** | A number that makes a random training run reproducible. One seed = one run. |
| **Inference** | Using a trained model to make a prediction. |
| **NOT SURE** | A demo decision for scores near the boundary; it means “insufficient evidence”, not “real” or “fake”. |

---

## 13. What to tell someone who asks “does it work?”

A good answer:

> “It works well on the benchmark test sets I used: about 0.25% EER on ASVspoof and 2.6% EER on unseen TTS systems from MLAAD. But those tests still have dataset shortcuts, it is one seed, and live microphone audio is out-of-domain. The demo reports borderline cases as NOT SURE instead of pretending to be certain. It is a strong research demo, not a forensic tool.”

A bad answer:

> “It can tell if any voice is AI-generated.”

That would be false.

---

## 14. If I continue the project

The most valuable next steps, in order:

1. Test on an external dataset (for example ASVspoof 5) and on phone/compressed audio.
2. Train with a second seed and report mean/variance.
3. Build a clean **matched and unseen-system** test set.
4. Add personal calibration: fit a threshold on 20–30 real recordings from the user’s own microphone.
5. Add Docker and a simple deployment.
6. Improve the demo UI: show the score history, allow downloading the exact WAV, and add a device selector.

---

## 15. The shortest possible version

- **Goal:** detect AI-generated voice clips.
- **Final model:** v3, XLS-R all-layer fusion + AASIST-style classifier.
- **Data:** ASVspoof 2019 LA + MLAAD-tiny, split by TTS system.
- **Good results:** 0.25% EER on ASVspoof, 2.60% on unseen TTS systems.
- **Biggest caveat:** microphone audio is out-of-domain and dataset shortcuts are real.
- **Demo behavior:** real below 0.5, fake above 0.8, NOT SURE in between; silence rejected.
- **Status:** complete as a research/demo project, not production-ready.
