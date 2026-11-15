"""
Copy ONLY the files listed in your CSVs into one folder (first 4 sec, 16 kHz mono, FLAC)
and zip it for Google Drive / Colab.
  python3 pack_v2.py
Output: colab_pack_v2.zip  (contains audio/ and data/*.csv with RELATIVE paths)
On Colab: unzip, then cd into the unzipped folder before running anything.
"""
import os, shutil, argparse
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

SR = 16000
MAX_LEN = 4 * SR

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="colab_pack_v2")
ap.add_argument("--full_length", action="store_true", help="keep full audio instead of first 4 sec")
a = ap.parse_args()

# csv name -> folder inside audio/   (only the NEW MLAAD split; train/dev/eval are already in colab_pack.zip)
CSVS = {"mlaad_train2.csv": "mlaad2", "mlaad_test_unseen.csv": "mlaad2"}

if os.path.exists(a.out):
    shutil.rmtree(a.out)
os.makedirs(f"{a.out}/data")

mapping = {}      # old absolute path -> new relative path (so files shared between CSVs are written once)
for csv, folder in CSVS.items():
    p = os.path.join(a.data, csv)
    if not os.path.exists(p):
        print("skip (not found):", p)
        continue
    df = pd.read_csv(p)
    os.makedirs(f"{a.out}/audio/{folder}", exist_ok=True)
    new_paths = []
    for i, old in enumerate(tqdm(df.path, desc=csv)):
        if old not in mapping:
            x, sr = sf.read(old, dtype="float32")
            if x.ndim > 1:
                x = x.mean(axis=1)
            assert sr == SR, f"{old}: sample rate {sr}, expected {SR}"
            if not a.full_length:
                x = x[:MAX_LEN]
            name = f"{folder}_{len(mapping):05d}.flac"
            rel = f"audio/{folder}/{name}"
            sf.write(f"{a.out}/{rel}", x, SR, format="FLAC")
            mapping[old] = rel
        new_paths.append(mapping[old])
    df["path"] = new_paths
    df.to_csv(f"{a.out}/data/{csv}", index=False)
    print(f"{csv}: {len(df)} rows")

print("Zipping...")
shutil.make_archive(a.out, "zip", a.out)
mb = os.path.getsize(f"{a.out}.zip") / 1e6
print(f"Done: {a.out}.zip ({mb:.0f} MB). Files written: {len(mapping)}")
