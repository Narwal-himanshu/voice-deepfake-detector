"""
Reads ASVspoof 2019 LA protocol files and makes small CSVs (path,label).
label: 0 = bonafide (real), 1 = spoof (fake)

Usage:
  python prepare_data.py --root /path/to/LA --out data --n_train 4000 --n_dev 1000 --n_eval 2000
"""
import argparse, os
import pandas as pd

SPLITS = {
    "train": ("ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt"),
    "dev":   ("ASVspoof2019_LA_dev",   "ASVspoof2019.LA.cm.dev.trl.txt"),
    "eval":  ("ASVspoof2019_LA_eval",  "ASVspoof2019.LA.cm.eval.trl.txt"),
}

def load_split(root, split):
    folder, proto = SPLITS[split]
    proto_path = os.path.join(root, "ASVspoof2019_LA_cm_protocols", proto)
    # line: SPEAKER UTT_ID - ATTACK_ID LABEL
    df = pd.read_csv(proto_path, sep=" ", header=None,
                     names=["speaker", "utt", "env", "attack", "label"])
    df["path"] = df["utt"].apply(lambda u: os.path.join(root, folder, "flac", u + ".flac"))
    df["label"] = (df["label"] == "spoof").astype(int)
    return df[["path", "utt", "attack", "label"]]

def balanced_subset(df, n, seed=42):
    """Take n files with ~50% real / 50% fake (stratified), shuffled."""
    if n is None or n >= len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    half = n // 2
    real = df[df.label == 0]
    fake = df[df.label == 1]
    out = pd.concat([real.sample(min(half, len(real)), random_state=seed),
                     fake.sample(min(n - half, len(fake)), random_state=seed)])
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path to the LA folder")
    ap.add_argument("--out", default="data")
    ap.add_argument("--n_train", type=int, default=4000)
    ap.add_argument("--n_dev", type=int, default=1000)
    ap.add_argument("--n_eval", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for split, n in [("train", a.n_train), ("dev", a.n_dev), ("eval", a.n_eval)]:
        df = balanced_subset(load_split(a.root, split), n)
        df.to_csv(os.path.join(a.out, f"{split}.csv"), index=False)
        print(f"{split}: {len(df)} files | real={int((df.label==0).sum())} fake={int((df.label==1).sum())}")
