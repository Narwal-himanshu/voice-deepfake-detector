import argparse, glob, os
import pandas as pd

def find_wavs(folder):
    return sorted(glob.glob(os.path.join(folder, "**", "*.wav"), recursive=True))

def build(root):
    rows = []
    for lang_dir in sorted(glob.glob(os.path.join(root, "original", "*"))):
        if not os.path.isdir(lang_dir):
            continue
        lang = os.path.basename(lang_dir)
        for p in find_wavs(lang_dir):
            rows.append({"path": p, "utt": os.path.basename(p), "attack": "real", "label": 0, "lang": lang})
    for lang_dir in sorted(glob.glob(os.path.join(root, "fake", "*"))):
        if not os.path.isdir(lang_dir):
            continue
        lang = os.path.basename(lang_dir)
        for sys_dir in sorted(glob.glob(os.path.join(lang_dir, "*"))):
            if not os.path.isdir(sys_dir):
                continue
            system = os.path.basename(sys_dir)
            for p in find_wavs(sys_dir):
                rows.append({"path": p, "utt": os.path.basename(p), "attack": system, "label": 1, "lang": lang})
    return pd.DataFrame(rows)

def balanced(df, n, seed=42):
    real = df[df.label == 0]
    fake = df[df.label == 1]
    nr = min(n, len(real), len(fake))
    real = real.sample(nr, random_state=seed)
    systems = fake["attack"].nunique()
    per = max(1, -(-nr // systems))
    fake = fake.groupby("attack", group_keys=False).apply(
        lambda g: g.sample(min(per, len(g)), random_state=seed))
    fake = fake.sample(min(nr, len(fake)), random_state=seed)
    return pd.concat([real, fake]).sample(frac=1, random_state=seed).reset_index(drop=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="MLAAD-tiny")
    ap.add_argument("--out", default="data")
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = build(a.root)
    if df.empty:
        raise SystemExit("No wav files found. Check --root path.")
    print("Found:", df.groupby(["lang", "label"]).size().to_string())

    parts = []
    for lang, g in df.groupby("lang"):
        if g.label.nunique() < 2:
            print(f"skip {lang}: needs both real and fake files")
            continue
        sub = balanced(g, a.n)
        sub.to_csv(os.path.join(a.out, f"mlaad_{lang}.csv"), index=False)
        print(f"mlaad_{lang}.csv: {len(sub)} files | real={int((sub.label==0).sum())} fake={int((sub.label==1).sum())}")
        parts.append(sub)
    if parts:
        allp = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
        allp.to_csv(os.path.join(a.out, "mlaad_all.csv"), index=False)
        print(f"mlaad_all.csv: {len(allp)} files")