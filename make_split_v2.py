"""
Step 1+2: split MLAAD by TTS system (unseen systems for testing) and check the noise shortcut.
  python3 make_split_v2.py --root MLAAD-tiny
Writes data/mlaad_train2.csv and data/mlaad_test_unseen.csv  (test = TTS systems NOT in train).
Then trains a tiny classifier on simple audio numbers only (floor, loudness, ...). If that tiny
classifier can already separate real/fake on the unseen test set, a big model can cheat the same way.
"""
import argparse, os, zlib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from prepare_mlaad import build
from floor_eq import load4, feats, randomize_style

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="MLAAD-tiny")
ap.add_argument("--out", default="data")
ap.add_argument("--test_share", type=float, default=0.3, help="share of fake clips whose TTS systems go to test")
ap.add_argument("--per_sys_train", type=int, default=150, help="max train clips per TTS system")
ap.add_argument("--max_test", type=int, default=1000, help="max fake clips in test")
ap.add_argument("--floor_lo", type=float, default=-50.0, help="random noise floor range, low end (dB)")
ap.add_argument("--floor_hi", type=float, default=-40.0, help="random noise floor range, high end (dB)")
ap.add_argument("--gain_db", type=float, default=10.0, help="random loudness change +-dB")
ap.add_argument("--skip_raw", action="store_true", help="skip the RAW check (already seen)")
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

df = build(a.root)
if df.empty:
    raise SystemExit("No wav files found. Check --root.")
real, fake = df[df.label == 0], df[df.label == 1]
print(f"Found in total: real={len(real)} fake={len(fake)} | fake TTS systems={fake.attack.nunique()}")

# ---- split TTS systems into train systems / unseen test systems ----
rng = np.random.RandomState(a.seed)
counts = fake.attack.value_counts()
test_sys, acc = [], 0
for s in rng.permutation(counts.index.tolist()):
    if acc >= a.test_share * len(fake):
        break
    test_sys.append(s); acc += counts[s]
fake_te = fake[fake.attack.isin(test_sys)]
fake_te = fake_te.sample(min(a.max_test, len(fake_te)), random_state=a.seed)
fake_tr = fake[~fake.attack.isin(test_sys)]
fake_tr = pd.concat([g.sample(min(a.per_sys_train, len(g)), random_state=a.seed) for _, g in fake_tr.groupby("attack")])

# ---- real clips: disjoint clips for test and train ----
n_te = min(len(fake_te), len(real) // 2)
real_te = real.sample(n_te, random_state=a.seed)
fake_te = fake_te.sample(n_te, random_state=a.seed)
real_rest = real.drop(real_te.index)
n_tr = min(len(fake_tr), len(real_rest))
real_tr = real_rest.sample(n_tr, random_state=a.seed)
fake_tr = fake_tr.sample(n_tr, random_state=a.seed)

tr = pd.concat([real_tr, fake_tr]).sample(frac=1, random_state=a.seed).reset_index(drop=True)
te = pd.concat([real_te, fake_te]).sample(frac=1, random_state=a.seed).reset_index(drop=True)
os.makedirs(a.out, exist_ok=True)
tr.to_csv(f"{a.out}/mlaad_train2.csv", index=False)
te.to_csv(f"{a.out}/mlaad_test_unseen.csv", index=False)
print(f"\ntrain: {len(tr)} clips ({n_tr} real + {n_tr} fake) from {fake_tr.attack.nunique()} TTS systems")
print(f"test : {len(te)} clips ({n_te} real + {n_te} fake) from {fake_te.attack.nunique()} UNSEEN TTS systems")
print("Unseen test systems:", ", ".join(sorted(fake_te.attack.unique())))
print("NOTE: real clips of train and test may share speakers/books (not known). Keep this in the README.")

# ---- shortcut check: tiny classifier on simple numbers ----
def table(d, eq):
    rows = []
    for p in tqdm(d.path, desc=("equalised" if eq else "raw") + " features", leave=False):
        x = load4(p)
        if eq:
            x = randomize_style(x, np.random.RandomState(zlib.crc32(p.encode()) % (2 ** 32)), a.floor_lo, a.floor_hi, a.gain_db)
        rows.append(feats(x))
    return pd.DataFrame(rows)

print("\n=== SHORTCUT CHECK (tiny classifier, simple audio numbers only) ===")
print("AUC near 0.5 = good (no shortcut). AUC far from 0.5 = the data has a shortcut.\n")
for eq in ((True,) if a.skip_raw else (False, True)):
    Xtr, Xte = table(tr, eq), table(te, eq)
    single = {c: roc_auc_score(te.label, Xte[c]) for c in Xte.columns}
    single = {c: max(v, 1 - v) for c, v in single.items()}
    clf = GradientBoostingClassifier(random_state=0).fit(Xtr, tr.label)
    auc = roc_auc_score(te.label, clf.predict_proba(Xte)[:, 1])
    print(("AFTER random loudness + random noise floor (%.0f..%.0f dB, gain +-%.0f dB)" % (a.floor_lo, a.floor_hi, a.gain_db)) if eq else "RAW audio")
    print("  single-feature separability (0.5 = none):", {c: round(v, 3) for c, v in single.items()})
    print(f"  tiny classifier, trained on train systems, tested on unseen systems: AUC = {auc:.3f}\n")
