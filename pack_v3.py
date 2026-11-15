"""Zip ONLY the v3 code (a few KB) for Colab. The audio stays in the existing
colab_pack.zip + colab_pack_v2.zip, so you do not need to re-upload 1.4 GB.

  python3 pack_v3.py              -> colab_v3_code.zip

On Colab:
  !unzip -q colab_pack.zip    -d /content/v3/colab_pack
  !unzip -q colab_pack_v2.zip -d /content/v3/colab_pack_v2
  !unzip -q colab_v3_code.zip -d /content/v3
  %cd /content/v3
  !python data_v3.py --root colab_pack --root colab_pack_v2
"""
import argparse
import os
import zipfile

FILES = [
    "model_v3.py", "augment_v3.py", "data_v3.py", "losses_v3.py",
    "calibration_v3.py", "train_v3.py", "evaluate_v3.py",
    "selftest_v3.py", "metrics.py", "RawBoost.py", "rawboost_aug.py",
    "floor_eq.py", "xlsr_aasist.py", "app.py", "index.html",
    "requirements.txt",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="colab_v3_code.zip")
    a = ap.parse_args()
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in FILES:
            if os.path.exists(f):
                z.write(f, arcname=f)
                print("added", f)
            else:
                print("missing", f)
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e3:.1f} KB)")


if __name__ == "__main__":
    main()
