"""Offline smoke test for the whole v3 pipeline.

No XLS-R download, no GPU: a tiny randomly-initialised Wav2Vec2 stands in for
XLS-R, and a handful of already-packed clips are used as data.

  python3 selftest_v3.py
"""
import os
import tempfile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from augment_v3 import DEFAULT_TRAIN_AUG, augment_train, equalize_floor, noise_floor_db
from calibration_v3 import fit_calibration, summarize
from data_v3 import AudioDatasetV3, build_v3_splits, make_weighted_sampler
from losses_v3 import build_loss
from model_v3 import XLSR_AASIST_V3, build_tiny_ssl, predict_logits


def small_csv(path):
    df = pd.read_csv("colab_pack/data/mlaad_all.csv")
    real = df[df.label == 0].head(4)
    fake = df[df.label == 1].head(4)
    df = pd.concat([real, fake])
    df["path"] = ["colab_pack/" + p for p in df["path"]]
    df["source"] = "mlaad"
    df.to_csv(path, index=False)
    return path


def test_augment():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(64000) * 1e-4).astype(np.float32)  # ~-80 dB floor
    floor0 = noise_floor_db(x)
    y = equalize_floor(x, rng, lo=-40, hi=-40, prob=1.0)
    assert noise_floor_db(y) > floor0, "equalize_floor did not raise the floor"
    x = (rng.standard_normal(64000) * 0.05).astype(np.float32)
    z = augment_train(x, rng, {**DEFAULT_TRAIN_AUG, "rawboost_prob": 1.0,
                               "rawboost_algos": (4,), "noise_prob": 1.0})
    assert z.shape == x.shape and np.isfinite(z).all() and np.abs(z).max() <= 1.01
    print("  augment: ok")


def test_dataset(csv):
    aug = dict(rawboost_prob=0.5, rawboost_algos=(4, 8), noise_prob=0.5,
               reverb_prob=0.5, codec_prob=0.5, floor_prob=0.5)
    ds = AudioDatasetV3(csv, train=True, aug=aug, seed=1)
    for i in range(min(6, len(ds))):
        wav, y, src = ds[i]
        assert wav.shape == (64000,) and torch.isfinite(wav).all()
        assert float(wav.abs().max()) <= 1.01
    ev = AudioDatasetV3(csv, train=False, seed=1)
    wav, y, src = ev[0]
    assert wav.shape == (64000,)
    sampler = make_weighted_sampler(pd.read_csv(csv), source_weights={"mlaad": 1.0})
    assert sampler.num_samples == len(ds)
    print("  dataset + sampler: ok")


def test_model(csv):
    model = XLSR_AASIST_V3(ssl=build_tiny_ssl(), d=8, layers="all", tiny=True)
    x = torch.randn(2, 64000)
    y = torch.tensor([0.0, 1.0])
    logits = model(x)
    assert logits.shape == (2,) and torch.isfinite(logits).all()

    before = model.proj.weight.detach().clone()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = build_loss("focal")
    for _ in range(2):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    assert torch.isfinite(loss) and not torch.equal(before, model.proj.weight)
    assert torch.allclose(model.layer_weights().sum(), torch.tensor(1.0), atol=1e-5)

    ds = AudioDatasetV3(csv, train=False, seed=1)
    loader = DataLoader(Subset(ds, list(range(4))), batch_size=2)
    labels, lg = predict_logits(model, loader, torch.device("cpu"), amp=False)
    assert len(labels) == len(lg) == 4
    print("  model forward/backward/predict: ok")


def test_calibration():
    rng = np.random.default_rng(0)
    labels = (rng.random(300) < 0.5).astype(int)
    logits = labels * 2.0 + rng.normal(size=300)
    cal = fit_calibration(logits, labels)
    assert 0.05 <= cal["temperature"] <= 20.0
    m = summarize(labels, logits, temperature=cal["temperature"],
                  threshold=cal["fake_thr"], real_thr=cal["real_thr"])
    for k in ("EER", "AUC", "balanced_acc", "real_thr", "fake_thr", "uncertain_rate"):
        assert k in m
    assert 0.0 <= m["AUC"] <= 1.0
    print("  calibration + metrics: ok")


def test_splits():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "data_v3")
        paths = build_v3_splits(out_dir=out, roots=(".", "colab_pack", "colab_pack_v2"))
        for k in ("train", "dev", "dev_asv", "dev_mlaad", "test_asv", "test_mlaad_unseen"):
            assert os.path.exists(paths[k]), k
        tr = pd.read_csv(paths["train"])
        te = pd.read_csv(paths["test_mlaad_unseen"])
        tr_sys = set(tr[tr.label == 1].attack.astype(str))
        te_sys = set(te[te.label == 1].attack.astype(str))
        assert not (tr_sys & te_sys), f"TTS system leakage: {tr_sys & te_sys}"
        assert {"asv", "mlaad"} <= set(tr.source.unique())
        print(f"  splits: train={len(tr)} dev={len(pd.read_csv(paths['dev']))} "
              f"unseen_test={len(te)} (no system leakage): ok")


def main():
    assert os.path.exists("colab_pack/data/mlaad_all.csv"), "run from the project root"
    print("v3 self-test")
    with tempfile.TemporaryDirectory() as tmp:
        csv = small_csv(os.path.join(tmp, "small.csv"))
        test_augment()
        test_dataset(csv)
        test_model(csv)
        test_calibration()
    test_splits()
    print("ALL V3 SELFTESTS PASSED")


if __name__ == "__main__":
    main()
