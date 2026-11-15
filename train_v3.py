"""Train the v3 cross-domain anti-spoofing model.

Example (Colab / GPU):
  python train_v3.py --train-csv data_v3/v3_train.csv --dev-csv data_v3/v3_dev.csv \
      --name xlsr_v3 --epochs 20 --batch 16 --loss focal

What is different from train.py / the v1 notebook:
  * XLS-R layer fusion (all 24 layers) instead of the last layer only
  * ASVspoof + MLAAD multi-dataset training, balanced per (source, label)
  * random crops, RawBoost with random algos, SNR noise, reverb, codec and
    noise-floor equalisation (breaks the MLAAD recording shortcut)
  * focal / hybrid loss instead of plain BCE
  * checkpoint selected by mean AUC over ASV dev AND unseen-TTS MLAAD dev
"""
import argparse
import json
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from calibration_v3 import summarize
from data_v3 import AudioDatasetV3, make_weighted_sampler, worker_init_fn
from losses_v3 import build_loss
from model_v3 import XLSR_AASIST_V3, load_checkpoint, predict_logits, save_checkpoint


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if os.environ.get("FORCE_CPU") == "1":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_kv(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = float(v)
    return out


def make_aug(args):
    if args.no_aug:
        return dict(normalize_rms=True, floor_prob=0.0, rawboost_prob=0.0,
                    noise_prob=0.0, reverb_prob=0.0, codec_prob=0.0)
    return dict(
        target_rms_db=args.target_rms_db,
        floor_prob=args.floor_prob,
        floor_lo=args.floor_lo,
        floor_hi=args.floor_hi,
        rawboost_prob=args.rawboost_prob,
        rawboost_algos=tuple(int(v) for v in args.rawboost_algos.split(",") if v != ""),
        noise_prob=args.noise_prob,
        snr_lo=args.snr_lo,
        snr_hi=args.snr_hi,
        reverb_prob=args.reverb_prob,
        codec_prob=args.codec_prob,
    )


def build_loader(csv, train, args, aug, epoch=0, sampler=None):
    ds = AudioDatasetV3(csv, train=train, aug=aug, seed=args.seed, epoch=epoch)
    if sampler is None and train:
        sampler = make_weighted_sampler(
            pd.read_csv(csv), class_balance=True,
            source_weights=parse_kv(args.source_weights))
    return DataLoader(
        ds, batch_size=args.batch, sampler=sampler,
        shuffle=False, drop_last=train, num_workers=args.workers,
        worker_init_fn=worker_init_fn, pin_memory=(args.device == "cuda"),
    ), ds


def evaluate(model, loaders, device, amp):
    out = {}
    for name, (loader, _) in loaders.items():
        labels, logits = predict_logits(model, loader, device, amp)
        out[name] = summarize(labels, logits)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", default="data_v3/v3_train.csv")
    ap.add_argument("--dev-csv", action="append", default=None,
                    help="name=csv, repeatable (default: asv_dev=data_v3/v3_dev.csv)")
    ap.add_argument("--name", default="xlsr_v3")
    ap.add_argument("--out", default="runs_v3")
    ap.add_argument("--resume", action="store_true", help="continue from <out>/<name>.pt")
    ap.add_argument("--save-optimizer", action="store_true",
                    help="also store optimizer/scheduler in the checkpoint (bigger, enables exact resume)")
    ap.add_argument("--save-last", action="store_true",
                    help="write <name>_last.pt every epoch so --resume continues from the last epoch")
    ap.add_argument("--ssl-name", default="facebook/wav2vec2-xls-r-300m")
    ap.add_argument("--debug-tiny", action="store_true",
                    help="random tiny SSL for a fast CPU smoke test (no download)")
    ap.add_argument("--layers", default="all", help="'all' or e.g. 0,4,8,12,16,20,24")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr-ssl", type=float, default=1e-5)
    ap.add_argument("--lr-back", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--amp", dest="amp", action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--freeze-ssl-epochs", type=int, default=0)
    ap.add_argument("--loss", default="focal", choices=["bce", "focal", "hybrid"])
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    ap.add_argument("--focal-alpha", type=float, default=0.25)
    ap.add_argument("--hybrid-weight", type=float, default=0.5)
    ap.add_argument("--source-weights", action="append", default=[],
                    help="e.g. asv=1.0 mlaad=1.0")
    ap.add_argument("--select", default="mean_auc", choices=["mean_auc", "mean_eer"])
    # augmentation knobs
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--target-rms-db", type=float, default=-23.0)
    ap.add_argument("--floor-prob", type=float, default=0.5)
    ap.add_argument("--floor-lo", type=float, default=-62.0)
    ap.add_argument("--floor-hi", type=float, default=-45.0)
    ap.add_argument("--rawboost-prob", type=float, default=0.7)
    ap.add_argument("--rawboost-algos", default="4,5,8")
    ap.add_argument("--noise-prob", type=float, default=0.25)
    ap.add_argument("--snr-lo", type=float, default=10.0)
    ap.add_argument("--snr-hi", type=float, default=35.0)
    ap.add_argument("--reverb-prob", type=float, default=0.15)
    ap.add_argument("--codec-prob", type=float, default=0.20)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    args.device = device.type
    amp = bool(args.amp and device.type == "cuda")
    os.makedirs(args.out, exist_ok=True)

    if not os.path.exists(args.train_csv):
        raise SystemExit(f"missing {args.train_csv}; run: python data_v3.py")

    dev_specs = args.dev_csv or [f"asv_dev={args.train_csv.replace('train', 'dev')}"]
    dev_specs = [s if "=" in s else f"set{i}={s}" for i, s in enumerate(dev_specs)]
    dev_specs = [s.split("=", 1) for s in dev_specs]
    dev_specs = [(n, p) for n, p in dev_specs if os.path.exists(p)]

    aug = make_aug(args)
    print("device:", device, "| amp:", amp, "| layers:", args.layers, "| loss:", args.loss)
    print("train:", args.train_csv)
    print("dev:", dev_specs)
    print("aug:", aug)

    train_df = pd.read_csv(args.train_csv)
    sampler = make_weighted_sampler(train_df, class_balance=True,
                                    source_weights=parse_kv(args.source_weights))
    train_loader, train_ds = build_loader(args.train_csv, True, args, aug, sampler=sampler)
    dev_loaders = {n: build_loader(p, False, args, aug) for n, p in dev_specs}

    model = XLSR_AASIST_V3(ssl_name=args.ssl_name, d=args.d, layers=args.layers,
                           dropout=args.dropout, tiny=args.debug_tiny).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"trainable params: {n_train:.1f} M")

    if args.freeze_ssl_epochs > 0:
        for n, p in model.ssl.named_parameters():
            if not n.startswith("feature_extractor"):
                p.requires_grad = False

    ssl_p = [p for p in model.ssl.parameters() if p.requires_grad]
    ssl_ids = {id(p) for p in ssl_p}
    back_p = [p for p in model.parameters() if p.requires_grad and id(p) not in ssl_ids]
    opt = torch.optim.AdamW([
        {"params": ssl_p, "lr": args.lr_ssl},
        {"params": back_p, "lr": args.lr_back},
    ], weight_decay=args.weight_decay)

    steps_per_epoch = max(1, len(train_loader))
    total = args.epochs * steps_per_epoch
    warm = max(1, int(args.warmup * total))

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = build_loss(args.loss, gamma=args.focal_gamma, alpha=args.focal_alpha,
                         focal_weight=args.hybrid_weight)

    ckpt_path = os.path.join(args.out, f"{args.name}.pt")
    last_path = os.path.join(args.out, f"{args.name}_last.pt")
    hist_path = os.path.join(args.out, f"history_{args.name}.csv")
    best = -1e9 if args.select == "mean_auc" else 1e9
    bad, history, start_epoch = 0, [], 1

    resume_path = last_path if (args.resume and os.path.exists(last_path)) else ckpt_path
    if args.resume and os.path.exists(resume_path):
        obj = load_checkpoint(resume_path)
        if isinstance(obj, dict) and "state_dict" in obj:
            model.load_state_dict(obj["state_dict"])
            start_epoch = int(obj.get("epoch", 0)) + 1
            best = float(obj.get("best_score", best))
            history = list(obj.get("history", []))
            bad = int(obj.get("bad", 0))
            sched_loaded = False
            for key, target in (("optimizer", opt), ("scheduler", sched)):
                if obj.get(key) is not None:
                    try:
                        target.load_state_dict(obj[key])
                        sched_loaded = sched_loaded or key == "scheduler"
                    except Exception as e:
                        print(f"resume: could not load {key}: {e}")
            if not sched_loaded:
                for _ in range(max(0, (start_epoch - 1) * steps_per_epoch)):
                    sched.step()
            print(f"resumed {resume_path}: start epoch {start_epoch}, best={best:.4f}")
        else:
            model.load_state_dict(obj)
            print(f"resume: loaded weights only from {resume_path}")

    if start_epoch > args.epochs:
        raise SystemExit(f"start_epoch {start_epoch} > --epochs {args.epochs}; increase --epochs")

    for ep in range(start_epoch, args.epochs + 1):
        if args.freeze_ssl_epochs and ep == args.freeze_ssl_epochs + 1:
            for n, p in model.ssl.named_parameters():
                if not n.startswith("feature_extractor"):
                    p.requires_grad = True
            ssl_p = [p for p in model.ssl.parameters() if p.requires_grad]
            ssl_ids = {id(p) for p in ssl_p}
            back_p = [p for p in model.parameters() if p.requires_grad and id(p) not in ssl_ids]
            opt = torch.optim.AdamW([
                {"params": ssl_p, "lr": args.lr_ssl},
                {"params": back_p, "lr": args.lr_back},
            ], weight_decay=args.weight_decay)
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda s: lr_lambda(s + (ep - 1) * steps_per_epoch))
            print(f"epoch {ep}: SSL unfrozen")

        # recreate loader so the random crop / augmentation epoch seed changes
        train_loader, train_ds = build_loader(args.train_csv, True, args, aug,
                                              epoch=ep, sampler=sampler)
        model.train()
        tot, n = 0.0, 0
        t0 = time.time()
        for wav, y, _src in train_loader:
            wav, y = wav.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(wav)
            loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            n += 1

        metrics = evaluate(model, dev_loaders, device, amp)
        row = {"epoch": ep, "train_loss": tot / max(n, 1), "lr": sched.get_last_lr()[0],
               "sec": round(time.time() - t0, 1)}
        for name, m in metrics.items():
            row[f"{name}_eer"] = m["EER"]
            row[f"{name}_auc"] = m["AUC"]
            row[f"{name}_bacc"] = m["balanced_acc"]
        mean_auc = float(np.mean([m["AUC"] for m in metrics.values()])) if metrics else 0.0
        mean_eer = float(np.mean([m["EER"] for m in metrics.values()])) if metrics else 100.0
        row["mean_auc"], row["mean_eer"] = mean_auc, mean_eer
        history.append(row)
        pd.DataFrame(history).to_csv(hist_path, index=False)
        pretty = " ".join(f"{k}:EER={v['EER']:.2f}/AUC={v['AUC']:.3f}" for k, v in metrics.items())
        print(f"epoch {ep:2d} loss={row['train_loss']:.4f} | {pretty} | mean_auc={mean_auc:.4f}")

        score = mean_auc if args.select == "mean_auc" else -mean_eer
        improved = score > best + 1e-5
        if improved:
            best, bad = score, 0
            extra = {"epoch": ep, "metrics": metrics, "args": vars(args),
                     "best_score": best, "history": history, "bad": bad}
            if args.save_optimizer:
                extra["optimizer"] = opt.state_dict()
                extra["scheduler"] = sched.state_dict()
            save_checkpoint(ckpt_path, model, extra=extra)
            print(f"  saved best -> {ckpt_path}")
        else:
            bad += 1
        if args.save_last:
            save_checkpoint(last_path, model, extra={
                "epoch": ep, "metrics": metrics, "args": vars(args),
                "best_score": best, "history": history, "bad": bad})
        if not improved and bad >= args.patience:
            print(f"early stop after {ep} epochs (no improvement for {bad})")
            break

    print(f"done. best {args.select}={best:.4f} | checkpoint: {ckpt_path}")
    with open(os.path.join(args.out, f"args_{args.name}.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
