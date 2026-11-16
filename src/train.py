"""Shared training script for ALL models.

    python src\\train.py --config configs\\mlp.yaml
    python src\\train.py --config configs\\resnet1d.yaml --seeds 0 1 2
    python src\\train.py --config configs\\mlp.yaml --epochs 5 --seeds 0   (quick test)

For every seed it trains on the TRAIN split, picks the best epoch by
VALIDATION macro-F1 (with early stopping), and saves:
    checkpoints/<config-name>_seed<k>.pt
    experiments/<config-name>/seed<k>_log.csv       (per-epoch metrics)
    experiments/<config-name>/seed<k>_summary.json  (best epoch, time, params, size)
    experiments/<config-name>/val_summary.json      (mean ± SD over seeds)

The TEST split is never touched here -- it is only used by src\\evaluate.py.

Model contract: every model class accepts `num_classes`, takes x of shape
[B, 200, 12] and returns logits [B, num_classes].
"""
import argparse
import csv
import importlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from datasets.wisdm import DEFAULT_NPZ, class_weights, get_dataloaders  # noqa: E402

NUM_CLASSES = 18

# config `model` -> (module, class, config keys passed to the constructor)
MODEL_REGISTRY = {
    "mlp": ("models.mlp", "SmallMLP", ["hidden"]),
    "cnn1d": ("models.cnn1d", "CNN1D", []),
    "transformer": ("models.transformer", "TransformerClassifier", ["d_model", "n_heads", "n_layers"]),
    "resnet1d": ("models.resnet1d", "ResNet1D", []),
}


def build_model(cfg: dict) -> nn.Module:
    name = cfg["model"]
    if name not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model '{name}'. Options: {list(MODEL_REGISTRY)}")
    module_name, class_name, keys = MODEL_REGISTRY[name]
    module = importlib.import_module(module_name)
    if not hasattr(module, class_name):
        raise SystemExit(
            f"Class '{class_name}' is not implemented yet in src/models/{module_name.split('.')[-1]}.py"
        )
    kwargs = {k: cfg[k] for k in keys if k in cfg}
    return getattr(module, class_name)(num_classes=NUM_CLASSES, **kwargs)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_optimizer(cfg: dict, params):
    name = cfg.get("optimizer", "adam").lower()
    lr, wd = cfg.get("lr", 1e-3), cfg.get("weight_decay", 0.0)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    raise SystemExit(f"Unknown optimizer '{name}' (adam | adamw | sgd)")


def make_criterion(cfg: dict, data_path, device):
    loss = cfg.get("loss", "cross_entropy")
    if loss == "cross_entropy":
        return nn.CrossEntropyLoss()
    if loss == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights(data_path, NUM_CLASSES).to(device))
    raise SystemExit(f"Unknown loss '{loss}' (cross_entropy | weighted_cross_entropy)")


def run_epoch(model, loader, criterion, device, optimizer=None):
    """One pass over `loader`. Trains if an optimizer is given, else evaluates."""
    train = optimizer is not None
    model.train(train)
    total_loss, n = 0.0, 0
    preds, targets = [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            n += len(y)
            preds.append(logits.argmax(1).cpu())
            targets.append(y.cpu())
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    return {
        "loss": total_loss / n,
        "acc": float((preds == targets).mean()),
        "macro_f1": float(f1_score(targets, preds, average="macro")),
    }


def train_one_seed(cfg, seed, exp_name, device, data_path):
    set_seed(seed)
    loaders = get_dataloaders(
        data_path, batch_size=cfg.get("batch_size", 64),
        augmentation=cfg.get("augmentation", "none"),
        num_workers=cfg.get("num_workers", 0), seed=seed,
    )
    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = make_optimizer(cfg, model.parameters())
    criterion = make_criterion(cfg, data_path, device)

    epochs = cfg.get("epochs", 50)
    patience = cfg.get("patience", 10)

    exp_dir = ROOT / "experiments" / exp_name
    ckpt_dir = ROOT / "checkpoints"
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{exp_name}_seed{seed}.pt"
    log_path = exp_dir / f"seed{seed}_log.csv"

    print(f"\n=== {exp_name} | seed {seed} | {n_params:,} params | device {device} ===")
    best, best_epoch, bad = {"macro_f1": -1.0}, 0, 0
    start = time.time()
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "train_macro_f1",
                         "val_loss", "val_acc", "val_macro_f1"])
        for epoch in range(1, epochs + 1):
            tr = run_epoch(model, loaders["train"], criterion, device, optimizer)
            va = run_epoch(model, loaders["val"], criterion, device)
            writer.writerow([epoch, tr["loss"], tr["acc"], tr["macro_f1"],
                             va["loss"], va["acc"], va["macro_f1"]])
            f.flush()
            print(f"epoch {epoch:3d} | train loss {tr['loss']:.4f} f1 {tr['macro_f1']:.4f} "
                  f"| val loss {va['loss']:.4f} acc {va['acc']:.4f} f1 {va['macro_f1']:.4f}")
            if va["macro_f1"] > best["macro_f1"]:
                best, best_epoch, bad = va, epoch, 0
                torch.save({"model_state": model.state_dict(), "config": cfg, "seed": seed,
                            "epoch": epoch, "val_macro_f1": va["macro_f1"]}, ckpt_path)
            else:
                bad += 1
                if bad >= patience:
                    print(f"Early stopping at epoch {epoch} (no val improvement for {patience} epochs)")
                    break
    train_time = time.time() - start

    summary = {
        "experiment": exp_name, "model": cfg["model"], "seed": seed,
        "best_epoch": best_epoch, "best_val_macro_f1": best["macro_f1"],
        "best_val_acc": best["acc"], "best_val_loss": best["loss"],
        "train_time_sec": round(train_time, 1), "epochs_run": epoch,
        "params": n_params, "model_size_mb": round(ckpt_path.stat().st_size / 1e6, 3),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    (exp_dir / f"seed{seed}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Best val macro-F1 {best['macro_f1']:.4f} at epoch {best_epoch} | "
          f"train time {train_time:.0f}s | saved {ckpt_path.name}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="override seeds from the config")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs from the config")
    ap.add_argument("--data", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--device", default=None, help="cpu | cuda (default: cuda if available)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    seeds = args.seeds if args.seeds is not None else cfg.get("seeds", [0, 1, 2])
    exp_name = args.config.stem  # one config file per experiment
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if not Path(args.data).exists():
        raise SystemExit(f"{args.data} not found. Run: python data\\preprocess.py")

    summaries = [train_one_seed(cfg, s, exp_name, device, args.data) for s in seeds]

    f1s = np.array([s["best_val_macro_f1"] for s in summaries])
    agg = {"experiment": exp_name, "seeds": seeds,
           "val_macro_f1_mean": float(f1s.mean()), "val_macro_f1_std": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
           "per_seed": summaries}
    (ROOT / "experiments" / exp_name / "val_summary.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
    print(f"\n{exp_name}: val macro-F1 = {agg['val_macro_f1_mean']:.4f} ± {agg['val_macro_f1_std']:.4f} "
          f"over seeds {seeds}")


if __name__ == "__main__":
    main()