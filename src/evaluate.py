"""Shared evaluation for ALL models, on the TEST split.

    python src\\evaluate.py --config configs\\mlp.yaml
    python src\\evaluate.py --config configs\\cnn1d.yaml --seeds 0 1 2

Loads checkpoints\\<config-name>_seed<k>.pt (made by train.py) and reports:
  * macro-F1, accuracy, weighted-F1 per seed and mean ± SD over seeds
  * per-class precision / recall / F1 (mean ± SD over seeds)
  * confusion matrix (summed over seeds) as CSV and PNG
  * computational cost: parameters, model size, inference time, train time, hardware
  * raw test predictions per seed (for the error analysis)

Saved in experiments\\<config-name>\\ :
    test_results.json, per_class_test.csv, confusion_matrix_test.csv,
    confusion_matrix_test.png, test_preds_seed<k>.npz
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from datasets.wisdm import DEFAULT_NPZ, WISDMDataset, load_meta  # noqa: E402
from train import NUM_CLASSES, build_model  # noqa: E402


def mean_std(values):
    v = np.asarray(values, dtype=np.float64)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


@torch.no_grad()
def predict(model, loader, device):
    """Return (y_true, y_pred) in the SAME order as the test set (no shuffling)."""
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        targets.append(y)
    return torch.cat(targets).numpy(), torch.cat(preds).numpy()


@torch.no_grad()
def inference_ms_per_window(model, device, n_warmup=20, n_runs=200):
    """Average latency for ONE window (batch size 1), in milliseconds."""
    model.eval()
    x = torch.randn(1, 200, 12, device=device)
    for _ in range(n_warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_runs * 1000


def class_names(npz_path):
    try:
        labels = load_meta(npz_path)["labels"]
        return [labels[str(i)]["name"] for i in range(NUM_CLASSES)]
    except Exception:
        return [str(i) for i in range(NUM_CLASSES)]


def save_confusion_png(cm, names, title, path):
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)  # rows sum to 1
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, label="fraction of true class")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--data", type=Path, default=DEFAULT_NPZ)
    ap.add_argument("--device", default=None, help="cpu | cuda (default: cuda if available)")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    cfg_file = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seeds = args.seeds if args.seeds is not None else cfg_file.get("seeds", [0, 1, 2])
    exp_name = args.config.stem
    exp_dir = ROOT / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    hardware = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"

    test_loader = DataLoader(WISDMDataset(args.data, "test"), batch_size=args.batch_size, shuffle=False)
    names = class_names(args.data)
    labels = list(range(NUM_CLASSES))

    per_seed, f1_cls, prec_cls, rec_cls = [], [], [], []
    cm_total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    support, cost = None, {}

    for seed in seeds:
        ckpt_path = ROOT / "checkpoints" / f"{exp_name}_seed{seed}.pt"
        if not ckpt_path.exists():
            print(f"[skip] {ckpt_path.name} not found (train it first)")
            continue
        ckpt = torch.load(ckpt_path, map_location=device)
        model = build_model(ckpt["config"]).to(device)  # architecture from the saved config
        model.load_state_dict(ckpt["model_state"])

        y_true, y_pred = predict(model, test_loader, device)
        np.savez(exp_dir / f"test_preds_seed{seed}.npz", y_true=y_true, y_pred=y_pred)

        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        f1_cls.append(f)
        prec_cls.append(p)
        rec_cls.append(r)
        support = s
        cm_total += confusion_matrix(y_true, y_pred, labels=labels)

        m = {
            "seed": seed,
            "test_macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "test_accuracy": float(accuracy_score(y_true, y_pred)),
            "test_weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
            "val_macro_f1_at_checkpoint": float(ckpt.get("val_macro_f1", float("nan"))),
            "best_epoch": int(ckpt.get("epoch", -1)),
        }
        summ = exp_dir / f"seed{seed}_summary.json"
        if summ.exists():
            m["train_time_sec"] = json.loads(summ.read_text(encoding="utf-8")).get("train_time_sec")
        per_seed.append(m)
        print(f"seed {seed}: test macro-F1 {m['test_macro_f1']:.4f} | acc {m['test_accuracy']:.4f} "
              f"| (val macro-F1 {m['val_macro_f1_at_checkpoint']:.4f})")

        if not cost:  # same architecture for every seed -> measure once
            cost = {
                "params": sum(q.numel() for q in model.parameters()),
                "model_size_mb": round(ckpt_path.stat().st_size / 1e6, 3),
                "inference_ms_per_window": round(inference_ms_per_window(model, device), 3),
                "hardware": hardware,
            }

    if not per_seed:
        raise SystemExit("No checkpoints found. Train first: python src\\train.py --config " + str(args.config))

    agg = {}
    for key in ("test_macro_f1", "test_accuracy", "test_weighted_f1"):
        mu, sd = mean_std([m[key] for m in per_seed])
        agg[f"{key}_mean"], agg[f"{key}_std"] = mu, sd
    times = [m["train_time_sec"] for m in per_seed if m.get("train_time_sec") is not None]
    if times:
        cost["train_time_sec_mean"] = round(float(np.mean(times)), 1)

    f1_cls, prec_cls, rec_cls = map(np.array, (f1_cls, prec_cls, rec_cls))
    per_class = {}
    for i, n in enumerate(names):
        f_mu, f_sd = mean_std(f1_cls[:, i])
        per_class[n] = {"precision_mean": float(prec_cls[:, i].mean()), "recall_mean": float(rec_cls[:, i].mean()),
                        "f1_mean": f_mu, "f1_std": f_sd, "support": int(support[i])}

    result = {"experiment": exp_name, "model": ckpt["config"]["model"], "split": "test",
              "seeds": [m["seed"] for m in per_seed], **agg, "cost": cost,
              "per_class": per_class, "per_seed": per_seed}
    (exp_dir / "test_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    with open(exp_dir / "per_class_test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision_mean", "recall_mean", "f1_mean", "f1_std", "support"])
        for n, d in per_class.items():
            w.writerow([n, f"{d['precision_mean']:.4f}", f"{d['recall_mean']:.4f}",
                        f"{d['f1_mean']:.4f}", f"{d['f1_std']:.4f}", d["support"]])

    with open(exp_dir / "confusion_matrix_test.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + names)
        for n, row in zip(names, cm_total):
            w.writerow([n] + row.tolist())
    save_confusion_png(cm_total, names, f"{exp_name} - test confusion matrix (sum over {len(per_seed)} seeds)",
                       exp_dir / "confusion_matrix_test.png")

    print(f"\n{exp_name} on TEST ({len(per_seed)} seeds)")
    print(f"  macro-F1 : {agg['test_macro_f1_mean']:.4f} ± {agg['test_macro_f1_std']:.4f}")
    print(f"  accuracy : {agg['test_accuracy_mean']:.4f} ± {agg['test_accuracy_std']:.4f}")
    print(f"  params {cost['params']:,} | size {cost['model_size_mb']} MB | "
          f"inference {cost['inference_ms_per_window']} ms/window on {hardware}")
    worst = sorted(per_class.items(), key=lambda kv: kv[1]["f1_mean"])[:3]
    print("  weakest classes: " + ", ".join(f"{n} (F1 {d['f1_mean']:.2f})" for n, d in worst))
    print(f"  saved to {exp_dir}")


if __name__ == "__main__":
    main()