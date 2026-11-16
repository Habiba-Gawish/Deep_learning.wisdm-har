"""Preprocessing for WISDM (UCI #507) -> windows shared by ALL four models.

Pipeline
--------
1. Read the raw files  data_<subject>_<sensor>_<device>.txt  (4 per subject:
   phone/accel, phone/gyro, watch/accel, watch/gyro). Line format:
       subject,activity_code,timestamp_ns,x,y,z;
2. For every (subject, activity) segment, put the 4 streams on ONE common
   20 Hz time grid (linear interpolation on time elapsed since the start of
   the segment; phone and watch clocks are not synchronised, so absolute
   timestamps cannot be compared directly).
   -> 12 channels = phone_accel(xyz) + phone_gyro(xyz) + watch_accel(xyz) + watch_gyro(xyz)
3. Cut each segment into fixed windows of 200 timesteps (10 s at 20 Hz).
   A window never crosses an activity boundary, so it has exactly one label.
   -> one window = [200, 12] = 2400 values (MLP input after flatten).
4. Leakage-free split: BY SUBJECT (a person is only in train OR val OR test).
5. Per-channel z-score normalisation, mean/std computed on TRAIN only.
6. Save everything to data/processed/.

Saved layout: X has shape [N, 200, 12] (time-major). Models that need
channels-first (Conv1d) just do  x.permute(0, 2, 1) -> [N, 12, 200].

Usage (Command Prompt, from the project root):
    python data\\preprocess.py
    python data\\preprocess.py --window 200 --stride 100 --seed 42
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_RAW = ROOT / "raw"
DEFAULT_OUT = ROOT / "processed"

FS = 20  # sampling rate (Hz)

# Official activity codes (activity_key.txt). There is no 'N'.
ACTIVITY_NAMES = {
    "A": "walking", "B": "jogging", "C": "stairs", "D": "sitting",
    "E": "standing", "F": "typing", "G": "brushing teeth",
    "H": "eating soup", "I": "eating chips", "J": "eating pasta",
    "K": "drinking from cup", "L": "eating sandwich",
    "M": "kicking soccer ball", "O": "playing catch w/ tennis ball",
    "P": "dribbling basketball", "Q": "writing", "R": "clapping",
    "S": "folding clothes",
}
ACTIVITY_CODES = sorted(ACTIVITY_NAMES)  # 18 codes -> labels 0..17
CODE_TO_LABEL = {c: i for i, c in enumerate(ACTIVITY_CODES)}

# (device, sensor) order defines the channel order.
STREAMS = [("phone", "accel"), ("phone", "gyro"), ("watch", "accel"), ("watch", "gyro")]
CHANNEL_NAMES = [f"{dev}_{sen}_{ax}" for dev, sen in STREAMS for ax in "xyz"]


# --------------------------------------------------------------------------
# 1. Reading raw files
# --------------------------------------------------------------------------
def find_raw_files(raw_dir: Path) -> dict:
    """Return {(subject_id, device, sensor): path} for every raw file found."""
    files = {}
    for p in raw_dir.rglob("data_*_*_*.txt"):
        parts = p.stem.split("_")  # data, subject, sensor, device
        if len(parts) != 4 or not parts[1].isdigit():
            continue
        _, subj, sensor, device = parts
        if (device, sensor) in STREAMS:
            files[(int(subj), device, sensor)] = p
    return files


def read_stream(path: Path) -> dict:
    """Read one raw file -> {activity_code: (t_seconds[n], xyz[n, 3])}.

    t is the time elapsed since the first sample of that activity segment.
    """
    df = pd.read_csv(
        path, header=None, dtype=str, on_bad_lines="skip",
        names=["subject", "activity", "timestamp", "x", "y", "z"],
    )
    df["z"] = df["z"].str.rstrip(";").str.strip()  # lines end with ';'
    df["activity"] = df["activity"].str.strip()
    for c in ["timestamp", "x", "y", "z"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["activity", "timestamp", "x", "y", "z"])

    segments = {}
    for code, g in df.groupby("activity"):
        if code not in CODE_TO_LABEL:
            continue
        g = g.sort_values("timestamp").drop_duplicates("timestamp")
        if len(g) < 2:
            continue
        t = (g["timestamp"].to_numpy(dtype=np.float64) - g["timestamp"].iloc[0]) / 1e9
        segments[code] = (t, g[["x", "y", "z"]].to_numpy(dtype=np.float64))
    return segments


# --------------------------------------------------------------------------
# 2-3. Alignment (12 channels, 20 Hz) + windowing
# --------------------------------------------------------------------------
def align_segment(per_stream: list) -> np.ndarray:
    """per_stream: 4 x (t, xyz) -> array [n, 12] on a common 20 Hz grid."""
    duration = min(t[-1] for t, _ in per_stream)  # shortest of the 4 streams
    n = int(duration * FS) + 1
    grid = np.arange(n) / FS
    cols = []
    for t, xyz in per_stream:
        for k in range(3):
            cols.append(np.interp(grid, t, xyz[:, k]))
    return np.stack(cols, axis=1).astype(np.float32)


def make_windows(signal: np.ndarray, window: int, stride: int) -> np.ndarray:
    """[n, 12] -> [num_windows, window, 12]; incomplete tail is dropped."""
    n = signal.shape[0]
    if n < window:
        return np.empty((0, window, signal.shape[1]), dtype=np.float32)
    starts = range(0, n - window + 1, stride)
    return np.stack([signal[s:s + window] for s in starts])


def process_subject(subject: int, files: dict, window: int, stride: int):
    """Return (X [m, window, 12], y [m]) for one subject, or None if unusable."""
    streams = []
    for dev, sen in STREAMS:
        path = files.get((subject, dev, sen))
        if path is None:
            print(f"  [skip] subject {subject}: missing file for {dev}/{sen}")
            return None
        streams.append(read_stream(path))

    common = set(streams[0])
    for s in streams[1:]:
        common &= set(s)

    X_parts, y_parts = [], []
    for code in sorted(common):
        sig = align_segment([s[code] for s in streams])
        w = make_windows(sig, window, stride)
        if len(w) == 0:
            continue
        X_parts.append(w)
        y_parts.append(np.full(len(w), CODE_TO_LABEL[code], dtype=np.int64))
    if not X_parts:
        return None
    return np.concatenate(X_parts), np.concatenate(y_parts)


# --------------------------------------------------------------------------
# 4. Subject-wise split
# --------------------------------------------------------------------------
def split_subjects(subjects, seed: int, val_frac: float, test_frac: float):
    rng = np.random.RandomState(seed)
    subs = np.array(sorted(subjects))
    rng.shuffle(subs)
    n_test = int(round(test_frac * len(subs)))
    n_val = int(round(val_frac * len(subs)))
    test = sorted(subs[:n_test].tolist())
    val = sorted(subs[n_test:n_test + n_val].tolist())
    train = sorted(subs[n_test + n_val:].tolist())
    return train, val, test


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--window", type=int, default=200, help="timesteps per window (200 = 10 s)")
    ap.add_argument("--stride", type=int, default=100, help="step between windows (100 = 50%% overlap)")
    ap.add_argument("--seed", type=int, default=42, help="seed for the subject split")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    args = ap.parse_args()

    files = find_raw_files(args.raw_dir)
    if not files:
        raise SystemExit(f"No raw WISDM files found under {args.raw_dir}. Run: python data\\download.py")
    subjects = sorted({s for s, _, _ in files})
    print(f"Found {len(files)} raw files for {len(subjects)} subjects.")

    # --- build windows per subject
    per_subject = {}
    for i, subj in enumerate(subjects, 1):
        out = process_subject(subj, files, args.window, args.stride)
        if out is None:
            continue
        per_subject[subj] = out
        print(f"[{i}/{len(subjects)}] subject {subj}: {len(out[1])} windows")
    usable = sorted(per_subject)
    if len(usable) < 3:
        raise SystemExit("Need at least 3 usable subjects to make train/val/test splits.")

    # --- split by subject (no person appears in two splits)
    tr_s, va_s, te_s = split_subjects(usable, args.seed, args.val_frac, args.test_frac)
    assert not (set(tr_s) & set(va_s)) and not (set(tr_s) & set(te_s)) and not (set(va_s) & set(te_s))

    def gather(subs):
        X = np.concatenate([per_subject[s][0] for s in subs])
        y = np.concatenate([per_subject[s][1] for s in subs])
        g = np.concatenate([np.full(len(per_subject[s][1]), s, dtype=np.int64) for s in subs])
        return X, y, g

    X_tr, y_tr, g_tr = gather(tr_s)
    X_va, y_va, g_va = gather(va_s)
    X_te, y_te, g_te = gather(te_s)

    # --- normalise with TRAIN statistics only
    mean = X_tr.reshape(-1, X_tr.shape[-1]).mean(axis=0)
    std = X_tr.reshape(-1, X_tr.shape[-1]).std(axis=0) + 1e-8
    X_tr, X_va, X_te = [((x - mean) / std).astype(np.float32) for x in (X_tr, X_va, X_te)]

    # --- save
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "wisdm_windows.npz",
        X_train=X_tr, y_train=y_tr, subj_train=g_tr,
        X_val=X_va, y_val=y_va, subj_val=g_va,
        X_test=X_te, y_test=y_te, subj_test=g_te,
        mean=mean.astype(np.float32), std=std.astype(np.float32),
    )

    def class_counts(y):
        return {ACTIVITY_NAMES[c]: int((y == CODE_TO_LABEL[c]).sum()) for c in ACTIVITY_CODES}

    meta = {
        "dataset": "WISDM Smartphone and Smartwatch Activity and Biometrics (UCI 507)",
        "sampling_rate_hz": FS,
        "window": args.window,
        "stride": args.stride,
        "seed": args.seed,
        "x_shape_per_window": [args.window, len(CHANNEL_NAMES)],
        "channels": CHANNEL_NAMES,
        "labels": {str(CODE_TO_LABEL[c]): {"code": c, "name": ACTIVITY_NAMES[c]} for c in ACTIVITY_CODES},
        "split": {
            "train_subjects": tr_s, "val_subjects": va_s, "test_subjects": te_s,
            "n_windows": {"train": len(y_tr), "val": len(y_va), "test": len(y_te)},
        },
        "class_counts": {"train": class_counts(y_tr), "val": class_counts(y_va), "test": class_counts(y_te)},
        "normalisation": "per-channel z-score, mean/std from train only (saved in npz as mean/std)",
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # --- summary
    print("\nDone.")
    print(f"  train: {X_tr.shape}  ({len(tr_s)} subjects)")
    print(f"  val  : {X_va.shape}  ({len(va_s)} subjects)")
    print(f"  test : {X_te.shape}  ({len(te_s)} subjects)")
    print(f"  saved to {args.out_dir}\\wisdm_windows.npz and meta.json")


if __name__ == "__main__":
    main()