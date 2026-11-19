"""Interactive demo: every trained model classifies the SAME sensor window, side by side.

    python demo\\app.py                 (then open the printed http://127.0.0.1:7860 link)
    python demo\\app.py --port 7861     (if 7860 is busy)
    python demo\\app.py --share         (temporary public link, e.g. for a remote viva)

What it needs (all produced by the normal pipeline):
    data/processed/wisdm_windows.npz + meta.json    <- python data\\preprocess.py
    checkpoints/<model>_seed<k>.pt                  <- python src\\train.py --config configs\\<model>.yaml
    experiments/<model>/test_results.json           <- python src\\evaluate.py --config configs\\<model>.yaml

A model without a checkpoint is shown as "no checkpoint yet" instead of crashing the demo.
The demo only reads the TEST split and the saved checkpoints; it never trains anything.
"""
import argparse
import html
import json
import sys
import tempfile
import time
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from train import build_model  # noqa: E402  (same constructor logic as training / evaluation)

NPZ_PATH = ROOT / "data" / "processed" / "wisdm_windows.npz"
META_PATH = ROOT / "data" / "processed" / "meta.json"
CKPT_DIR = ROOT / "checkpoints"
EXP_DIR = ROOT / "experiments"

NUM_CLASSES = 18
SEEDS = (0, 1, 2)

# key (= config / checkpoint prefix), display name, role
MODELS = [
    ("mlp", "Small MLP", "Baseline"),
    ("cnn1d", "1D-CNN", "Classic"),
    ("gru", "GRU", "Classic"),
    ("transformer", "Transformer", "Modern"),
    ("resnet1d", "ResNet1D", "Advanced"),
]

FALLBACK_CLASSES = [
    "walking", "jogging", "stairs", "sitting", "standing", "typing", "brushing teeth",
    "eating soup", "eating chips", "eating pasta", "drinking from cup", "eating sandwich",
    "kicking soccer ball", "playing catch w/ tennis ball", "dribbling basketball",
    "writing", "clapping", "folding clothes",
]
FALLBACK_CHANNELS = [f"{d}_{s}_{a}" for d, s in
                     [("phone", "accel"), ("phone", "gyro"), ("watch", "accel"), ("watch", "gyro")]
                     for a in "xyz"]

SEED_CHOICES = ["Seed 0", "Seed 1", "Seed 2", "Average of 3 seeds"]
ALL_SUBJECTS = "All subjects"


# ---------------------------------------------------------------------------
# Data (test split only)
# ---------------------------------------------------------------------------
def load_data():
    """Return (data dict or None, error message or None)."""
    if not NPZ_PATH.exists():
        return None, f"{NPZ_PATH} not found. Run: python data\\preprocess.py"
    try:
        with np.load(NPZ_PATH) as d:
            data = {
                "X": d["X_test"].astype(np.float32),
                "y": d["y_test"].astype(np.int64),
                "subj": d["subj_test"].astype(np.int64) if "subj_test" in d.files
                else np.zeros(len(d["y_test"]), dtype=np.int64),
                "mean": d["mean"].astype(np.float32),
                "std": d["std"].astype(np.float32),
            }
    except Exception as e:  # corrupted / incompatible file
        return None, f"Could not read {NPZ_PATH.name}: {e}"
    names, channels = FALLBACK_CLASSES, FALLBACK_CHANNELS
    try:
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
        names = [meta["labels"][str(i)]["name"] for i in range(NUM_CLASSES)]
        channels = meta.get("channels", FALLBACK_CHANNELS)
    except Exception:
        pass
    data["names"], data["channels"] = names, channels
    return data, None


DATA, DATA_ERROR = load_data()
CLASS_NAMES = DATA["names"] if DATA else FALLBACK_CLASSES
CHANNELS = DATA["channels"] if DATA else FALLBACK_CHANNELS


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
_CACHE = {}


def get_model(key, seed):
    """Load (and cache) one checkpoint. Returns a dict or None if the file does not exist."""
    if (key, seed) in _CACHE:
        return _CACHE[(key, seed)]
    path = CKPT_DIR / f"{key}_seed{seed}.pt"
    if not path.exists():
        return None  # not cached, so a checkpoint added later is picked up
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    entry = {
        "model": model,
        "params": sum(p.numel() for p in model.parameters()),
        "size_mb": path.stat().st_size / 1e6,
        "epoch": ckpt.get("epoch"),
        "val_f1": ckpt.get("val_macro_f1"),
    }
    _CACHE[(key, seed)] = entry
    return entry


def seeds_from_choice(choice):
    if choice.startswith("Average"):
        return list(SEEDS)
    return [int(choice.split()[-1])]


@torch.no_grad()
def run_model(key, seeds, x):
    """Softmax probabilities (averaged over the chosen seeds) + latency for one window.

    Returns None if no checkpoint is available for the requested seeds.
    """
    probs, lat, params = [], [], None
    for s in seeds:
        entry = get_model(key, s)
        if entry is None:
            continue
        params = entry["params"]
        times = []
        for _ in range(3):  # median of 3 forward passes, batch size 1
            t0 = time.perf_counter()
            logits = entry["model"](x)
            times.append((time.perf_counter() - t0) * 1000)
        probs.append(torch.softmax(logits, dim=1)[0].numpy())
        lat.append(float(np.median(times)))
    if not probs:
        return None
    return {"probs": np.mean(probs, axis=0), "latency": float(np.mean(lat)),
            "n_seeds": len(probs), "params": params}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
CSS = """
<style>
.hero{padding:22px 26px;border-radius:14px;margin-bottom:6px;color:#fff;
  background:linear-gradient(120deg,#1e3a8a 0%,#4338ca 55%,#7c3aed 100%)}
.hero h1{margin:0 0 6px 0;font-size:1.65rem;font-weight:700;color:#fff}
.hero p{margin:0;opacity:.92;font-size:.98rem;color:#fff}
.hero,.hero *{color:#fff !important}
.hero .pills{margin-top:12px;display:flex;flex-wrap:wrap;gap:8px}
.hero .pill{background:rgba(255,255,255,.16);border-radius:999px;padding:3px 12px;font-size:.82rem}
.banner{padding:14px 18px;border-radius:12px;border:1px solid var(--border-color-primary);
  background:var(--background-fill-secondary);line-height:1.55}
.banner .big{font-size:1.15rem;font-weight:700}
.banner.err{border-color:#dc2626;background:rgba(220,38,38,.10)}
.muted{opacity:.75;font-size:.88rem}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:14px;margin-top:8px}
.card{border:1px solid var(--border-color-primary);border-radius:14px;padding:14px 15px;
  background:var(--background-fill-secondary);display:flex;flex-direction:column;gap:8px;
  border-top-width:4px}
.card.baseline{border-top-color:#6b7280}.card.classic{border-top-color:#2563eb}
.card.modern{border-top-color:#9333ea}.card.advanced{border-top-color:#16a34a}
.card.missing{opacity:.6;border-style:dashed}
.card .head{display:flex;justify-content:space-between;align-items:center;gap:8px}
.card .name{font-weight:700;font-size:1.02rem}
.chip{font-size:.7rem;font-weight:600;padding:2px 9px;border-radius:999px;letter-spacing:.02em}
.chip.baseline{background:rgba(107,114,128,.2)}.chip.classic{background:rgba(37,99,235,.18)}
.chip.modern{background:rgba(147,51,234,.18)}.chip.advanced{background:rgba(22,163,74,.2)}
.badge{align-self:flex-start;font-size:.78rem;font-weight:700;padding:2px 10px;border-radius:8px}
.badge.ok{background:rgba(22,163,74,.2);color:#15803d}
.badge.bad{background:rgba(220,38,38,.18);color:#b91c1c}
.pred{font-size:1.12rem;font-weight:700;line-height:1.25}
.conf{font-size:.85rem;opacity:.8}
.bar{margin:3px 0}
.bar .lbl{display:flex;justify-content:space-between;font-size:.8rem;gap:6px}
.bar .track{height:7px;border-radius:5px;background:rgba(128,128,128,.22);overflow:hidden}
.bar .fill{height:100%;border-radius:5px;background:#4f46e5}
.bar.true .fill{background:#16a34a}
.bar.true .lbl{font-weight:700}
.foot{margin-top:auto;padding-top:6px;border-top:1px dashed var(--border-color-primary);
  display:flex;justify-content:space-between;font-size:.78rem;opacity:.8}
table.cmp{width:100%;border-collapse:collapse;font-size:.92rem}
table.cmp th,table.cmp td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--border-color-primary)}
table.cmp th:first-child,table.cmp td:first-child{text-align:left}
table.cmp th{font-size:.8rem;text-transform:uppercase;letter-spacing:.04em;opacity:.8}
table.cmp tr.best td{font-weight:700}
</style>
"""

HERO = CSS + """
<div class="hero">
  <h1>WISDM Wearable Activity Recognition</h1>
  <p>One 10-second sensor window. Five models trained from scratch. See what each one predicts.</p>
  <div class="pills">
    <span class="pill">18 activities</span><span class="pill">phone + watch, 12 channels, 20 Hz</span>
    <span class="pill">subject-wise split</span><span class="pill">3 seeds per model</span>
    <span class="pill">Deep Learning (AI403), Pharos University</span>
  </div>
</div>
"""


def fmt_int(n):
    return f"{int(n):,}"


def banner(html_body, error=False):
    return f'<div class="banner{" err" if error else ""}">{html_body}</div>'


def signal_figure(window):
    """window: [200, 12] (z-scored, exactly what the models receive)."""
    fig = Figure(figsize=(9.6, 5.6), dpi=100, facecolor="white")
    axes = fig.subplots(4, 1, sharex=True)
    t = np.arange(window.shape[0]) / 20.0
    titles = ["Phone accelerometer", "Phone gyroscope", "Watch accelerometer", "Watch gyroscope"]
    colors = ["#2563eb", "#16a34a", "#dc2626"]
    for i, ax in enumerate(axes):
        for j, (c, axis_name) in enumerate(zip(colors, "xyz")):
            ax.plot(t, window[:, i * 3 + j], color=c, lw=1.0, label=axis_name)
        ax.set_ylabel(titles[i].replace(" ", "\n", 1), fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(ncol=3, fontsize=8, loc="upper right", frameon=False)
    axes[-1].set_xlabel("time (s)", fontsize=9)
    fig.suptitle("Input window (z-scored, 200 steps x 12 channels)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def heatmap_figure(results, true_idx):
    """Rows = models, columns = classes, colour = predicted probability."""
    rows = [(name, r) for (_, name, _), r in zip(MODELS, results) if r is not None]
    if not rows:
        return None
    mat = np.stack([r["probs"] for _, r in rows])
    fig = Figure(figsize=(9.6, 0.62 * len(rows) + 3.2), dpi=100, facecolor="white")
    ax = fig.subplots()
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([n for n, _ in rows], fontsize=9)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=55, ha="right", fontsize=8)
    for i in range(len(rows)):
        top = int(mat[i].argmax())
        ax.text(top, i, f"{mat[i, top] * 100:.0f}%", ha="center", va="center",
                fontsize=8, color="white" if mat[i, top] > 0.5 else "black")
    if true_idx is not None:
        ax.add_patch(Rectangle(
            (true_idx - 0.5, -0.5), 1, len(rows), fill=False, edgecolor="#16a34a", lw=2.4))
    ax.set_title("Probability given to every activity" + (" (green box = true activity)" if true_idx is not None else ""),
                 fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    fig.tight_layout()
    return fig


def card_html(name, role, res, true_idx):
    cls = role.lower()
    if res is None:
        return (f'<div class="card missing {cls}"><div class="head"><span class="name">{html.escape(name)}</span>'
                f'<span class="chip {cls}">{role}</span></div>'
                f'<div class="pred">no checkpoint yet</div>'
                f'<div class="conf">Train it first:<br>python src\\train.py --config configs\\&lt;model&gt;.yaml</div></div>')
    probs = res["probs"]
    order = np.argsort(-probs)[:3]
    top = int(order[0])
    badge = ""
    if true_idx is not None:
        badge = ('<span class="badge ok">&#10003; Correct</span>' if top == true_idx
                 else '<span class="badge bad">&#10007; Wrong</span>')
    bars = []
    for i in order:
        i = int(i)
        mark = " (true)" if i == true_idx else ""
        bars.append(
            f'<div class="bar{" true" if i == true_idx else ""}"><div class="lbl"><span>{html.escape(CLASS_NAMES[i])}{mark}</span>'
            f'<span>{probs[i] * 100:.1f}%</span></div>'
            f'<div class="track"><div class="fill" style="width:{probs[i] * 100:.1f}%"></div></div></div>')
    if true_idx is not None and true_idx not in [int(i) for i in order]:
        bars.append(
            f'<div class="bar true"><div class="lbl"><span>{html.escape(CLASS_NAMES[true_idx])} (true)</span>'
            f'<span>{probs[true_idx] * 100:.1f}%</span></div>'
            f'<div class="track"><div class="fill" style="width:{probs[true_idx] * 100:.1f}%"></div></div></div>')
    n_note = f" &middot; {res['n_seeds']} seeds avg" if res["n_seeds"] > 1 else ""
    return (f'<div class="card {cls}"><div class="head"><span class="name">{html.escape(name)}</span>'
            f'<span class="chip {cls}">{role}</span></div>{badge}'
            f'<div class="pred">{html.escape(CLASS_NAMES[top])}</div>'
            f'<div class="conf">confidence {probs[top] * 100:.1f}%{n_note}</div>'
            f'{"".join(bars)}'
            f'<div class="foot"><span>{res["latency"]:.2f} ms / window</span>'
            f'<span>{fmt_int(res["params"])} params</span></div></div>')


def consensus_html(results, true_idx):
    votes = {}
    for r in results:
        if r is not None:
            votes.setdefault(int(r["probs"].argmax()), 0)
            votes[int(r["probs"].argmax())] += 1
    n = sum(votes.values())
    if n == 0:
        return ""
    best, cnt = max(votes.items(), key=lambda kv: kv[1])
    text = f'<b>{cnt} of {n}</b> models predict <b>{html.escape(CLASS_NAMES[best])}</b>'
    if len(votes) > 1:
        others = ", ".join(f"{html.escape(CLASS_NAMES[k])} ({v})" for k, v in sorted(votes.items(), key=lambda kv: -kv[1])[1:])
        text += f' &middot; others: {others}'
    if true_idx is not None:
        n_ok = sum(1 for r in results if r is not None and int(r["probs"].argmax()) == true_idx)
        text += f' &middot; <b>{n_ok} of {n}</b> match the true activity'
    return text


def render(window, true_idx, title_html, seed_choice):
    """Run all models on one normalised window [200, 12] -> (banner, signal fig, cards, heatmap)."""
    seeds = seeds_from_choice(seed_choice)
    x = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32)).unsqueeze(0)
    results = [run_model(key, seeds, x) for key, _, _ in MODELS]
    if all(r is None for r in results):
        return (banner("No trained checkpoints found in <code>checkpoints/</code> for the selected seed. "
                       "Train the models first (see README, section 5).", error=True), None, "", None)
    cards = '<div class="grid">' + "".join(
        card_html(name, role, r, true_idx) for (_, name, role), r in zip(MODELS, results)) + "</div>"
    head = banner(f'{title_html}<br><span class="muted">{consensus_html(results, true_idx)}</span>')
    return head, signal_figure(window), cards, heatmap_figure(results, true_idx)


NO_DATA = (banner(html.escape(DATA_ERROR or ""), error=True), None, "", None)


# ---------------------------------------------------------------------------
# Tab 1: a window from the held-out TEST subjects
# ---------------------------------------------------------------------------
def class_indices(cls_name, subj):
    y = DATA["y"]
    mask = y == CLASS_NAMES.index(cls_name)
    if subj and subj != ALL_SUBJECTS:
        mask &= DATA["subj"] == int(subj)
    return np.where(mask)[0]


def subject_choices(cls_name):
    idx = class_indices(cls_name, ALL_SUBJECTS)
    return [ALL_SUBJECTS] + [str(s) for s in sorted(set(DATA["subj"][idx].tolist()))]


def classify_sample(cls_name, subj, k, seed_choice):
    if DATA is None:
        return NO_DATA
    idx = class_indices(cls_name, subj)
    if len(idx) == 0:
        return (banner("No test window for this combination.", error=True), None, "", None)
    k = int(min(max(int(k or 1), 1), len(idx)))
    g = int(idx[k - 1])
    title = (f'True activity: <span class="big">{html.escape(cls_name)}</span><br>'
             f'<span class="muted">test subject {int(DATA["subj"][g])} &middot; window {k} of {len(idx)} '
             f'&middot; this person was never seen during training</span>')
    return render(DATA["X"][g], int(DATA["y"][g]), title, seed_choice)


def on_class_change(cls_name):
    if DATA is None:
        return gr.update(), gr.update()
    n = len(class_indices(cls_name, ALL_SUBJECTS))
    return (gr.update(choices=subject_choices(cls_name), value=ALL_SUBJECTS),
            gr.update(label=f"Window number (1 to {n})", maximum=max(n, 2), value=1))


def on_subject_change(cls_name, subj):
    n = len(class_indices(cls_name, subj)) if DATA is not None else 1
    return gr.update(label=f"Window number (1 to {n})", maximum=max(n, 2), value=1)


def on_random(seed_choice):
    if DATA is None:
        return (gr.update(), gr.update(), gr.update(), *NO_DATA)
    g = int(np.random.randint(len(DATA["y"])))
    cls_name = CLASS_NAMES[int(DATA["y"][g])]
    idx = class_indices(cls_name, ALL_SUBJECTS)
    k = int(np.where(idx == g)[0][0]) + 1
    return (gr.update(value=cls_name),
            gr.update(choices=subject_choices(cls_name), value=ALL_SUBJECTS),
            gr.update(label=f"Window number (1 to {len(idx)})", maximum=max(len(idx), 2), value=k),
            *classify_sample(cls_name, ALL_SUBJECTS, k, seed_choice))


# ---------------------------------------------------------------------------
# Tab 2: upload a CSV window
# ---------------------------------------------------------------------------
def read_window_csv(path):
    df = pd.read_csv(path, header=None, dtype=str)
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")  # drops a header row, if any
    if df.shape[1] != 12:
        raise ValueError(f"expected 12 numeric columns (phone accel xyz, phone gyro xyz, watch accel xyz, "
                         f"watch gyro xyz), found {df.shape[1]}")
    if df.shape[0] < 200:
        raise ValueError(f"expected at least 200 rows (10 s at 20 Hz), found {df.shape[0]}")
    return df.to_numpy(dtype=np.float32)[:200], df.shape[0]


def classify_upload(file_path, raw_units, seed_choice):
    if DATA is None:
        return NO_DATA
    if not file_path:
        return (banner("Upload a CSV file first (or create an example below).", error=True), None, "", None)
    try:
        window, n_rows = read_window_csv(file_path)
    except Exception as e:
        return (banner(f"Could not read the file: {html.escape(str(e))}", error=True), None, "", None)
    if raw_units:
        window = (window - DATA["mean"]) / DATA["std"]  # same z-score as training (train mean/std)
    note = f" (first 200 of {n_rows} rows used)" if n_rows > 200 else ""
    title = (f'Uploaded window{note}<br><span class="muted">no true label, so no correct / wrong badges. '
             f'{"Values normalised with the training mean and std." if raw_units else "Values used as they are (already z-scored)."}</span>')
    return render(window, None, title, seed_choice)


def make_example_csv():
    if DATA is None:
        return None, "Test data not available."
    g = int(np.random.randint(len(DATA["y"])))
    raw = DATA["X"][g] * DATA["std"] + DATA["mean"]
    out = Path(tempfile.mkdtemp()) / "example_window.csv"
    pd.DataFrame(raw, columns=CHANNELS).to_csv(out, index=False)
    return str(out), (f"Example created from a test window whose true activity is "
                      f"**{CLASS_NAMES[int(DATA['y'][g])]}** (raw units, tick the box below). "
                      f"Upload it above and compare.")


# ---------------------------------------------------------------------------
# Tab 3: results of the whole study (read from experiments/*/)
# ---------------------------------------------------------------------------
def load_study_rows():
    rows = []
    for key, name, role in MODELS:
        row = {"key": key, "name": name, "role": role}
        try:
            v = json.loads((EXP_DIR / key / "val_summary.json").read_text(encoding="utf-8"))
            row["val"], row["val_sd"] = v["val_macro_f1_mean"], v["val_macro_f1_std"]
        except Exception:
            pass
        try:
            t = json.loads((EXP_DIR / key / "test_results.json").read_text(encoding="utf-8"))
            row.update(test=t["test_macro_f1_mean"], test_sd=t["test_macro_f1_std"],
                       acc=t["test_accuracy_mean"], acc_sd=t["test_accuracy_std"])
            c = t.get("cost", {})
            row.update(params=c.get("params"), size=c.get("model_size_mb"),
                       infer=c.get("inference_ms_per_window"), train=c.get("train_time_sec_mean"))
        except Exception:
            pass
        rows.append(row)
    return rows


def fmt_time(sec):
    if sec is None:
        return "-"
    return f"{sec:.0f} s" if sec < 120 else f"{sec / 60:.0f} min"


def study_table_html(rows):
    have = [r["test"] for r in rows if "test" in r]
    best = max(have) if have else None
    body = []
    for r in rows:
        chip = f'<span class="chip {r["role"].lower()}">{r["role"]}</span>'
        if "test" not in r:
            body.append(f'<tr><td>{r["name"]} {chip}</td><td colspan="6">no results yet</td></tr>')
            continue

        def pm(m, s):
            return f"{r[m]:.3f} &plusmn; {r[s]:.3f}" if m in r else "-"
        cls = ' class="best"' if r["test"] == best else ""
        infer = f'{r["infer"]:.2f} ms' if r.get("infer") is not None else "-"
        params = fmt_int(r["params"]) if r.get("params") else "-"
        body.append(
            f'<tr{cls}><td>{r["name"]} {chip}</td><td>{pm("val", "val_sd")}</td>'
            f'<td>{pm("test", "test_sd")}</td><td>{pm("acc", "acc_sd")}</td>'
            f'<td>{params}</td><td>{infer}</td><td>{fmt_time(r.get("train"))}</td></tr>')
    return ('<table class="cmp"><thead><tr><th>Model</th><th>Val macro-F1</th><th>Test macro-F1</th>'
            '<th>Test accuracy</th><th>Params</th><th>Inference</th><th>Train / seed</th></tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def study_figure(rows):
    rows = [r for r in rows if "test" in r]
    if not rows:
        return None
    fig = Figure(figsize=(9.2, 4.0), dpi=100, facecolor="white")
    ax = fig.subplots()
    x = np.arange(len(rows))
    w = 0.38
    ax.bar(x - w / 2, [r.get("val", np.nan) for r in rows], w, yerr=[r.get("val_sd", 0) for r in rows],
           color="#93c5fd", label="validation", capsize=3)
    ax.bar(x + w / 2, [r["test"] for r in rows], w, yerr=[r["test_sd"] for r in rows],
           color="#4f46e5", label="test", capsize=3)
    for xi, r in zip(x, rows):
        ax.text(xi + w / 2, r["test"] + r["test_sd"] + 0.012, f'{r["test"]:.2f}', ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([r["name"] for r in rows], fontsize=9)
    ax.set_ylabel("macro-F1 (mean &plusmn; SD, 3 seeds)".replace("&plusmn;", "±"), fontsize=9)
    ax.set_ylim(0, 0.9)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Macro-F1 per model", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


ABOUT = """
### What you are looking at
Each model receives the **same window**: 200 time steps (10 s at 20 Hz) x 12 channels
(phone accelerometer, phone gyroscope, watch accelerometer, watch gyroscope; x, y, z each),
normalised with the training mean and standard deviation.

### How the models were trained
- Trained **from scratch** (random initial weights, no pretrained model) on WISDM only.
- Same recipe for all: Adam, batch size 64, up to 50 epochs, early stopping on validation macro-F1.
- **Subject-wise split**: 35 train / 8 validation / 8 test people. The windows in the first tab come from
  the 8 test people, whom no model saw during training.
- 3 seeds per model. "Average of 3 seeds" averages the three models' probabilities.

### How to read the results honestly
- Test macro-F1 is about 0.65 for the best models, so **roughly one window in three is wrong**.
  Some activities (for example *eating sandwich*) are hard for every model.
- CNN, Transformer and ResNet1D are **statistically close** on the test set. A single window cannot
  rank them; use the *Model comparison* tab for the numbers over all test windows.
- Latency is measured live on this computer (CPU, batch size 1, median of 3 passes), so it can differ
  a little from the table in the report.
- Research and classroom demo only. Do not use it for safety-critical decisions.
"""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
def build_ui():
    default_cls = CLASS_NAMES[0]
    if DATA is not None:  # first activity that actually exists in the test split
        present = sorted(set(DATA["y"].tolist()))
        default_cls = CLASS_NAMES[present[0]]
    with gr.Blocks(title="WISDM Activity Recognition: model comparison") as demo:
        gr.HTML(HERO)
        with gr.Tabs():
            # ---- tab 1
            with gr.Tab("Live comparison"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=290):
                        cls_dd = gr.Dropdown(CLASS_NAMES, value=default_cls, label="True activity (test set)")
                        subj_dd = gr.Dropdown(
                            subject_choices(default_cls) if DATA is not None else [ALL_SUBJECTS],
                            value=ALL_SUBJECTS, label="Test subject")
                        n0 = len(class_indices(default_cls, ALL_SUBJECTS)) if DATA is not None else 2
                        idx_sl = gr.Slider(1, max(n0, 2), value=1, step=1, label=f"Window number (1 to {n0})")
                        seed_rb = gr.Radio(SEED_CHOICES, value=SEED_CHOICES[0], label="Which trained model copy")
                        go_btn = gr.Button("Classify", variant="primary")
                        rnd_btn = gr.Button("Random test window")
                    with gr.Column(scale=3):
                        banner_out = gr.HTML()
                        signal_out = gr.Plot(label="Sensor window", show_label=False)
                gr.Markdown("### Predictions")
                cards_out = gr.HTML()
                heat_out = gr.Plot(label="Probability map", show_label=False)

                ins = [cls_dd, subj_dd, idx_sl, seed_rb]
                outs = [banner_out, signal_out, cards_out, heat_out]
                cls_dd.input(on_class_change, [cls_dd], [subj_dd, idx_sl]).then(classify_sample, ins, outs)
                subj_dd.input(on_subject_change, [cls_dd, subj_dd], [idx_sl]).then(classify_sample, ins, outs)
                idx_sl.release(classify_sample, ins, outs)
                seed_rb.input(classify_sample, ins, outs)
                go_btn.click(classify_sample, ins, outs)
                rnd_btn.click(on_random, [seed_rb], [cls_dd, subj_dd, idx_sl] + outs)
                demo.load(classify_sample, ins, outs)

            # ---- tab 2
            with gr.Tab("Upload your own window"):
                gr.Markdown(
                    "Upload a CSV with **12 numeric columns** (phone accel xyz, phone gyro xyz, watch accel xyz, "
                    "watch gyro xyz) and **at least 200 rows** (10 s at 20 Hz). A header row is allowed; "
                    "only the first 200 rows are used.")
                with gr.Row():
                    with gr.Column(scale=1, min_width=290):
                        file_in = gr.File(label="CSV file", file_types=[".csv", ".txt"], type="filepath")
                        raw_cb = gr.Checkbox(True, label="Values are in raw sensor units (normalise them for me)")
                        seed_rb2 = gr.Radio(SEED_CHOICES, value=SEED_CHOICES[0], label="Which trained model copy")
                        up_btn = gr.Button("Classify uploaded window", variant="primary")
                        gr.Markdown("**No file at hand?** Create an example from a random test window:")
                        ex_btn = gr.Button("Create example CSV")
                        ex_file = gr.File(label="Example (download, then upload it above)", interactive=False)
                        ex_note = gr.Markdown()
                    with gr.Column(scale=3):
                        banner2 = gr.HTML()
                        signal2 = gr.Plot(show_label=False)
                cards2 = gr.HTML()
                heat2 = gr.Plot(show_label=False)
                outs2 = [banner2, signal2, cards2, heat2]
                up_btn.click(classify_upload, [file_in, raw_cb, seed_rb2], outs2)
                ex_btn.click(make_example_csv, None, [ex_file, ex_note])

            # ---- tab 3
            with gr.Tab("Model comparison"):
                rows = load_study_rows()
                gr.Markdown("Results over **all test windows** of the 8 held-out subjects "
                            "(mean &plusmn; SD over 3 seeds, from `experiments/*/test_results.json`).")
                gr.HTML(study_table_html(rows))
                gr.Plot(value=study_figure(rows), show_label=False)

            # ---- tab 4
            with gr.Tab("About"):
                gr.Markdown(ABOUT)
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a temporary public link")
    args = ap.parse_args()

    if DATA_ERROR:
        print(f"[warning] {DATA_ERROR}")
    for key, name, _ in MODELS:
        have = [s for s in SEEDS if (CKPT_DIR / f"{key}_seed{s}.pt").exists()]
        print(f"  {name:12s} checkpoints: {have if have else 'none yet'}")
    torch.set_grad_enabled(False)
    build_ui().launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()