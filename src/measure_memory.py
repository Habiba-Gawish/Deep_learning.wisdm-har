"""Peak RAM of every model (CPU), for Table 7.

    pip install psutil
    python src\\measure_memory.py

Each model runs in its OWN fresh process (so numbers do not mix). We measure the peak
resident memory (a) during inference on 1 window and (b) during one training step
(batch 64, forward + backward + Adam). We subtract the memory of an empty process that
only imported torch, so the table shows what the MODEL adds. Random input is used
(memory does not depend on the values). Results -> experiments/memory.json
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import psutil

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
MODELS = ["mlp", "cnn1d", "gru", "transformer", "resnet1d"]


def peak_mb():
    mi = psutil.Process().memory_info()
    peak = getattr(mi, "peak_wset", None)          # Windows: true peak
    if peak is None:                                # Linux/macOS
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux gives KB
    return peak / 1e6


def child(name, mode):
    import torch
    sys.path.insert(0, str(SRC))
    if mode != "base":
        import yaml
        from train import build_model
        cfg = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))
        torch.manual_seed(0)
        model = build_model(cfg)
        if mode == "infer":
            model.eval()
            with torch.no_grad():
                model(torch.randn(1, 200, 12))
        else:  # train step
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            x, y = torch.randn(64, 200, 12), torch.randint(0, 18, (64,))
            loss = torch.nn.functional.cross_entropy(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
    print(json.dumps({"peak_mb": peak_mb()}))


def run(name, mode):
    out = subprocess.run([sys.executable, __file__, "--child", name, mode],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])["peak_mb"]


def main():
    base = run("none", "base")
    res = {"baseline_process_mb": round(base, 1), "models": {}}
    print(f"empty process (torch imported): {base:.0f} MB\n")
    print(f"{'model':12s} {'inference (MB)':>15s} {'train step (MB)':>16s}   (extra over empty process)")
    for m in MODELS:
        inf, trn = run(m, "infer") - base, run(m, "train") - base
        res["models"][m] = {"inference_extra_mb": round(inf, 1), "train_step_extra_mb": round(trn, 1)}
        print(f"{m:12s} {inf:15.1f} {trn:16.1f}")
    (ROOT / "experiments").mkdir(exist_ok=True)
    (ROOT / "experiments" / "memory.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\nsaved experiments\\memory.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=2, metavar=("NAME", "MODE"))
    a = ap.parse_args()
    child(*a.child) if a.child else main()