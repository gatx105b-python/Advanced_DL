import json
import shutil
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from models import DEVICE


def pick_ckpts(pairs, surge_th: float = 0.03):
    """Return the fixed checkpoint steps [20000, 30000, 50000]."""
    return [20000, 30000, 50000]


def pick_ckpts_phaseA(pairs, th: float = 0.03):
    return pick_ckpts(pairs, th)


def rise_region(pairs, Δ: float = 0.005):
    pairs.sort(key=lambda x: x[0])
    steps, acc = zip(*pairs)
    dif = np.diff(acc)
    rise = np.where(dif > Δ)[0]
    if len(rise) == 0:
        return None, None, None
    s_idx, e_idx = rise[0], rise[-1] + 1
    mid = int(round((steps[s_idx] + steps[e_idx]) / 2))
    return steps[s_idx], steps[e_idx], mid


def best_mid_worst(pairs):
    best = max(pairs, key=lambda x: x[1])
    worst = min(pairs, key=lambda x: x[1])
    mid_target = (best[1] + worst[1]) / 2
    mid = min(pairs, key=lambda x: abs(x[1] - mid_target))
    return best, mid, worst


def plot_curve(pairs, *, sel_steps=None, add_lines: List[Tuple[float, str]] = [],
               title: str = "", fname: str = "curve.png", out_dir: Path | str = "."):
    pairs.sort(key=lambda x: x[0])
    x, y = zip(*pairs)
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, label=title)
    if sel_steps:
        for s in sel_steps:
            plt.axvline(s, ls="--", lw=1, c="red")
    for val, lab in add_lines:
        plt.hlines(val, x[0], x[-1], linestyles=":", label=lab)
    plt.xlabel("step"); plt.ylabel("accuracy"); plt.legend()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_dir / fname, dpi=250); plt.close()


def corr_curve(ckpt_pattern: str, model_fn, dim, k, lab, out_path, loader, out_json=None):
    xs, ys = [], []
    for p in sorted(Path(ckpt_pattern).parent.glob(Path(ckpt_pattern).name),
                    key=lambda q: int(q.stem.split("-")[-1])):
        step = int(p.stem.split("-")[-1])
        mdl = model_fn(dim, lab)
        mdl.load_state_dict(torch.load(p, map_location=DEVICE))
        tot = None; n = 0
        mdl.eval()
        with torch.no_grad():
            for X, _ in loader:
                X = X.to(DEVICE)
                p1 = torch.softmax(mdl(X), -1)[:, 1]
                prod = (p1[:, None] * X).cpu().numpy()
                tot = prod.sum(0) if tot is None else tot + prod.sum(0)
                n += len(X)
        corr_mean = np.abs(tot / n)[:k].mean()
        xs.append(step); ys.append(corr_mean)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker='o')
    plt.xlabel("step"); plt.ylabel("mean |corr| (deg-1)")
    plt.title(Path(ckpt_pattern).parent.name)
    plt.tight_layout(); plt.savefig(out_path, dpi=250); plt.close()
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(
            {"step": [int(s) for s in xs], "corr": [float(c) for c in ys]},
            open(out_json, "w"),
            indent=2,
        )


def move_keep(src_root: Path, keep_steps: List[int], tag: str, dst_root: Path):
    dst_root.mkdir(parents=True, exist_ok=True)
    for p in src_root.glob(f"ckpt-{tag}-*.pt"):
        step = int(p.stem.split('-')[-1])
        if step in keep_steps:
            shutil.copy2(p, dst_root / p.name)
        p.unlink()


def nearest_logged_step(target_step, logged_pairs):
    return min(logged_pairs, key=lambda x: abs(x[0] - target_step))[0]
