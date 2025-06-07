#!/usr/bin/env python
import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import make_loader, set_seed
from models import DEVICE, build_mlp


@torch.no_grad()
def corr_degree(model, loader, degree: int, k: int) -> float:
    combos = list(combinations(range(k), degree))
    tot = np.zeros(len(combos))
    n = 0
    model.eval()
    for X, _ in loader:
        X = X.to(DEVICE)
        p1 = torch.softmax(model(X), dim=-1)[:, 1].cpu().numpy()
        Xk = X[:, :k].cpu().numpy()
        for idx, cmb in enumerate(combos):
            prod = np.prod(Xk[:, cmb], axis=1)
            tot[idx] += np.sum(p1 * prod)
        n += len(X)
    return np.abs(tot / n).mean()


def infer_hidden(name: str) -> int:
    base = Path(name).stem.split('-')[1]
    if base.startswith('PA') or base.startswith('PC'):
        return 100
    return 50000


def process_group(files, loader, dim, k, lab, out_json, plot_dir):
    files = sorted(files, key=lambda p: int(p.stem.split('-')[-1]))
    steps = [int(p.stem.split('-')[-1]) for p in files]
    deg_corr = {d: [] for d in range(2, 7)}
    hidden = infer_hidden(files[0].name)
    for p in files:
        model = build_mlp(dim, lab, hidden)
        model.load_state_dict(torch.load(p, map_location=DEVICE))
        for d in range(2, 7):
            deg_corr[d].append(corr_degree(model, loader, d, k))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        'step': steps,
        **{f'deg{d}': [float(c) for c in deg_corr[d]] for d in range(2,7)}
    }, open(out_json, 'w'), indent=2)
    plot_dir.mkdir(parents=True, exist_ok=True)
    # plot degrees 2 & 3
    plt.figure(figsize=(6,4))
    for d in [2,3]:
        plt.plot(steps, deg_corr[d], marker='o', label=f'deg-{d}')
    for s in [20000, 30000, 50000]:
        plt.axvline(s, ls='--', c='red', lw=1)
    plt.xlabel('step'); plt.ylabel('mean |corr|'); plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f'{files[0].stem.rsplit("-",1)[0]}_deg23.png', dpi=250)
    plt.close()
    # plot degrees 4-6
    plt.figure(figsize=(6,4))
    for d in [4,5,6]:
        plt.plot(steps, deg_corr[d], marker='o', label=f'deg-{d}')
    for s in [20000, 30000, 50000]:
        plt.axvline(s, ls='--', c='red', lw=1)
    plt.xlabel('step'); plt.ylabel('mean |corr|'); plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f'{files[0].stem.rsplit("-",1)[0]}_deg46.png', dpi=250)
    plt.close()


def main(args):
    root = Path(args.dir)
    all_files = list(root.glob('*.pt'))
    groups = {}
    for p in all_files:
        base = p.stem.split('-')[1]
        if base.startswith('PD'):
            # skip phase-D checkpoints
            continue
        prefix = '-'.join(p.stem.split('-')[:-1])
        groups.setdefault(prefix, []).append(p)
    dim, k, lab = 100, 6, 2
    loader = make_loader(0, dim, k, lab, 0, seed=42, eval_only=True)
    for prefix, files in groups.items():
        out_json = root / 'correlation' / 'high_correlation' / 'json' / f'{Path(prefix).name}.json'
        plot_dir = root / 'correlation' / 'high_correlation' / 'plotting'
        process_group(files, loader, dim, k, lab, out_json, plot_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='outputs_tmp', help='directory with checkpoint .pt files')
    args = parser.parse_args()
    set_seed(42)
    main(args)
