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

    print(f"[INFO] Processing group: {files[0].stem.rsplit('-', 1)[0]}")
    for p in files:
        step = int(p.stem.split('-')[-1])
        print(f"  - Loading model at step {step} from {p.name}")
        model = build_mlp(dim, lab, hidden)
        model.load_state_dict(torch.load(p, map_location=DEVICE))
        model.eval()

        for d in range(2, 7):
            corr = corr_degree(model, loader, d, k)
            deg_corr[d].append(corr)
            print(f"    Degree-{d} correlation: {corr:.6f}")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump({
            'step': steps,
            **{f'deg{d}': [float(c) for c in deg_corr[d]] for d in range(2, 7)}
        }, f, indent=2)
    print(f"  -> Saved correlation JSON: {out_json}")

    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    for d in [2, 3]:
        plt.plot(steps, deg_corr[d], marker='o', label=f'deg-{d}')
    for s in [20000, 30000, 50000]:
        plt.axvline(s, ls='--', c='red', lw=1)
    plt.xlabel('step'); plt.ylabel('mean |corr|'); plt.legend()
    plt.tight_layout()
    plot_path1 = plot_dir / f'{files[0].stem.rsplit("-", 1)[0]}_deg23.png'
    plt.savefig(plot_path1, dpi=250)
    plt.close()
    print(f"  -> Saved plot: {plot_path1}")

    plt.figure(figsize=(6, 4))
    for d in [4, 5, 6]:
        plt.plot(steps, deg_corr[d], marker='o', label=f'deg-{d}')
    for s in [20000, 30000, 50000]:
        plt.axvline(s, ls='--', c='red', lw=1)
    plt.xlabel('step'); plt.ylabel('mean |corr|'); plt.legend()
    plt.tight_layout()
    plot_path2 = plot_dir / f'{files[0].stem.rsplit("-", 1)[0]}_deg46.png'
    plt.savefig(plot_path2, dpi=250)
    plt.close()
    print(f"  -> Saved plot: {plot_path2}\n")


def main(args):
    root = Path(args.dir)
    save_root = Path('/data/dy0718/repos/parity_comp/correlations')
    print(f"[INFO] Root directory: {root.resolve()}")
    subdirs = [d for d in root.iterdir() if d.is_dir() and d.name != 'phase0']
    print(f"[INFO] Found {len(subdirs)} phase directories (excluding phase0)")

    dim, k, lab = 100, 6, 2
    loader = make_loader(0, dim, k, lab, 0, seed=42, eval_only=True)
    try:
        test_batch = next(iter(loader))
        print(f"[INFO] Example batch shape: X={test_batch[0].shape}, y={test_batch[1].shape}")
    except Exception as e:
        print(f"[ERROR] Failed to fetch batch from loader: {e}")
        return

    for phase_dir in subdirs:
        print(f"\n[PHASE] Processing folder: {phase_dir.name}")
        all_files = list(phase_dir.glob('*.pt'))
        if not all_files:
            print("  [SKIP] No checkpoint files found.")
            continue

        groups = {}
        for p in all_files:
            base = p.stem.split('-')[1]
            if base.startswith('PD'):
                print(f"  [SKIP] {p.name} (Phase-D checkpoint)")
                continue
            prefix = '-'.join(p.stem.split('-')[:-1])
            groups.setdefault(prefix, []).append(p)

        for prefix, files in groups.items():
            out_json = save_root / phase_dir.name / 'json' / f'{Path(prefix).name}.json'
            plot_dir = save_root / phase_dir.name / 'plotting'
            try:
                process_group(files, loader, dim, k, lab, out_json, plot_dir)
            except Exception as e:
                print(f"[ERROR] Failed to process group '{prefix}': {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='/local_datasets/parity', help='Top-level parity experiment directory')
    args = parser.parse_args()
    set_seed(42)
    main(args)

