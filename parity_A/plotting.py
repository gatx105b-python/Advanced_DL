#!/usr/bin/env python
# parity_progress_plots.py
# ---------------------------------------------------------------
# • Phase-A student(acc) 곡선 + 선택된 4 ckpt 표시
# • 4 개 ckpt마다 Phase-B(큰 S)·Phase-C(작은 S) 곡선 그리기
#   - teacher 최고 acc, phase-A 선택 acc → 점선(hline)
# ---------------------------------------------------------------
import argparse, random
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")

# ------------------------- utils -------------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def _boolean_tree_features(n_labels, dim, k):
    assert (n_labels - 1) * k <= dim
    return [range(i, i + k) for i in range(0, (n_labels - 1) * k, k)]

def _boolean_tree_samples(n, dim, n_labels, feat_sets):
    X = 2 * np.random.randint(0, 2, size=(n, dim)) - 1
    scores = np.zeros((n, n_labels))
    def _score(x, lbl):
        s = 0
        while lbl > 1:
            f  = feat_sets[lbl // 2 - 1]
            s += (1 - 2 * (lbl % 2)) * np.prod(x[:, f], axis=-1)
            lbl //= 2
        return s
    for lbl in range(n_labels):
        scores[:, lbl] = _score(X, lbl + n_labels)
    y = scores.argmax(-1).astype(np.int64)
    return X.astype(np.float32), y

class ParityDataset(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i], self.y[i]

def make_eval_loader(dim=100, k=6, n_labels=2, batch=2048, seed=43):
    set_seed(seed)
    feats      = _boolean_tree_features(n_labels, dim, k)
    _,  y   = _boolean_tree_samples(10_000, dim, n_labels, feats)  # throw
    X,  y   = _boolean_tree_samples(10_000, dim, n_labels, feats)
    return DataLoader(ParityDataset(X, y), batch_size=batch, shuffle=False,
                      num_workers=0, pin_memory=True)

def build_mlp(d_in, d_out, hidden):
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, d_out)
    ).to(DEVICE)

def evaluate(model, loader):
    model.eval(); tot = n = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            tot += (model(X).argmax(-1) == y).sum().item(); n += len(X)
    return tot / n

def acc_pairs(hist): return [(s,a) for s,a in hist]
def pick_ckpts(pairs, surge_th=0.03):
    pairs.sort(key=lambda x: x[0])
    steps, accs      = zip(*pairs)
    diffs            = np.diff(accs)
    k                = int(np.argmax(diffs))
    pre, during      = steps[max(k-1,0)], steps[k]
    post             = during
    for j in range(k+1,len(diffs)):
        if diffs[j] < surge_th: post = steps[j+1]; break
    best             = steps[int(np.argmax(accs))]
    return sorted({pre,during,post,best})

# --------------------- main plotting ---------------------------
def main(args):
    out_dir     = Path(args.dir)
    plots_dir   = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # 하이퍼파라미터 (학습 때와 동일해야 함)
    data_dim, n_labels, k  = 100, 2, 6
    teacher_h, student_h   = 50000, 100

    ev_loader   = make_eval_loader(data_dim, k, n_labels, seed=43)

    # ---------- (0) teacher 최고 accuracy 구하기 ----------------
    t_ckpts  = sorted(out_dir.glob("checkpoint-T-*.pt"),
                      key=lambda p: int(p.stem.split('-')[-1]))
    if not t_ckpts:
        raise RuntimeError("checkpoint-T-*.pt not found")

    best_acc, best_step = -1.0, None
    for p in t_ckpts:
        model = build_mlp(data_dim, n_labels, teacher_h)
        model.load_state_dict(torch.load(p, map_location=DEVICE))
        acc   = evaluate(model, ev_loader)
        if acc > best_acc: best_acc, best_step = acc, int(p.stem.split('-')[-1])

    # ---------- (1) Phase-A 학생 곡선 ----------------------------
    A_pairs = []
    a_ckpts = sorted(out_dir.glob("checkpoint-A-*.pt"),
                     key=lambda p: int(p.stem.split('-')[-1]))
    for p in a_ckpts:
        step  = int(p.stem.split('-')[-1])
        model = build_mlp(data_dim, n_labels, student_h)
        model.load_state_dict(torch.load(p, map_location=DEVICE))
        A_pairs.append( (step, evaluate(model, ev_loader)) )

    sel_A_steps = pick_ckpts(A_pairs)
    sel_A_accs  = {s:a for s,a in A_pairs if s in sel_A_steps}

    # --- plot Phase-A
    xs, ys = zip(*sorted(A_pairs))
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, label="Phase-A (small-S)")
    plt.hlines(best_acc, xs[0], xs[-1], linestyles=":", label="teacher best")
    for s in sel_A_steps:
        plt.scatter(s, sel_A_accs[s], c="r", marker="D")
    plt.xlabel("step"); plt.ylabel("accuracy"); plt.legend()
    plt.tight_layout(); plt.savefig(plots_dir / "phaseA_curve.png"); plt.close()

    # ---------- (2) Phase-B / C 4 개 세트 ------------------------
    for idx, sel_step in enumerate(sel_A_steps, 1):
        # Phase-B 큰 S
        B_pairs = []
        for p in sorted(out_dir.glob(f"checkpoint-B{idx}-*.pt"),
                        key=lambda p: int(p.stem.split('-')[-1])):
            st  = int(p.stem.split('-')[-1])
            m   = build_mlp(data_dim, n_labels, teacher_h)
            m.load_state_dict(torch.load(p, map_location=DEVICE))
            B_pairs.append( (st, evaluate(m, ev_loader)) )

        # Phase-C 작은 S
        C_pairs = []
        for p in sorted(out_dir.glob(f"checkpoint-C{idx}-*.pt"),
                        key=lambda p: int(p.stem.split('-')[-1])):
            st  = int(p.stem.split('-')[-1])
            m   = build_mlp(data_dim, n_labels, student_h)
            m.load_state_dict(torch.load(p, map_location=DEVICE))
            C_pairs.append( (st, evaluate(m, ev_loader)) )

        # 그림
        plt.figure(figsize=(6,4))
        if B_pairs:
            xb, yb = zip(*sorted(B_pairs))
            plt.plot(xb, yb, label=f"Phase-B{idx} (big-S)")
        if C_pairs:
            xc, yc = zip(*sorted(C_pairs))
            plt.plot(xc, yc, label=f"Phase-C{idx} (small-S)")
        plt.hlines(best_acc, 0, max(xb[-1], xc[-1]), linestyles=":", label="teacher best")
        plt.hlines(sel_A_accs[sel_step], 0, max(xb[-1], xc[-1]),
                   linestyles=":", colors="g", label="Phase-A ckpt")
        plt.xlabel("step"); plt.ylabel("accuracy"); plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / f"phaseB{idx}_C{idx}_curve.png")
        plt.close()

    print(f"▶ 그림 저장 위치: {plots_dir.resolve()}")

# ----------------------- CLI ------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="outputs_Tsplit",
                        help="체크포인트 디렉터리")
    args = parser.parse_args()
    main(args)

