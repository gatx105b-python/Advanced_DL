#!/usr/bin/env python
# plot_corr_deg1_phaseA_v2.py
# -------------------------------------------------------------
# Phase-A 학생 f_S 와 degree-1 monomials 상관계수 시각화
# (support 1~6 각각 & rest 평균) – 논문 Figure 2 방식 준수
# -------------------------------------------------------------
import argparse, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")

# ------------- 데이터 ------------------------------------------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
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
            f = feat_sets[lbl // 2 - 1]
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
    def __getitem__(self, i): return self.X[i], self.y[i]

def make_eval_loader(dim=100, k=6, n_labels=2, batch=2048, seed=42):
    set_seed(seed)
    feats = _boolean_tree_features(n_labels, dim, k)
    X, y = _boolean_tree_samples(10_000, dim, n_labels, feats)
    return DataLoader(ParityDataset(X, y), batch_size=batch,
                      shuffle=False, num_workers=0, pin_memory=True)

# ------------- 모델 --------------------------------------------------
def build_mlp(d_in, d_out, hidden):
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, d_out)
    ).to(DEVICE)

@torch.no_grad()
def corr_deg1_abs(model, loader):
    """|E_x [ p1(x) x_j ]|  (dim,)"""
    model.eval()
    tot_corr = None; n_tot = 0
    for X,_ in loader:
        X = X.to(DEVICE)
        p = torch.softmax(model(X), dim=-1)[:,1]      # (B,)
        # p * x_j  →  (B,dim)
        prod = (p[:,None] * X).cpu().numpy()
        if tot_corr is None: tot_corr = prod.sum(axis=0)
        else:                tot_corr += prod.sum(axis=0)
        n_tot += prod.shape[0]
    corr = np.abs(tot_corr / n_tot)                   # (dim,)
    return corr

# ------------- ckpt 선택 (기존 pick_ckpts() 그대로) ------------------
def acc_pairs(pairs): return [(s,a) for s,a in pairs]
def pick_ckpts(pairs, surge_th=0.03):
    pairs.sort(key=lambda x: x[0])
    steps, accs = zip(*pairs)
    diffs = np.diff(accs); k = int(np.argmax(diffs))
    pre,during = steps[max(k-1,0)], steps[k]
    post = during
    for j in range(k+1,len(diffs)):
        if diffs[j] < surge_th: post = steps[j+1]; break
    best = steps[int(np.argmax(accs))]
    return sorted({pre,during,post,best})

# ------------- 메인 --------------------------------------------------
def main(args):
    out_dir   = Path(args.dir)
    plots_dir = out_dir / "plots"; plots_dir.mkdir(exist_ok=True)

    # 설정 (학습 때 값과 일치해야 함)
    dim, k, n_labels = 100, 6, 2
    student_h        = 100
    loader           = make_eval_loader(dim, k, n_labels)

    # Phase-A ckpt 목록
    A_ckpts = sorted(out_dir.glob("checkpoint-A-*.pt"),
                     key=lambda p: int(p.stem.split('-')[-1]))
    if not A_ckpts:
        raise RuntimeError("checkpoint-A-*.pt 파일이 없습니다")

    support_corr = {i:[] for i in range(k)}
    rest_corr    = []
    steps        = []

    # 평가 함수 (훈련 스크립트와 동일)
    def evaluate(model):
        model.eval(); tot = n = 0
        with torch.no_grad():
            for X_, y_ in loader:
                X_, y_ = X_.to(DEVICE), y_.to(DEVICE)
                tot += (model(X_).argmax(-1) == y_).sum().item(); n += len(X_)
        return tot / n

    # 정확도 로그 → sel_steps 계산
    acc_log = []
    for p in A_ckpts:
        step  = int(p.stem.split('-')[-1])
        mdl   = build_mlp(dim, n_labels, student_h)
        mdl.load_state_dict(torch.load(p, map_location=DEVICE))
        # -------- 상관계수 ----------
        corr  = corr_deg1_abs(mdl, loader)            # (dim,)
        for i in range(k):
            support_corr[i].append(corr[i])
        rest_corr.append(corr[k:].mean())
        steps.append(step)
        # accuracy(선택용)
        acc = evaluate(mdl)
        acc_log.append((step, acc))

    sel_steps = pick_ckpts(acc_log)

    # ------------------- 그림 -------------------
    plt.figure(figsize=(7,4))
    colors = plt.cm.tab10.colors
    for i in range(k):
        plt.plot(steps, support_corr[i], label=f"x{i+1}", color=colors[i%10])
    plt.plot(steps, rest_corr, label="rest (7-100)", color="grey", lw=2)

    for s in sel_steps:
        plt.axvline(s, ls="--", color="black", lw=0.8)

    plt.xlabel("training step")
    plt.ylabel(r"$|\,\mathbb{E}[p_1(x)\,x_j]\,|$")
    plt.title("Phase-A  degree-1 monomial correlations")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    save_p = plots_dir / "phaseA_deg1_corr_v2.png"
    plt.savefig(save_p, dpi=300)
    plt.close()
    print(f"▶ 저장 완료: {save_p.resolve()}")

# ------------------- CLI ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="outputs_Tsplit",
                        help="checkpoint 디렉터리 (checkpoint-A-*.pt 포함)")
    main(parser.parse_args())

