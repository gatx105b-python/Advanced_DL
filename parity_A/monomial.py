#!/usr/bin/env python
# plot_corr_deg1_phaseA.py
# -------------------------------------------------
# Phase-A 학생(checkpoint-A-*.pt) f(x)와 degree-1
# monomials(각 입력 변수) 상관계수 그림.
# -------------------------------------------------
import argparse, random
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")

# ---------- 데이터 생성 util ------------------------------------
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

def make_loader(dim=100, k=6, n_labels=2, batch=2048, seed=987):
    set_seed(seed)
    feats  = _boolean_tree_features(n_labels, dim, k)
    X, y   = _boolean_tree_samples(10_000, dim, n_labels, feats)
    return DataLoader(ParityDataset(X, y), batch_size=batch,
                      shuffle=False, num_workers=0, pin_memory=True)

# ---------- 모델, 평가 -----------------------------------------
def build_mlp(d_in, d_out, hidden):
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, d_out)
    ).to(DEVICE)

@torch.no_grad()
def corr_deg1(model, loader, temp=1e-4):
    """
    논문 식:
      | E_x ( p_T(x)[1] * x_j ) |
    - p_T(x)[1] : temperature=1e-4 softmax 확률, class=1
    - 절대값 후 반환
    """
    model.eval()
    xs, ps1 = [], []
    for X, _ in loader:
        X = X.to(DEVICE)
        prob = F.softmax(model(X) / temp, dim=-1)[:, 1]  # (batch,)
        xs.append(X.cpu().numpy())                       # ±1
        ps1.append(prob.cpu().numpy())
    X = np.concatenate(xs, axis=0)       # (N, dim)
    p = np.concatenate(ps1, axis=0)      # (N,)
    corr = np.abs((X * p[:, None]).mean(axis=0))  # (dim,)
    return corr                       # (dim,)

# ---------- ckpt 선택 함수 (이전 스크립트와 동일) ---------------
def acc_pairs(hist): return [(s,a) for s,a in hist]
def pick_ckpts(pairs):
    """
    3 개 체크포인트 선택 (논문 Figure 2):
      ① 낮은 정확도(초기) → min(acc) 의 step
      ② 중간(phase transition 한가운데) → acc 가 min 과 max 의 중간값에 가장 가까운 step
      ③ 최고 정확도 → max(acc) 의 step
    """
    pairs.sort(key=lambda x: x[0])           # step 순 정렬
    steps, accs = zip(*pairs)
    lo_step     = steps[int(np.argmin(accs))]
    hi_step     = steps[int(np.argmax(accs))]
    mid_target  = (min(accs) + max(accs)) / 2
    mid_step    = min(pairs, key=lambda x: abs(x[1] - mid_target))[0]
    return [lo_step, mid_step, hi_step]

# -------------------- main --------------------------------------
def main(args):
    out_dir   = Path(args.dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # 하이퍼파라미터 (학습과 동일)
    dim, k, n_labels = 100, 6, 2
    student_h        = 100

    # ① 평가용 데이터
    loader = make_loader(dim, k, n_labels)

    # ② phase-A ckpt 불러와 상관계수 계산
    A_ckpts = sorted(out_dir.glob("checkpoint-A-*.pt"),
                     key=lambda p: int(p.stem.split('-')[-1]))
    if not A_ckpts:
        raise RuntimeError("checkpoint-A-*.pt 파일이 없습니다.")

    corr_each, corr_rest_mean, steps = [[] for _ in range(6)], [], []
    for p in A_ckpts:
        step  = int(p.stem.split('-')[-1])
        model = build_mlp(dim, n_labels, student_h)
        model.load_state_dict(torch.load(p, map_location=DEVICE))
        c     = corr_deg1(model, loader)           # (dim,)
        for i in range(6):                         # x1~x6 개별 저장
            corr_each[i].append(c[i])
        corr_rest_mean.append( c[6:].mean() )           # 나머지
        steps.append(step)
    corr_sup = [ np.mean([corr_each[i][idx] for i in range(6)])
                 for idx in range(len(steps)) ]
    # ③ 선택된 4 개 step (세로 점선용)
    sel_steps = pick_ckpts(list(zip(steps, corr_sup)))

    # ④ 그림
    plt.figure(figsize=(7,4))
    colors = plt.cm.tab10.colors          # x1~x6 색상
    for i in range(6):
        plt.plot(steps, corr_each[i], label=f"x{i+1}", color=colors[i])
    plt.plot(steps, corr_rest_mean, label="rest (7~100)", color="black",
             linestyle="--", linewidth=1.5)
    for s in sel_steps:
        plt.axvline(s, ls="--", lw=1, color="gray")
    plt.xlabel("training step"); plt.ylabel("|corr(f,x)|")
    plt.title("Phase-A   correlation to degree-1 monomials")
    plt.legend()
    plt.tight_layout()
    save_p = plots_dir / "phaseA_deg1_corr.png"
    plt.savefig(save_p); plt.close()
    print(f"▶ 그림 저장: {save_p.resolve()}")

# ---------------- CLI -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="outputs_Tsplit",
                        help="checkpoint 디렉터리 (checkpoint-A-*.pt 가 있어야 함)")
    args = parser.parse_args()
    main(args)

