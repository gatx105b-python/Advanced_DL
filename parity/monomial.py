#!/usr/bin/env python
# plot_corr_deg1_teacher.py
# -------------------------------------------------------------
# Teacher(checkpoint-T-*.pt)  |E[p1(x) x_j]|  상관계수 그림
#   - support 변수 x1~x6 각각 곡선
#   - non-support 평균 곡선
#   - 선택된 4개 teacher ckpt step(논문 방식) 세로 점선
# -------------------------------------------------------------
import argparse, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")

# ---------- 데이터 ----------------------------------------------------------------
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

def make_eval_loader(dim=100, k=6, n_labels=2, batch=2048, seed=111):
    set_seed(seed)
    feats = _boolean_tree_features(n_labels, dim, k)
    X, y = _boolean_tree_samples(10_000, dim, n_labels, feats)
    return DataLoader(ParityDataset(X, y), batch_size=batch,
                      shuffle=False, num_workers=0, pin_memory=True)

# ---------- 모델 ------------------------------------------------------------------
def build_mlp(d_in, d_out, hidden):
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, d_out)
    ).to(DEVICE)

@torch.no_grad()
def corr_deg1_abs(model, loader):
    """
    | E_x [ p1(x) * x_j ] |  (dim,)  — 논문 정의 그대로, 절대값 사용
    p1(x) = softmax(f(x))_1
    """
    model.eval()
    tot, n = None, 0
    for X,_ in loader:
        X = X.to(DEVICE)
        p1 = torch.softmax(model(X), dim=-1)[:,1]           # (B,)
        prod = (p1[:,None] * X).cpu().numpy()               # (B,dim)
        tot = prod.sum(axis=0) if tot is None else tot + prod.sum(axis=0)
        n += prod.shape[0]
    return np.abs(tot / n)                                  # (dim,)

# ---------- ckpt 선택(논문 방식) -------------------------------------
def pick_ckpts(pairs, surge_th=0.03):
    """
    (1) 급증 직전, (2) 급증 중(최대 기울기), (3) plateau 첫 지점, (4) 최고 acc
    """
    pairs.sort(key=lambda x: x[0])
    steps, accs = zip(*pairs)
    diffs = np.diff(accs); k = int(np.argmax(diffs))
    pre, during = steps[max(k-1,0)], steps[k]
    post = during
    for j in range(k+1, len(diffs)):
        if diffs[j] < surge_th: post = steps[j+1]; break
    best = steps[int(np.argmax(accs))]
    return sorted({pre, during, post, best})

# ---------- 메인 -----------------------------------------------------
def main(args):
    out_dir   = Path(args.dir)
    plots_dir = out_dir / "plots"; plots_dir.mkdir(exist_ok=True)

    # 하이퍼파라미터 (학습 때 설정과 동일)
    dim, k, n_labels = 100, 6, 2
    teacher_h        = 50000
    eval_loader      = make_eval_loader(dim, k, n_labels)

    # Teacher ckpt 목록
    T_ckpts = sorted(out_dir.glob("checkpoint-T-*.pt"),
                     key=lambda p: int(p.stem.split('-')[-1]))
    if not T_ckpts:
        raise RuntimeError("checkpoint-T-*.pt 파일이 없습니다.")

    # 상관계수 및 정확도 로그
    support_corr = {i: [] for i in range(k)}
    rest_corr, steps, acc_log = [], [], []

    for p in T_ckpts:
        step = int(p.stem.split('-')[-1])
        model = build_mlp(dim, n_labels, teacher_h)
        model.load_state_dict(torch.load(p, map_location=DEVICE))

        # correlation
        corr = corr_deg1_abs(model, eval_loader)
        for i in range(k):
            support_corr[i].append(corr[i])
        rest_corr.append(corr[k:].mean())
        steps.append(step)

        # accuracy (선택용)
        with torch.no_grad():
            X,y = next(iter(eval_loader))
            acc = (model(X.to(DEVICE)).argmax(-1) == y.to(DEVICE)).float().mean().item()
        acc_log.append((step, acc))

    sel_steps = pick_ckpts(acc_log)

    # ---------------- 그림 ----------------
    plt.figure(figsize=(7,4))
    colors = plt.cm.Set1.colors           # 9 가지 컬러맵
    for i in range(k):
        plt.plot(steps, support_corr[i], label=f"x{i+1}", color=colors[i%9])
    plt.plot(steps, rest_corr, label="rest (7-100)", color="grey", lw=2)

    for s in sel_steps:
        plt.axvline(s, ls="--", color="black", lw=0.8)

    plt.xlabel("training step")
    plt.ylabel(r"$|\,\mathbb{E}[p_1(x)\,x_j]\,|$")
    plt.title("Teacher   degree-1 monomial correlations")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    save_p = plots_dir / "phaseT_deg1_corr.png"
    plt.savefig(save_p, dpi=300)
    plt.close()
    print(f"▶ 그림 저장 위치: {save_p.resolve()}")

# ------------- CLI ---------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="outputs_Tsplit",
                        help="Teacher 체크포인트가 들어있는 폴더")
    main(parser.parse_args())

