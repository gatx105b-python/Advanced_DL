# parity_role_reversal_kd_Tsplit.py
# -*- coding: utf-8 -*-
"""
Role-Reversal KD on Sparse-Parity
─────────────────────────────────
• Phase-0 : large Teacher, **T = 1e7 step**
• Phase-A : BigT → SmallS, **S = 5e5 step**
• Phase-B : SmallT → BigS, **S = 5e5 step**
• Phase-C : BigT → Fresh SmallS, **S = 5e5 step**

Teacher ckpt 3개(초·중·최고 acc) 기반으로 A/B/C 반복.
일반 distillation (forward-KL), hard teacher (τ = 1e-4).
"""

import argparse, random
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_float32_matmul_precision("high")

# ------------------------------------------------------------------
# utils
# ------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_teacher(log, ckpts, save_path):
    import matplotlib.pyplot as plt
    xs, ys = zip(*acc_pairs(log))
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, label="Teacher acc")
    for s in ckpts:
        plt.axvline(s, ls="--", lw=1, c="r")
    plt.xlabel("step"); plt.ylabel("accuracy"); plt.legend()
    plt.tight_layout(); plt.savefig(save_path); plt.close()

# ------------------------------------------------------------------
# data
# ------------------------------------------------------------------

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


def make_loader(n_samples, dim, k, n_labels, batch, seed):
    set_seed(seed)
    feats = _boolean_tree_features(n_labels, dim, k)
    X, y = _boolean_tree_samples(n_samples + 10_000, dim, n_labels, feats)
    X_tr, y_tr, X_ev, y_ev = X[:-10_000], y[:-10_000], X[-10_000:], y[-10_000:]
    tr = DataLoader(ParityDataset(X_tr, y_tr), batch_size=batch, shuffle=True, num_workers=0, pin_memory=True)
    ev = DataLoader(ParityDataset(X_ev, y_ev), batch_size=2048, shuffle=False, num_workers=0, pin_memory=True)
    return tr, ev

# ------------------------------------------------------------------
# model & loss
# ------------------------------------------------------------------

def build_mlp(d_in, d_out, hidden):
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, d_out)
    ).to(DEVICE)


def kd_loss(s_logits, t_logits, y, alpha, temp):
    ce = F.cross_entropy(s_logits, y)
    q_t = F.softmax(t_logits / temp, dim=-1).detach()
    log_p_s = F.log_softmax(s_logits, dim=-1)
    kl = torch.mean(torch.sum(q_t * -log_p_s, dim=-1))
    return alpha * ce + (1 - alpha) * kl

# ------------------------------------------------------------------
# training loop
# ------------------------------------------------------------------

def evaluate(model, loader):
    model.eval(); tot = n = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            tot += (model(X).argmax(-1) == y).sum().item(); n += len(X)
    return tot / n


def train(model, tr_loader, ev_loader, *, kd, teacher, lr, alpha, temp, tag, out_dir, eval_steps, save_steps, max_steps):
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    log, gstep = [], 0; pbar = tqdm(total=max_steps, desc=tag)
    if teacher is not None: teacher.eval()
    while gstep < max_steps:
        for X, y in tr_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            if kd:
                s_logits = model(X); t_logits = teacher(X).detach()
                loss = kd_loss(s_logits, t_logits, y, alpha, temp)
            else:
                loss = F.cross_entropy(model(X), y)
            loss.backward(); optim.step(); optim.zero_grad(); gstep += 1; pbar.update(1)
            if gstep % eval_steps == 0 or gstep == max_steps:
                acc = evaluate(model, ev_loader); log.append({"step": gstep, "eval_accuracy": acc})
                tqdm.write(f"[{tag}] {gstep:,}/{max_steps:,} acc={acc:.4f}")
            if gstep % save_steps == 0 or gstep == max_steps:
                torch.save(model.state_dict(), f"{out_dir}/checkpoint-{tag}-{gstep}.pt")
            if gstep >= max_steps: break
    pbar.close(); return log

# ------------------------------------------------------------------
# util
# ------------------------------------------------------------------

def acc_pairs(hist):
    return [(h["step"], h["eval_accuracy"]) for h in hist]


def pick_ckpts(pairs, surge_th=0.03):
    """
    4 개 체크포인트 선택:
    (1) 급증 직전
    (2) 급증 도중(증가폭 최대)
    (3) 급증 완료 후 첫 plateau
    (4) 최고 accuracy
    """
    pairs.sort(key=lambda x: x[0])          # 시간 순
    steps, accs = zip(*pairs)

    # -- 증가량 계산
    diffs = np.diff(accs)
    k     = int(np.argmax(diffs))           # 가장 가파른 상승점

    pre   = steps[max(k-1, 0)]
    during= steps[k]

    # plateau : diff가 surge_th 보다 작아지는 첫 지점
    post  = during
    for j in range(k+1, len(diffs)):
        if diffs[j] < surge_th:
            post = steps[j+1]; break

    best  = steps[int(np.argmax(accs))]
    # 중복 제거 & 정렬
    sel = sorted({pre, during, post, best})
    return sel

# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main(cfg):
    set_seed(cfg.seed)
    out_root = Path(cfg.out); (out_root / "plots").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # ❶ 준비: 학생용 데이터 로더 (train / eval)
    # ------------------------------------------------------------------
    phase_steps = cfg.phase_steps
    tr_S, ev_S = make_loader(phase_steps, cfg.data_dim, cfg.k, cfg.n_labels,
                             cfg.batch, cfg.seed + 1)

    # ------------------------------------------------------------------
    # ❷ 이미 학습된 Teacher ckpt들의 정확도 계산
    # ------------------------------------------------------------------
    log_T = []
    teacher_ckpts = sorted(out_root.glob("checkpoint-T-*.pt"),                           key=lambda p: int(p.stem.split('-')[-1]))
    if not teacher_ckpts:
        raise RuntimeError("checkpoint-T-*.pt 파일을 찾을 수 없습니다!")

    for p in teacher_ckpts:
        step = int(p.stem.split('-')[-1])
        tmodel = build_mlp(cfg.data_dim, cfg.n_labels, cfg.teacher_h)
        tmodel.load_state_dict(torch.load(p, map_location=DEVICE))
        acc = evaluate(tmodel, ev_S)
        log_T.append({"step": step, "eval_accuracy": acc})

    ckpts = pick_ckpts(acc_pairs(log_T))          # 4 개 선택
    plot_teacher(log_T, ckpts, out_root / "plots/teacher_acc.png")

    # teacher 인스턴스 하나를 재사용 (가중치는 루프마다 덮어씀)
    teacher = build_mlp(cfg.data_dim, cfg.n_labels, cfg.teacher_h)
    # Iterate ckpt sets
    for idx, step_sel in enumerate(ckpts, 1):
        # ----- load teacher snapshot -----
        teacher.load_state_dict(torch.load(out_root / f"checkpoint-T-{step_sel}.pt"))

        # Phase-A: snapshot T → Small S
        small_S = build_mlp(cfg.data_dim, cfg.n_labels, cfg.student_h)
        train(small_S, tr_S, ev_S, kd=True, teacher=teacher, lr=cfg.lr, alpha=cfg.alpha, temp=cfg.temp,
              tag=f"A{idx}", out_dir=cfg.out, eval_steps=cfg.eval_steps, save_steps=cfg.save_steps, max_steps=phase_steps)

        # small teacher for Phase-B
        small_T = build_mlp(cfg.data_dim, cfg.n_labels, cfg.student_h)
        small_T.load_state_dict(torch.load(out_root / f"checkpoint-A{idx}-{phase_steps}.pt"))

        # Phase-B: SmallT → BigS
        big_S = build_mlp(cfg.data_dim, cfg.n_labels, cfg.teacher_h)
        train(big_S, tr_S, ev_S, kd=True, teacher=small_T, lr=cfg.lr, alpha=cfg.alpha, temp=cfg.temp,
              tag=f"B{idx}", out_dir=cfg.out, eval_steps=cfg.eval_steps, save_steps=cfg.save_steps, max_steps=phase_steps)

        # big teacher for Phase-C
        big_T = build_mlp(cfg.data_dim, cfg.n_labels, cfg.teacher_h)
        big_T.load_state_dict(torch.load(out_root / f"checkpoint-B{idx}-{phase_steps}.pt"))

        # Phase-C: BigT → Fresh SmallS
        fresh_S = build_mlp(cfg.data_dim, cfg.n_labels, cfg.student_h)
        train(fresh_S, tr_S, ev_S, kd=True, teacher=big_T, lr=cfg.lr, alpha=cfg.alpha, temp=cfg.temp,
              tag=f"C{idx}", out_dir=cfg.out, eval_steps=cfg.eval_steps, save_steps=cfg.save_steps, max_steps=phase_steps)

    print("[DONE] teacher 1e7 step, each student phase 5e5 step finished")

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="outputs_Tsplit")
    ap.add_argument("--data_dim", type=int, default=100)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--n_labels", type=int, default=2)
    ap.add_argument("--teacher_h", type=int, default=50000)
    ap.add_argument("--student_h", type=int, default=100)
    ap.add_argument("--teacher_steps", type=int, default=int(1e7))
    ap.add_argument("--phase_steps", type=int, default=int(5e5))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--temp", type=float, default=1e-4)
    ap.add_argument("--eval_steps", type=int, default=100000)
    ap.add_argument("--save_steps", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)

if __name__ == "__main__":
    cli()

