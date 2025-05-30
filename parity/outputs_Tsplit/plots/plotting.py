# plot_parity_phases.py
# -*- coding: utf-8 -*-
"""
✓ phase-0 Teacher 전체 궤적 → 최고 accuracy(수평 점선)
✓ 각 idx(1‒4)에 대해
   • Phase-A  : 꺾은선 (모든 ckpt)
   • Phase-C  : 꺾은선 (모든 ckpt)
   • Phase-B 의 Teacher(= A 의 마지막 ckpt) → 정확도 수평 점선
총 4개 png 파일: phase_plot_1.png … phase_plot_4.png
"""

import re, torch, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------
# 데이터셋 & 모델 util (기존 parity 코드 그대로 간단히 재현)
# ---------------------------------------------------------------------
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

def make_eval_loader(dim=100, k=6, n_labels=2, batch=2048, seed=0):
    np.random.seed(seed)
    feats = _boolean_tree_features(n_labels, dim, k)
    X, y = _boolean_tree_samples(10_000, dim, n_labels, feats)
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

# ---------------------------------------------------------------------
# 체크포인트 스캔 & 정확도 계산
# ---------------------------------------------------------------------
def collect_acc(pattern, hidden, eval_loader):
    """
    pattern : glob 패턴 (e.g. 'checkpoint-A1-*.pt')
    hidden  : MLP hidden dim
    return  : list[(step, acc)], 정렬 O
    """
    pairs = []
    for p in sorted(Path(".").glob(pattern),
                    key=lambda f: int(re.findall(r'-(\d+)\.pt$', f.name)[0])):
        step = int(re.findall(r'-(\d+)\.pt$', p.name)[0])
        model = build_mlp(100, 2, hidden)
        model.load_state_dict(torch.load(p, map_location=DEVICE))
        acc = evaluate(model, eval_loader)
        pairs.append((step, acc))
    return pairs

# ---------------------------------------------------------------------
# 메인 프로시저
# ---------------------------------------------------------------------
def main():
    eval_loader = make_eval_loader()

    # 0) Teacher 전체 궤적
    teacher_pairs = collect_acc('checkpoint-T-*.pt', hidden=50_000, eval_loader=eval_loader)
    teacher_pairs.sort(key=lambda x: x[0])
    teacher_best_acc = max(a for _, a in teacher_pairs)

    # idx = 1‥4
    for idx in range(1, 5):
        # Phase-A & C 궤적
        A_pairs = collect_acc(f'checkpoint-A{idx}-*.pt', hidden=100,     eval_loader=eval_loader)
        C_pairs = collect_acc(f'checkpoint-C{idx}-*.pt', hidden=100,     eval_loader=eval_loader)

        # Phase-B 의 teacher = A 마지막 ckpt (step 500 000)
        small_T_file = Path(f'checkpoint-A{idx}-500000.pt')
        if not small_T_file.exists():
            print(f"[경고] {small_T_file} 없음 → idx {idx} 스킵"); continue
        small_T = build_mlp(100, 2, 100)
        small_T.load_state_dict(torch.load(small_T_file, map_location=DEVICE))
        small_T_acc = evaluate(small_T, eval_loader)

        # ----------------- Plot -----------------
        plt.figure(figsize=(6,4))

        # Phase-A
        if A_pairs:
            xs, ys = zip(*A_pairs)
            plt.plot(xs, ys, label=f'Phase-A{idx}')

        # Phase-C
        if C_pairs:
            xs, ys = zip(*C_pairs)
            plt.plot(xs, ys, label=f'Phase-C{idx}')

        # Teacher 최고 accuracy
        plt.axhline(teacher_best_acc, ls='--', lw=1, label='Teacher best')

        # Phase-B teacher 최고 accuracy
        plt.axhline(small_T_acc, ls='--', lw=1, color='gray', label='B-teacher best')

        plt.xlabel('step'); plt.ylabel('accuracy')
        plt.title(f'Parity KD – idx {idx}')
        plt.legend()
        plt.tight_layout()
        save_path = f'phase_plot_{idx}.png'
        plt.savefig(save_path); plt.close()
        print(f"[✓] {save_path} 저장 완료")

if __name__ == "__main__":
    main()

