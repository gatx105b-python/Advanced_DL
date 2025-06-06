import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _boolean_tree_features(n_labels: int, dim: int, k: int):
    assert (n_labels - 1) * k <= dim
    return [range(i, i + k) for i in range(0, (n_labels - 1) * k, k)]


def _boolean_tree_samples(n: int, dim: int, n_labels: int, feats):
    X = 2 * np.random.randint(0, 2, (n, dim)) - 1
    scores = np.zeros((n, n_labels))

    def _score(x, lbl):
        s = 0
        while lbl > 1:
            f = feats[lbl // 2 - 1]
            s += (1 - 2 * (lbl % 2)) * np.prod(x[:, f], axis=-1)
            lbl //= 2
        return s

    for lbl in range(n_labels):
        scores[:, lbl] = _score(X, lbl + n_labels)
    y = scores.argmax(-1).astype(np.int64)
    return X.astype(np.float32), y


class ParityDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X)
        self.y = torch.as_tensor(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_loader(n: int, dim: int, k: int, lab: int, batch: int, seed: int, eval_only: bool = False):
    """Return train and eval DataLoaders for the parity dataset."""
    set_seed(seed)
    feats = _boolean_tree_features(lab, dim, k)
    X, y = _boolean_tree_samples(n + 10_000, dim, lab, feats)
    Xtr, ytr, Xev, yev = X[:-10000], y[:-10000], X[-10000:], y[-10000:]
    ev = DataLoader(ParityDS(Xev, yev), 2048, False, num_workers=0, pin_memory=True)
    if eval_only:
        return ev
    tr = DataLoader(ParityDS(Xtr, ytr), batch, True, num_workers=0, pin_memory=True)
    return tr, ev
