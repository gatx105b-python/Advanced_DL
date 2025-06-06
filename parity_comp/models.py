import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")


def build_mlp(d_in: int, d_out: int, hid: int):
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, hid),
        torch.nn.ReLU(),
        torch.nn.Linear(hid, d_out)
    ).to(DEVICE)


def kd_loss(s_logits, t_logits, y, alpha: float, temp: float):
    ce = F.cross_entropy(s_logits, y)
    q_t = F.softmax(t_logits / temp, dim=-1).detach()
    log_p_s = F.log_softmax(s_logits, dim=-1)
    kl = torch.mean(torch.sum(q_t * -log_p_s, dim=-1))
    return alpha * ce + (1 - alpha) * kl


def evaluate(model, loader):
    model.eval(); tot = n = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            tot += (model(X).argmax(-1) == y).sum().item(); n += len(X)
    return tot / n
