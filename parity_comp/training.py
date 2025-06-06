import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from models import DEVICE, kd_loss, evaluate


def train(model, tr_loader, ev_loader, *, kd, teacher, lr, alpha, temp, tag, out_dir,
          eval_steps, save_steps, max_steps):
    """Generic training loop for parity models."""
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    log = []
    g = 0
    pbar = tqdm(total=max_steps, desc=tag)
    if teacher:
        teacher.eval()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    acc_json_dir = out_dir / "accuracy" / "json"
    acc_json_dir.mkdir(parents=True, exist_ok=True)
    while g < max_steps:
        for X, y in tr_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            if kd:
                loss = kd_loss(model(X), teacher(X).detach(), y, alpha, temp)
            else:
                loss = F.cross_entropy(model(X), y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            g += 1
            pbar.update(1)
            if g % eval_steps == 0 or g == max_steps:
                acc = evaluate(model, ev_loader)
                log.append({"step": g, "acc": acc})
                tqdm.write(f"[{tag}] {g:,}/{max_steps:,} acc={acc:.4f}")
            if g % save_steps == 0 or g == max_steps:
                torch.save(model.state_dict(), out_dir / f"ckpt-{tag}-{g}.pt")
            if g >= max_steps:
                break
    pbar.close()
    json.dump(log, open(acc_json_dir / f"log_{tag}.json", "w"), indent=2)
    return log
