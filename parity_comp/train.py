#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sparse-Parity Role-Reversal KD  (Phase-0‥D + ckpt 정리 + plotting)
─────────────────────────────────────────────────────────────────
* Phase-0 : big-teacher supervised  (T = 1e7 step)
* Phase-A : big-teacher  → small-student KD
* Phase-B : (A-ckpt별)  small-teacher → big-student KD
* Phase-C : (B-ckpt별)  big-teacher   → fresh small-student KD
* Phase-D : (C-ckpt별)  small-teacher → fresh big-student KD

플롯
────
phase0_curve.png, phaseA_curve.png, phaseB_set{i}.png …
corr_phaseA_curve.png, corr_phaseB_set{i}.png … (deg-1 corr 꺾은선)

※ Δ = 0.005, phase_steps = 5e5, seed 고정
"""
# ───────────────────────────────────────────────────────────────
import argparse, json, os, random, shutil
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("high")

# ───────────────────────────────────────────────────────────────
# Seed -------------------------------------------------------------------------
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ───────────────────────────────────────────────────────────────
# Data -------------------------------------------------------------------------
def _boolean_tree_features(n_labels, dim, k):
    assert (n_labels-1)*k <= dim
    return [range(i, i+k) for i in range(0, (n_labels-1)*k, k)]

def _boolean_tree_samples(n, dim, n_labels, feats):
    X = 2*np.random.randint(0,2,(n,dim))-1
    scores = np.zeros((n,n_labels))
    def _score(x,lbl):
        s=0
        while lbl>1:
            f = feats[lbl//2-1]
            s += (1-2*(lbl%2))*np.prod(x[:,f],axis=-1); lbl//=2
        return s
    for lbl in range(n_labels):
        scores[:,lbl] = _score(X, lbl+n_labels)
    y = scores.argmax(-1).astype(np.int64)
    return X.astype(np.float32), y

class ParityDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X)
        self.y = torch.as_tensor(y).long()
    def __len__(self): return len(self.X)
    def __getitem__(self,i): return self.X[i], self.y[i]

def make_loader(n,dim,k,lab,batch,seed,eval_only=False):
    set_seed(seed)
    feats=_boolean_tree_features(lab,dim,k)
    X,y=_boolean_tree_samples(n+10_000,dim,lab,feats)
    Xtr,ytr,Xev,yev=X[:-10000],y[:-10000],X[-10000:],y[-10000:]
    ev=DataLoader(ParityDS(Xev,yev),2048,False,num_workers=0,pin_memory=True)
    if eval_only: return ev
    tr=DataLoader(ParityDS(Xtr,ytr),batch,True,num_workers=0,pin_memory=True)
    return tr,ev

# ───────────────────────────────────────────────────────────────
# Model ------------------------------------------------------------------------
def build_mlp(d_in,d_out,hid):
    return torch.nn.Sequential(
        torch.nn.Linear(d_in,hid), torch.nn.ReLU(),
        torch.nn.Linear(hid,d_out)
    ).to(DEVICE)

def kd_loss(s_logits, t_logits, y, alpha, temp):
    ce = F.cross_entropy(s_logits, y)
    q_t = F.softmax(t_logits / temp, dim=-1).detach()
    log_p_s = F.log_softmax(s_logits, dim=-1)
    kl = torch.mean(torch.sum(q_t * -log_p_s, dim=-1))
    return alpha * ce + (1 - alpha) * kl

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); tot=n=0
    for X,y in loader:
        X,y = X.to(DEVICE), y.to(DEVICE)
        tot += (model(X).argmax(-1)==y).sum().item(); n+=len(X)
    return tot/n

# ───────────────────────────────────────────────────────────────
# Train loop -------------------------------------------------------------------
def train(model,tr,ev,*,kd,teacher,lr,alpha,temp,
          tag,out_dir,eval_steps,save_steps,max_steps):
    optim = torch.optim.SGD(model.parameters(),lr=lr)
    log=[]; g=0; pbar=tqdm(total=max_steps,desc=tag)
    if teacher: teacher.eval()
    Path(out_dir).mkdir(parents=True,exist_ok=True)
    while g<max_steps:
        for X,y in tr:
            X,y = X.to(DEVICE), y.to(DEVICE)
            if kd:
                loss = kd_loss(model(X), teacher(X).detach(),
                               y, alpha, temp)
            else:
                loss = F.cross_entropy(model(X), y)
            loss.backward(); optim.step(); optim.zero_grad()
            g+=1; pbar.update(1)
            if g%eval_steps==0 or g==max_steps:
                acc = evaluate(model, ev)
                log.append({"step":g,"acc":acc})
                tqdm.write(f"[{tag}] {g:,}/{max_steps:,} acc={acc:.4f}")
            if g%save_steps==0 or g==max_steps:
                torch.save(model.state_dict(),
                           f"{out_dir}/ckpt-{tag}-{g}.pt")
            if g>=max_steps: break
    pbar.close()
    json.dump(log, open(f"log_{tag}.json","w"), indent=2)
    return log

# ───────────────────────────────────────────────────────────────
# Checkpoint selection ---------------------------------------------------------
def pick_ckpts(pairs, surge_th=0.03):
    """Tsplit과 동일: 급증 전·중·후 + 최고"""
    pairs.sort(key=lambda x:x[0])
    steps, accs = zip(*pairs)
    diffs = np.diff(accs)
    k = int(np.argmax(diffs))
    pre = steps[max(k-1,0)]
    during = steps[k]
    post = during
    for j in range(k+1,len(diffs)):
        if diffs[j] < surge_th:
            post = steps[j+1]; break
    best = steps[int(np.argmax(accs))]
    return sorted({pre,during,post,best})

def pick_ckpts_phaseA(pairs, th=0.03):
    """Phase-A 내부 4개 선정 (기존 함수 유지)"""
    return pick_ckpts(pairs, th)

def rise_region(pairs, Δ=0.005):
    pairs.sort(key=lambda x:x[0]); steps,acc = zip(*pairs)
    dif = np.diff(acc)
    rise = np.where(dif>Δ)[0]
    if len(rise)==0: return None,None,None
    s_idx, e_idx = rise[0], rise[-1]+1
    mid = int(round((steps[s_idx]+steps[e_idx])/2))
    return steps[s_idx], steps[e_idx], mid

def best_mid_worst(pairs):
    best=max(pairs,key=lambda x:x[1])
    worst=min(pairs,key=lambda x:x[1])
    mid_target=(best[1]+worst[1])/2
    mid=min(pairs,key=lambda x:abs(x[1]-mid_target))
    return best,mid,worst

# ───────────────────────────────────────────────────────────────
# Plot utils -------------------------------------------------------------------
def plot_curve(pairs, *, sel_steps=None,
               add_lines:List[Tuple[float,str]]=[],
               title:str="", fname:str="curve.png"):
    pairs.sort(key=lambda x:x[0]); x,y = zip(*pairs)
    plt.figure(figsize=(6,4))
    plt.plot(x,y,label=title)
    if sel_steps:
        for s in sel_steps:
            plt.axvline(s, ls="--", lw=1, c="red")
    for val,lab in add_lines:
        plt.hlines(val, x[0], x[-1], linestyles=':', label=lab)
    plt.xlabel("step"); plt.ylabel("accuracy"); plt.legend()
    plt.tight_layout(); plt.savefig(fname,dpi=250); plt.close()

@torch.no_grad()
def corr_curve(ckpt_pattern:str, model_fn,
               dim,k,lab, out_path, loader):
    xs,ys = [],[]
    for p in sorted(Path(ckpt_pattern).parent.glob(
                    Path(ckpt_pattern).name),
                    key=lambda q:int(q.stem.split('-')[-1])):
        step = int(p.stem.split('-')[-1])
        mdl = model_fn(dim,lab); mdl.load_state_dict(
              torch.load(p,map_location=DEVICE))
        tot=None;n=0; mdl.eval()
        for X,_ in loader:
            X = X.to(DEVICE)
            p1 = torch.softmax(mdl(X),-1)[:,1]
            prod = (p1[:,None]*X).cpu().numpy()
            tot = prod.sum(0) if tot is None else tot+prod.sum(0)
            n += len(X)
        corr_mean = np.abs(tot/n)[:k].mean()
        xs.append(step); ys.append(corr_mean)
    plt.figure(figsize=(6,4))
    plt.plot(xs,ys,marker='o')
    plt.xlabel("step"); plt.ylabel("mean |corr| (deg-1)")
    plt.title(Path(ckpt_pattern).parent.name)
    plt.tight_layout(); plt.savefig(out_path,dpi=250); plt.close()

# ───────────────────────────────────────────────────────────────
# Copy & delete ---------------------------------------------------------------
def move_keep(src_root:Path, keep_steps:List[int],
              tag:str, dst_root:Path):
    dst_root.mkdir(parents=True,exist_ok=True)
    for p in src_root.glob(f"ckpt-{tag}-*.pt"):
        step = int(p.stem.split('-')[-1])
        if step in keep_steps:
            shutil.copy2(p, dst_root/p.name)
        p.unlink()

# ───────────────────────────────────────────────────────────────
def nearest_logged_step(target_step, logged_pairs):
    # logged_pairs: List[(step, acc)]
    return min(logged_pairs, key=lambda x: abs(x[0]-target_step))[0]
# Main ------------------------------------------------------------------------
def main(cfg):
    root = Path(__file__).resolve().parent
    out  = Path(cfg.out_dir);   out.mkdir(exist_ok=True)
    parity_dir = Path(cfg.parity_dir); parity_dir.mkdir(parents=True,exist_ok=True)
    dim,k,lab = cfg.data_dim, cfg.k, cfg.n_labels

    # ─ Phase-0 ──────────────────────────────────────────────────
    ev_global = make_loader(0,dim,k,lab,0,cfg.seed,eval_only=True)
    set_seed(cfg.seed)
    
    # plateau 시작점(3번째) 사용
    bigT = build_mlp(dim,lab,cfg.teacher_h)
    bigT.load_state_dict(torch.load(
        "checkpoint-T-4300000.pt",
        map_location=DEVICE))

    # ─ Phase-A ──────────────────────────────────────────────────
    trA,evA = make_loader(cfg.phase_steps,dim,k,lab,
                          cfg.batch,cfg.seed+1)
    smallS = build_mlp(dim,lab,cfg.student_h)
    logA = train(smallS,trA,evA,kd=True,teacher=bigT,
                 lr=cfg.lr,alpha=cfg.alpha,temp=cfg.temp,
                 tag="PA",out_dir=out,
                 eval_steps=cfg.eval, save_steps=cfg.save,
                 max_steps=cfg.phase_steps)
    pairsA=[(h["step"],h["acc"]) for h in logA]
    selA = pick_ckpts_phaseA(pairsA)
    move_keep(out, selA, "PA", parity_dir/"phaseA")
    acc_bigT = evaluate(bigT, ev_global)
    plot_curve(pairsA, sel_steps=selA,
               add_lines=[(acc_bigT, "Teacher acc")],
               title="Phase-A Student", fname="phaseA_curve.png")
    # corr curve (Phase-A)
    corr_curve(str(parity_dir/'phaseA'/'ckpt-PA-*.pt'),
               lambda d_in,d_out: build_mlp(d_in,d_out,cfg.student_h),
               dim,k,lab, "corr_phaseA_curve.png",
               ev_global)

    # ─ Phase-B / C / D 루프 ────────────────────────────────────
    set_idx=0
    for sA in selA:
        set_idx+=1
        # ---------- Phase-B ----------
        smallT = build_mlp(dim,lab,cfg.student_h)
        smallT.load_state_dict(torch.load(
            parity_dir/'phaseA'/f"ckpt-PA-{sA}.pt",
            map_location=DEVICE))
        bigS = build_mlp(dim,lab,cfg.teacher_h)
        tagB=f"PB{set_idx}"
        logB = train(bigS,trA,evA,kd=True,teacher=smallT,
                     lr=cfg.lr,alpha=cfg.alpha,temp=cfg.temp,
                     tag=tagB,out_dir=out,
                     eval_steps=cfg.eval, save_steps=cfg.save,
                     max_steps=cfg.phase_steps)
        pairsB=[(h["step"],h["acc"]) for h in logB]
        bestB,midB,_=best_mid_worst(pairsB)
        s_rs, e_rs, midRise = rise_region(pairsB, Δ=0.005)
        if midRise:
            midRise = nearest_logged_step(midRise, pairsB)
        keepB = [bestB[0], midB[0], midRise] if midRise else [bestB[0], midB[0]]
        if bestB[1] < 0.6: keepB=[]    # skip set
        move_keep(out, keepB, tagB, parity_dir/f"phaseB_set{set_idx}")
        plot_curve(pairsB, sel_steps=keepB,
                   add_lines=[(pairsA[[p[0] for p in pairsA].index(sA)][1],"selA")],
                   title=f"Phase-B set{set_idx}",
                   fname=f"phaseB_set{set_idx}.png")
        corr_curve(str(parity_dir/f"phaseB_set{set_idx}" / f"ckpt-{tagB}-*.pt"),
                   lambda d_in,d_out: build_mlp(d_in,d_out,cfg.teacher_h),
                   dim,k,lab, f"corr_phaseB_set{set_idx}.png",
                   ev_global)

        # ---------- Phase-C ----------
        sub_idx=0
        for keep_step in keepB:
            sub_idx+=1
            bigTB = build_mlp(dim,lab,cfg.teacher_h)
            bigTB.load_state_dict(torch.load(
                parity_dir/f"phaseB_set{set_idx}"/f"ckpt-{tagB}-{keep_step}.pt",
                map_location=DEVICE))
            smallS0 = build_mlp(dim,lab,cfg.student_h)
            tagC=f"PC{set_idx}_{sub_idx}"
            logC = train(smallS0,trA,evA,kd=True,teacher=bigTB,
                         lr=cfg.lr,alpha=cfg.alpha,temp=cfg.temp,
                         tag=tagC,out_dir=out,
                         eval_steps=cfg.eval, save_steps=cfg.save,
                         max_steps=cfg.phase_steps)
            pairsC=[(h["step"],h["acc"]) for h in logC]
            bestC,midC,_=best_mid_worst(pairsC)
            sC, eC, midRiseC = rise_region(pairsC, Δ=0.005)
            if midRiseC:
                midRiseC = nearest_logged_step(midRiseC, pairsC)
            keepC = [bestC[0], midRiseC, midC[0]] if midRiseC else [bestC[0], midC[0]]
            move_keep(out, keepC, tagC, parity_dir/f"phaseC_set{set_idx}_{sub_idx}")
            plot_curve(pairsC, sel_steps=keepC,
                       add_lines=[(pairsA[[p[0] for p in pairsA].index(sA)][1],"selA"),
                                  (bestB[1],"selB")],
                       title=f"Phase-C set{set_idx}_{sub_idx}",
                       fname=f"phaseC_set{set_idx}_{sub_idx}.png")
            corr_curve(str(parity_dir/f"phaseC_set{set_idx}_{sub_idx}" /
                           f"ckpt-{tagC}-*.pt"),
                       lambda d_in,d_out: build_mlp(d_in,d_out,cfg.student_h),
                       dim,k,lab,
                       f"corr_phaseC_set{set_idx}_{sub_idx}.png",
                       ev_global)

            # ---------- Phase-D ----------
            deep_idx=0
            for stC in keepC:
                deep_idx+=1
                smallTC = build_mlp(dim,lab,cfg.student_h)
                smallTC.load_state_dict(torch.load(
                    parity_dir/f"phaseC_set{set_idx}_{sub_idx}" /
                    f"ckpt-{tagC}-{stC}.pt", map_location=DEVICE))
                bigS0 = build_mlp(dim,lab,cfg.teacher_h)
                tagD=f"PD{set_idx}_{sub_idx}_{deep_idx}"
                logD = train(bigS0,trA,evA,kd=True,teacher=smallTC,
                             lr=cfg.lr,alpha=cfg.alpha,temp=cfg.temp,
                             tag=tagD,out_dir=out,
                             eval_steps=cfg.eval, save_steps=cfg.save,
                             max_steps=cfg.phase_steps)
                pairsD=[(h["step"],h["acc"]) for h in logD]
                plot_curve(pairsD, sel_steps=[cfg.phase_steps],
                           add_lines=[(bestB[1],"B"), (bestC[1],"C")],
                           title=f"Phase-D set{set_idx}_{sub_idx}_{deep_idx}",
                           fname=f"phaseD_set{set_idx}_{sub_idx}_{deep_idx}.png")
                corr_curve(str(out/f"ckpt-{tagD}-*.pt"),
                           lambda d_in,d_out: build_mlp(d_in,d_out,cfg.teacher_h),
                           dim,k,lab,
                           f"corr_phaseD_{set_idx}_{sub_idx}_{deep_idx}.png",
                           ev_global)
    print("✅ 전체 파이프라인 종료")

# ───────────────────────────────────────────────────────────────
def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="outputs_tmp")
    p.add_argument("--parity_dir", type=str, default="/local_datasets/parity")
    p.add_argument("--data_dim", type=int, default=100)
    p.add_argument("--k", type=int, default=6)
    p.add_argument("--n_labels", type=int, default=2)
    p.add_argument("--teacher_h", type=int, default=50000)
    p.add_argument("--student_h", type=int, default=100)
    p.add_argument("--teacher_steps", type=int, default=10_000_000)
    p.add_argument("--phase_steps", type=int, default=500_000)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=0.0)
    p.add_argument("--temp", type=float, default=1e-4)
    p.add_argument("--eval", type=int, default=100_000)
    p.add_argument("--save", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=42)
    cfg = p.parse_args(); main(cfg)

if __name__ == "__main__":
    cli()
