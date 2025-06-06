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
import argparse
from pathlib import Path
from typing import List, Tuple
import torch

from data import make_loader, set_seed
from models import DEVICE, build_mlp, evaluate
from training import train
from utils import (
    pick_ckpts,
    pick_ckpts_phaseA,
    rise_region,
    best_mid_worst,
    plot_curve,
    corr_curve,
    move_keep,
    nearest_logged_step,
)

# Main ------------------------------------------------------------------------
def main(cfg):
    root = Path(__file__).resolve().parent
    out  = Path(cfg.out_dir)
    (out / "accuracy" / "json").mkdir(parents=True, exist_ok=True)
    (out / "accuracy" / "plotting").mkdir(parents=True, exist_ok=True)
    (out / "correlation" / "json").mkdir(parents=True, exist_ok=True)
    (out / "correlation" / "plotting").mkdir(parents=True, exist_ok=True)
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
    plot_curve(
        pairsA,
        sel_steps=selA,
        add_lines=[(acc_bigT, "Teacher acc")],
        title="Phase-A Student",
        fname="phaseA_curve.png",
        out_dir=out / "accuracy" / "plotting",
    )
    # corr curve (Phase-A)
    corr_curve(
        str(parity_dir / "phaseA" / "ckpt-PA-*.pt"),
        lambda d_in, d_out: build_mlp(d_in, d_out, cfg.student_h),
        dim,
        k,
        lab,
        out / "correlation" / "plotting" / "corr_phaseA_curve.png",
        ev_global,
        out_json=out / "correlation" / "json" / "corr_phaseA_curve.json",
    )

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
        plot_curve(
            pairsB,
            sel_steps=keepB,
            add_lines=[(pairsA[[p[0] for p in pairsA].index(sA)][1], "selA")],
            title=f"Phase-B set{set_idx}",
            fname=f"phaseB_set{set_idx}.png",
            out_dir=out / "accuracy" / "plotting",
        )
        corr_curve(
            str(parity_dir / f"phaseB_set{set_idx}" / f"ckpt-{tagB}-*.pt"),
            lambda d_in, d_out: build_mlp(d_in, d_out, cfg.teacher_h),
            dim,
            k,
            lab,
            out / "correlation" / "plotting" / f"corr_phaseB_set{set_idx}.png",
            ev_global,
            out_json=out / "correlation" / "json" / f"corr_phaseB_set{set_idx}.json",
        )

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
            plot_curve(
                pairsC,
                sel_steps=keepC,
                add_lines=[
                    (pairsA[[p[0] for p in pairsA].index(sA)][1], "selA"),
                    (bestB[1], "selB"),
                ],
                title=f"Phase-C set{set_idx}_{sub_idx}",
                fname=f"phaseC_set{set_idx}_{sub_idx}.png",
                out_dir=out / "accuracy" / "plotting",
            )
            corr_curve(
                str(parity_dir / f"phaseC_set{set_idx}_{sub_idx}" / f"ckpt-{tagC}-*.pt"),
                lambda d_in, d_out: build_mlp(d_in, d_out, cfg.student_h),
                dim,
                k,
                lab,
                out / "correlation" / "plotting" / f"corr_phaseC_set{set_idx}_{sub_idx}.png",
                ev_global,
                out_json=out / "correlation" / "json" / f"corr_phaseC_set{set_idx}_{sub_idx}.json",
            )

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
                plot_curve(
                    pairsD,
                    sel_steps=[cfg.phase_steps],
                    add_lines=[(bestB[1], "B"), (bestC[1], "C")],
                    title=f"Phase-D set{set_idx}_{sub_idx}_{deep_idx}",
                    fname=f"phaseD_set{set_idx}_{sub_idx}_{deep_idx}.png",
                    out_dir=out / "accuracy" / "plotting",
                )
                corr_curve(
                    str(out / f"ckpt-{tagD}-*.pt"),
                    lambda d_in, d_out: build_mlp(d_in, d_out, cfg.teacher_h),
                    dim,
                    k,
                    lab,
                    out
                    / "correlation"
                    / "plotting"
                    / f"corr_phaseD_{set_idx}_{sub_idx}_{deep_idx}.png",
                    ev_global,
                    out_json=
                    out
                    / "correlation"
                    / "json"
                    / f"corr_phaseD_{set_idx}_{sub_idx}_{deep_idx}.json",
                )
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
