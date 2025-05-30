# -*- coding: utf-8 -*-
"""
bert_role_reversal_kd_v2.py
===========================
End‑to‑end **role‑reversal Knowledge Distillation** on GLUE/SST‑2 (BERT family)
-------------------------------------------------------------------------------
*Single command* → four phases (0/A/B/C) + per‑checkpoint plots.

Key updates (2025‑05‑24)
~~~~~~~~~~~~~~~~~~~~~~~~
1. **Phase‑A checkpoint mining = 4** → *best* (`max eval_accuracy`) **+ 3 largest‑Δ**.
2. **Batch size defaults doubled** → `train_bs = 64`, `eval_bs = 256` (tuned for ≥24 GB GPUs).
3. **Combined plots** now draw the **entire Phase‑A curve** (no front truncation).
4. Minor refactors & clearer CLI flags (`--top_k_delta`).

Run (all defaults)
~~~~~~~~~~~~~~~~~~
```bash
python bert_role_reversal_kd_v2.py
```
Override any hyper‑parameter via CLI, e.g. `--epochs 3 --train_bs 32`.
"""

import argparse
import json
import os
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from pynvml import (
    nvmlDeviceGetHandleByIndex,
    nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetName,
    nvmlInit,
)
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def safe_decode(name):
    """Return GPU name as str regardless of NVML return type."""
    return name.decode() if isinstance(name, (bytes, bytearray)) else str(name)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------------------------------------------------------
# GPU + accuracy monitor for tqdm
# -----------------------------------------------------------------------------

class AccGPUMonitorCallback(TrainerCallback):
    """Injects GPU memory + eval accuracy into tqdm & console."""

    def __init__(self):
        super().__init__()
        self.last_acc = None
        if torch.cuda.is_available():
            nvmlInit()
            self.handle = nvmlDeviceGetHandleByIndex(0)
            self.total_gb = nvmlDeviceGetMemoryInfo(self.handle).total / 1024 ** 3
            gpu_name = safe_decode(nvmlDeviceGetName(self.handle))
            tqdm.write(f"[GPU] {gpu_name} – {self.total_gb:.0f} GB detected")

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "eval_accuracy" in logs:
            self.last_acc = logs["eval_accuracy"]
            logs["loss"] = self.last_acc  # show accuracy value in bar
        if torch.cuda.is_available():
            mem = nvmlDeviceGetMemoryInfo(self.handle)
            used_gb = mem.used / 1024 ** 3
            pct = used_gb / self.total_gb * 100
            msg = (
                f"[step {state.global_step:>6}] GPU {used_gb:.1f}/{self.total_gb:.0f} GB ({pct:.0f}%) | "
                f"acc {self.last_acc:.4f}" if self.last_acc is not None else "…"
            )
            tqdm.write(msg)


# -----------------------------------------------------------------------------
# KD helpers
# -----------------------------------------------------------------------------

def soft_cross_entropy(p_logits, q_logits, T: float = 2.0):
    p = F.log_softmax(p_logits / T, dim=-1)
    q = F.softmax(q_logits / T, dim=-1)
    return F.kl_div(p, q, reduction="batchmean") * (T ** 2)


class KDTrainer(Trainer):
    """Trainer subclass implementing teacher→student KD."""

    def __init__(self, teacher_model, alpha=0.5, temperature=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.to(self.model.device)
        self.alpha = alpha
        self.T = temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        s_out = model(**inputs)
        with torch.no_grad():
            t_out = self.teacher(**inputs)
        ce = F.cross_entropy(s_out.logits, labels)
        kd = soft_cross_entropy(s_out.logits, t_out.logits, self.T)
        loss = self.alpha * ce + (1 - self.alpha) * kd
        return (loss, s_out) if return_outputs else loss


# -----------------------------------------------------------------------------
# Data utilities
# -----------------------------------------------------------------------------

def prepare_data(tokenizer):
    ds = load_dataset("glue", "sst2")

    def tok(batch):
        return tokenizer(batch["sentence"], truncation=True)

    ds_enc = ds.map(tok, batched=True)
    keep = ["input_ids", "attention_mask", "label"]
    ds_enc = ds_enc.remove_columns([c for c in ds_enc["train"].column_names if c not in keep])
    collator = DataCollatorWithPadding(tokenizer)
    return ds_enc, collator


def metric_accuracy(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": (logits.argmax(-1) == labels).mean()}


def extract_acc(log_hist):
    return [(h["step"], h["eval_accuracy"]) for h in log_hist if "eval_accuracy" in h]


def plot_line(xs, ys, label, style="-", color=None):
    plt.plot(xs, ys, style, label=label, color=color)


def mine_checkpoints(acc_list, delta_threshold=0.01, top_k=3):
    """Return *top_k* steps with largest accuracy jump ≥ threshold."""
    acc_list.sort(key=lambda x: x[0])  # by step
    deltas = [
        (acc_list[i][0], acc_list[i][1] - acc_list[i - 1][1])
        for i in range(1, len(acc_list))
    ]
    deltas = [(s, d) for s, d in deltas if d >= delta_threshold]
    if not deltas:
        # evenly spaced fallback
        steps = [acc_list[len(acc_list) * i // (top_k + 1)][0] for i in range(1, top_k + 1)]
        return steps
    deltas.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in deltas[:top_k]]


# -----------------------------------------------------------------------------
# Phase runners
# -----------------------------------------------------------------------------

def run_teacher_finetune(args, ds, coll, tok):
    model = AutoModelForSequenceClassification.from_pretrained(args.teacher_model, num_labels=2)
    targs = TrainingArguments(
        output_dir=f"{args.out}/phase0_teacher",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        weight_decay=0.01,
        logging_steps=args.eval_steps,
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tok,
        data_collator=coll,
        compute_metrics=metric_accuracy,
        callbacks=[AccGPUMonitorCallback()],
    )
    trainer.train()

    # plot teacher curve
    acc = extract_acc(trainer.state.log_history)
    xs, ys = zip(*acc)
    plt.figure()
    plot_line(xs, ys, "Teacher (phase0)")
    plt.xlabel("step"); plt.ylabel("accuracy"); plt.legend();
    plt.savefig(f"{args.out}/plots/phase0_teacher.png", dpi=180)
    plt.close()

    return trainer.state.best_model_checkpoint or targs.output_dir, ys[-1]


def run_kd(args, out_dir, teacher, student, ds, coll, tok, tag, load_best=False):
    tr_args = TrainingArguments(
        output_dir=out_dir,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
        logging_steps=args.eval_steps,
        load_best_model_at_end=load_best,
    )
    tr = KDTrainer(
        teacher_model=teacher,
        alpha=args.alpha,
        temperature=args.temp,
        model=student,
        args=tr_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tok,
        data_collator=coll,
        compute_metrics=metric_accuracy,
        callbacks=[AccGPUMonitorCallback()],
    )
    tr.train()

    # save final model explicitly (ensures config + weights present)
    tr.save_model(out_dir)

    with open(Path(out_dir) / f"logs_{tag}.json", "w") as f:
        json.dump(tr.state.log_history, f, indent=2)

    best_ckpt = (
        tr.state.best_model_checkpoint if load_best and tr.state.best_model_checkpoint else out_dir
    )
    return tr.state.log_history, best_ckpt


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="outputs_role_rev", help="output root dir")
    p.add_argument("--teacher_model", type=str, default="bert-base-uncased")
    p.add_argument("--student_model", type=str, default="prajjwal1/bert-mini")
    # doubled batch sizes (tuned for 24 GB)
    p.add_argument("--train_bs", type=int, default=64)
    p.add_argument("--eval_bs", type=int, default=256)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-5)
    # KD hyper‑parameters
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--temp", type=float, default=2.0)
    p.add_argument("--delta_threshold", type=float, default=0.01)
    p.add_argument("--top_k_delta", type=int, default=3, help="#largest‑Δ checkpoints to keep")
    # misc
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args([])  # allow bare `python file.py`


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    set_global_seed(args.seed)
    Path(f"{args.out}/plots").mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.teacher_model)
    ds, coll = prepare_data(tok)

    # ---------------- Phase 0: teacher fine‑tune ------------------
    teacher_ckpt, teacher_acc = run_teacher_finetune(args, ds, coll, tok)

    # ---------------- Phase A: teacher → student ------------------
    student0 = AutoModelForSequenceClassification.from_pretrained(
        args.student_model, num_labels=2
    )
    teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_ckpt)

    phaseA_dir = f"{args.out}/phaseA_teacher2student"
    histA, _ = run_kd(
        args,
        phaseA_dir,
        teacher_model,
        student0,
        ds,
        coll,
        tok,
        "phaseA",
        load_best=False,
    )
    accA = extract_acc(histA)
    xsA, ysA = zip(*accA)

    # --- determine checkpoints: best + top‑k Δ -------------------
    best_step = xsA[int(np.argmax(ysA))]
    delta_steps = mine_checkpoints(accA, args.delta_threshold, args.top_k_delta)
    ckpt_steps = list(dict.fromkeys([best_step] + delta_steps))[: 1 + args.top_k_delta]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"][: len(ckpt_steps)]

    ckpt_paths = [f"{phaseA_dir}/checkpoint-{s}" for s in ckpt_steps]
    step_to_acc = {s: y for s, y in accA}

    # --- Phase‑A global plot + mark checkpoints -------------------
    plt.figure()
    plot_line(xsA, ysA, "Student (phaseA)")
    plt.axhline(teacher_acc, color="red", ls="--", label="Teacher acc (phase0)")
    plt.scatter(ckpt_steps, [step_to_acc[s] for s in ckpt_steps], color="black")
    for s in ckpt_steps:
        plt.annotate(f"{step_to_acc[s]:.3f}", (s, step_to_acc[s]), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8)
    plt.xlabel("step"); plt.ylabel("accuracy"); plt.legend();
    plt.savefig(f"{args.out}/plots/phaseA_student.png", dpi=180)
    plt.close()

    # store for combined plots
    combined = []

    # ---------------- Phase B & C loop per checkpoint ------------
    for idx, (step_sel, ckpt_path, col) in enumerate(zip(ckpt_steps, ckpt_paths, colors), 1):
        # ---- Phase‑A segment plot (from ckpt onwards) -----------
        cut = xsA.index(step_sel)
        xs_seg, ys_seg = xsA[cut:], ysA[cut:]
        plt.figure()
        plot_line(xs_seg, ys_seg, f"Student phaseA (ckpt{idx})", color=col)
        plt.axhline(teacher_acc, color=col, ls="--", label="Teacher phase0")
        plt.xlabel("step"); plt.ylabel("accuracy"); plt.legend();
        plt.savefig(f"{args.out}/plots/phaseA_student_ckpt{idx}.png", dpi=180)
        plt.close()

        # ---- Phase B: role‑reversal -----------------------------
        small_teacher = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
        big_student = AutoModelForSequenceClassification.from_pretrained(teacher_ckpt)
        phaseB_dir = f"{args.out}/phaseB_rev_{idx}"
        histB, big_ckpt = run_kd(
            args,
            phaseB_dir,
            small_teacher,
            big_student,
            ds,
            coll,
            tok,
            f"phaseB_{idx}",
            load_best=False,
        )
        xsB, ysB = zip(*extract_acc(histB))
        small_teacher_acc = step_to_acc[step_sel]
        plt.figure()
        plot_line(xsB, ysB, "Big‑student phaseB", color=col)
        plt.axhline(small_teacher_acc, color="red", ls="--", label="Small‑teacher acc")
        plt.xlabel("step"); plt.ylabel("accuracy"); plt.legend();
        plt.savefig(f"{args.out}/plots/phaseB_rev_{idx}.png", dpi=180)
        plt.close()

        # ---- Phase C: fresh student trained from big teacher ----
        fresh_student = AutoModelForSequenceClassification.from_pretrained(
            args.student_model, num_labels=2
        )
        big_teacher = AutoModelForSequenceClassification.from_pretrained(big_ckpt)
        phaseC_dir = f"{args.out}/phaseC_student_{idx}"
        histC, _ = run_kd(
            args,
            phaseC_dir,
            big_teacher,
            fresh_student,
            ds,
            coll,
            tok,
            f"phaseC_{idx}",
            load_best=False,
        )
        xsC, ysC = zip(*extract_acc(histC))
        big_teacher_acc = ysB[-1]
        plt.figure()
        plot_line(xsC, ysC, "Student phaseC", style="-.", color=col)
        plt.axhline(big_teacher_acc, color="red", ls=":", label="Big‑teacher acc")
        plt.xlabel("step"); plt.ylabel("accuracy"); plt.legend();
        plt.savefig(f"{args.out}/plots/phaseC_student_{idx}.png", dpi=180)
        plt.close()

        # combined record
        combined.append(
            {
                "xsA": xsA,
                "ysA": ysA,
                "teacher0": teacher_acc,
                "xsC": xsC,
                "ysC": ysC,
                "teacherC": big_teacher_acc,
                "col": col,
                "tag": f"ckpt{idx}",
            }
        )

    # ---------------- Combined plots -----------------------------
    for rec in combined:
        plt.figure()
        plot_line(rec["xsA"], rec["ysA"], f"Student phaseA ({rec['tag']})", color=rec["col"])
        plt.axhline(rec["teacher0"], color=rec["col"], ls="--", label="Teacher phase0")
        plot_line(rec["xsC"], rec["ysC"], f"Student phaseC ({rec['tag']})", style="-.", color=rec["col"])
        plt.axhline(rec["teacherC"], color=rec["col"], ls=":", label="Teacher phaseC")
        plt.xlabel("step"); plt.ylabel("accuracy"); plt.legend();
        plt.savefig(f"{args.out}/plots/combined_{rec['tag']}.png", dpi=180)
        plt.close()

    print(f"[DONE] 모든 실험이 완료되었습니다. 결과는 '{args.out}' 폴더에 저장되었습니다.")


if __name__ == "__main__":
    main()
