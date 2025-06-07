import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metric_accuracy(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": (logits.argmax(-1) == labels).mean()}


def soft_ce(p_logits, q_logits, T: float) -> torch.Tensor:
    p = F.log_softmax(p_logits / T, dim=-1)
    q = F.softmax(q_logits / T, dim=-1)
    return F.kl_div(p, q, reduction="batchmean") * (T ** 2)


class KDTrainer(Trainer):
    """HuggingFace Trainer with knowledge distillation loss."""

    def __init__(self, teacher_model, alpha=0.5, temperature=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.alpha = alpha
        self.T = temperature

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        s_out = model(**inputs)
        with torch.no_grad():
            t_out = self.teacher(**inputs)
        ce = F.cross_entropy(s_out.logits, labels)
        kd = soft_ce(s_out.logits, t_out.logits, self.T)
        loss = self.alpha * ce + (1 - self.alpha) * kd
        return (loss, s_out) if return_outputs else loss


def prepare_data(tokenizer, cache_dir):
    ds = load_dataset("glue", "sst2", cache_dir=cache_dir)

    def tok(batch):
        return tokenizer(batch["sentence"], truncation=True)

    ds_enc = ds.map(tok, batched=True)
    keep = ["input_ids", "attention_mask", "label"]
    ds_enc = ds_enc.remove_columns([c for c in ds_enc["train"].column_names if c not in keep])
    collator = DataCollatorWithPadding(tokenizer)
    return ds_enc, collator


def run_kd(teacher, student, ds, coll, tok, out_dir, *, alpha, temp, args, tag):
    targs = TrainingArguments(
        output_dir=out_dir,
        evaluation_strategy="steps",
        save_strategy="no",
        eval_steps=args.eval_steps,
        logging_steps=args.eval_steps,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        seed=args.seed,
    )
    trainer = KDTrainer(
        teacher_model=teacher,
        alpha=alpha,
        temperature=temp,
        model=student,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tok,
        data_collator=coll,
        compute_metrics=metric_accuracy,
    )
    trainer.train()
    trainer.save_model(out_dir)
    hist = trainer.state.log_history
    with open(Path(out_dir) / f"log_{tag}.json", "w") as f:
        json.dump(hist, f, indent=2)
    return hist


def extract_acc(hist):
    return [(h["step"], h["eval_accuracy"]) for h in hist if "eval_accuracy" in h]


def main(args):
    set_seed(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.plot_dir).mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.teacher_model)
    ds, coll = prepare_data(tok, args.out_dir)

    teacher_model = AutoModelForSequenceClassification.from_pretrained(args.teacher_model)
    teacher_eval = Trainer(
        model=teacher_model,
        eval_dataset=ds["validation"],
        tokenizer=tok,
        data_collator=coll,
        compute_metrics=metric_accuracy,
    )
    teacher_acc = teacher_eval.evaluate()["eval_accuracy"]

    # ---------------- Phase A: grid search -----------------------
    best = {"acc": 0}
    phaseA_hist = None
    for alpha in args.alphas:
        for temp in args.temps:
            student = AutoModelForSequenceClassification.from_pretrained(
                args.student_model, num_labels=2
            )
            student.init_weights()
            tag = f"phaseA_a{alpha}_t{temp}"
            out_sub = Path(args.out_dir) / tag
            hist = run_kd(
                teacher_model,
                student,
                ds,
                coll,
                tok,
                out_sub,
                alpha=alpha,
                temp=temp,
                args=args,
                tag=tag,
            )
            acc = extract_acc(hist)[-1][1]
            if acc > best["acc"]:
                best.update({"acc": acc, "alpha": alpha, "temp": temp, "path": out_sub})
                phaseA_hist = hist
    assert phaseA_hist is not None
    phaseA_model = AutoModelForSequenceClassification.from_pretrained(best["path"])

    # ---------------- Phase B -----------------------------------
    tagB = "phaseB"
    histB = run_kd(
        teacher=phaseA_model,
        student=AutoModelForSequenceClassification.from_pretrained(args.teacher_model),
        ds=ds,
        coll=coll,
        tok=tok,
        out_dir=Path(args.out_dir) / tagB,
        alpha=best["alpha"],
        temp=best["temp"],
        args=args,
        tag=tagB,
    )
    phaseB_model = AutoModelForSequenceClassification.from_pretrained(Path(args.out_dir) / tagB)

    # ---------------- Phase C -----------------------------------
    tagC = "phaseC"
    histC = run_kd(
        teacher=phaseB_model,
        student=AutoModelForSequenceClassification.from_pretrained(best["path"]),
        ds=ds,
        coll=coll,
        tok=tok,
        out_dir=Path(args.out_dir) / tagC,
        alpha=best["alpha"],
        temp=best["temp"],
        args=args,
        tag=tagC,
    )
    phaseC_model = AutoModelForSequenceClassification.from_pretrained(Path(args.out_dir) / tagC)

    # ---------------- Phase D -----------------------------------
    tagD = "phaseD"
    histD = run_kd(
        teacher=phaseC_model,
        student=AutoModelForSequenceClassification.from_pretrained(Path(args.out_dir) / tagB),
        ds=ds,
        coll=coll,
        tok=tok,
        out_dir=Path(args.out_dir) / tagD,
        alpha=best["alpha"],
        temp=best["temp"],
        args=args,
        tag=tagD,
    )

    # ---------------- Plotting ----------------------------------
    plt.figure()
    plt.axhline(teacher_acc, color="black", linestyle="--", label="Teacher phase0")
    for tag, hist in zip([
        f"phaseA (α={best['alpha']},T={best['temp']})",
        "phaseB",
        "phaseC",
        "phaseD",
    ], [phaseA_hist, histB, histC, histD]):
        xs, ys = zip(*extract_acc(hist))
        plt.plot(xs, ys, label=tag)
    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(Path(args.plot_dir) / "phase_curves.png", dpi=180)

    print("[DONE] Training complete. Plot saved.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="/local_datasets/parity/bert2")
    p.add_argument("--plot_dir", type=str, default="/data/dy0718/repos/bert2")
    p.add_argument("--teacher_model", type=str, default="doyoungkim/bert-base-uncased-finetuned-sst2")
    p.add_argument("--student_model", type=str, default="prajjwal1/bert-mini")
    p.add_argument("--train_bs", type=int, default=64)
    p.add_argument("--eval_bs", type=int, default=256)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alphas", nargs="*", type=float, default=[0.1, 0.5, 0.9])
    p.add_argument("--temps", nargs="*", type=float, default=[1.0, 2.0, 4.0])
    args = p.parse_args()
    main(args)
