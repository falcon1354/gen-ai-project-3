"""Stage 4 – Direct Preference Optimization (no reward model)."""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import freeze_model, get_device, sequence_logprob, set_seed, slice_batch
from starter.batching import collate_pref
from starter.config import SEED
from starter.data import load_preference_splits
from starter.models import load_causal_lm, load_tokenizer


def dpo_loss(policy, ref, batch, beta, device):
    """
    L = -E[ log σ( β log(πθ(yw|x)/πref(yw|x)) - β log(πθ(yl|x)/πref(yl|x)) ) ]
    """
    chosen_ids = batch["chosen_ids"].to(device)
    chosen_mask = batch["chosen_mask"].to(device)
    rejected_ids = batch["rejected_ids"].to(device)
    rejected_mask = batch["rejected_mask"].to(device)
    prompt_lens = batch["prompt_lens"].to(device)

    logp_w = sequence_logprob(policy, chosen_ids, chosen_mask, prompt_lens)
    logp_l = sequence_logprob(policy, rejected_ids, rejected_mask, prompt_lens)

    with torch.no_grad():
        logp_ref_w = sequence_logprob(ref, chosen_ids, chosen_mask, prompt_lens)
        logp_ref_l = sequence_logprob(ref, rejected_ids, rejected_mask, prompt_lens)

    logits = beta * ((logp_w - logp_ref_w) - (logp_l - logp_ref_l))
    loss = -F.logsigmoid(logits).mean()
    acc = (logits > 0).float().mean()
    return loss, acc.detach()


@torch.no_grad()
def val_implicit_acc(policy, ref, loader, beta, device, micro_batch_size: int):
    policy.eval()
    correct, total = 0, 0
    for batch in loader:
        B = batch["chosen_ids"].shape[0]
        for start in range(0, B, micro_batch_size):
            mb = slice_batch(batch, start, start + micro_batch_size)
            _, acc = dpo_loss(policy, ref, mb, beta, device)
            bs = mb["chosen_ids"].shape[0]
            correct += acc.item() * bs
            total += bs
    policy.train()
    return correct / max(total, 1)


def _state_path(out_dir: Path) -> Path:
    return out_dir / "train_state.pt"


def save_checkpoint(out_dir: Path, policy, tokenizer, optimizer, scaler, step, log, elapsed):
    out_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "log": log,
            "elapsed": elapsed,
        },
        _state_path(out_dir),
    )
    print(f"  checkpoint saved at step {step} → {out_dir}")


def train(args):
    set_seed(SEED)
    device = get_device()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    assert args.batch_size % args.micro_batch_size == 0
    accum_steps = args.batch_size // args.micro_batch_size
    print(
        f"Effective batch={args.batch_size} "
        f"(micro_batch={args.micro_batch_size} × accum={accum_steps})"
    )

    tokenizer = load_tokenizer()
    print("Loading preference data…")
    pref_train, pref_val = load_preference_splits(tokenizer)

    # Fixed generator so resume can skip the same prefix of shuffled batches
    g = torch.Generator()
    g.manual_seed(SEED)

    def make_train_loader():
        return DataLoader(
            pref_train,
            batch_size=args.batch_size,
            shuffle=True,
            generator=g,
            collate_fn=partial(collate_pref, pad_id=tokenizer.pad_token_id),
            drop_last=True,
        )

    val_loader = DataLoader(
        pref_val,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_pref, pad_id=tokenizer.pad_token_id),
    )

    resume_state = _state_path(out)
    resuming = args.resume and resume_state.exists()

    if resuming:
        print(f"Resuming DPO from {out} …")
        policy = load_causal_lm(str(out), device=device)
        start_step = 0
    else:
        print(f"Loading policy + frozen ref from {args.sft_path}…")
        policy = load_causal_lm(args.sft_path, device=device)
        start_step = 0

    if hasattr(policy, "gradient_checkpointing_enable"):
        policy.gradient_checkpointing_enable()
        if hasattr(policy.config, "use_cache"):
            policy.config.use_cache = False

    ref = freeze_model(load_causal_lm(args.sft_path, device=device))
    assert all(not p.requires_grad for p in ref.parameters())
    print("Frozen check OK: π_ref requires_grad=False")

    policy.train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log = {"step": [], "train_loss": [], "val_acc": []}
    elapsed_prev = 0.0

    if resuming:
        state = torch.load(resume_state, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"])
        if "scaler" in state and use_amp:
            scaler.load_state_dict(state["scaler"])
        start_step = int(state["step"])
        log = state.get("log", log)
        elapsed_prev = float(state.get("elapsed", 0.0))
        print(f"Resumed at step {start_step}")
    else:
        # Correctness: first step loss ≈ log 2
        first_batch = slice_batch(
            next(iter(make_train_loader())), 0, min(2, args.micro_batch_size)
        )
        # reset generator after peek
        g.manual_seed(SEED)
        with torch.no_grad():
            loss0, _ = dpo_loss(policy, ref, first_batch, args.beta, device)
        print(f"DPO first-step loss (expect ≈ log 2 = 0.6931): {loss0.item():.4f}")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    steps_per_epoch = len(pref_train) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    print(f"Total steps this run: {total_steps} (resume from {start_step})")

    step = 0
    t0 = time.time()
    running, running_n = 0.0, 0

    for epoch in range(args.epochs):
        # re-seed each epoch start so epoch 0 order is reproducible on resume
        g.manual_seed(SEED + epoch)
        train_loader = make_train_loader()
        for batch in train_loader:
            step += 1
            if step <= start_step:
                continue

            optimizer.zero_grad(set_to_none=True)
            B = batch["chosen_ids"].shape[0]
            loss_accum = 0.0
            for start in range(0, B, args.micro_batch_size):
                mb = slice_batch(batch, start, start + args.micro_batch_size)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
                    loss, _ = dpo_loss(policy, ref, mb, args.beta, device)
                    loss = loss / accum_steps
                scaler.scale(loss).backward()
                loss_accum += float(loss.detach()) * accum_steps

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running += loss_accum
            running_n += 1

            if step % args.log_every == 0:
                train_l = running / max(running_n, 1)
                acc = val_implicit_acc(
                    policy, ref, val_loader, args.beta, device, args.micro_batch_size
                )
                print(f"step {step}: loss={train_l:.4f} val_implicit_acc={acc:.4f}")
                log["step"].append(step)
                log["train_loss"].append(train_l)
                log["val_acc"].append(acc)
                running, running_n = 0.0, 0
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if step % args.save_every == 0:
                elapsed = elapsed_prev + (time.time() - t0)
                save_checkpoint(
                    out, policy, tokenizer, optimizer, scaler, step, log, elapsed
                )
                with open(results_dir / "dpo_log.json", "w", encoding="utf-8") as f:
                    json.dump({**log, "last_step": step, "time_sec": elapsed}, f, indent=2)

    final_acc = val_implicit_acc(
        policy, ref, val_loader, args.beta, device, args.micro_batch_size
    )
    elapsed = elapsed_prev + (time.time() - t0)
    print(f"DPO done. val implicit acc={final_acc:.4f} time={elapsed/60:.1f} min")

    save_checkpoint(out, policy, tokenizer, optimizer, scaler, step, log, elapsed)

    with open(results_dir / "dpo_log.json", "w", encoding="utf-8") as f:
        json.dump(
            {**log, "final_val_acc": final_acc, "time_sec": elapsed, "last_step": step},
            f,
            indent=2,
        )

    if log["step"]:
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(log["step"], log["train_loss"], color="C0", label="train loss")
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss", color="C0")
        ax2 = ax1.twinx()
        ax2.plot(log["step"], log["val_acc"], color="C1", label="val acc")
        ax2.set_ylabel("implicit reward acc", color="C1")
        ax1.set_title("DPO training")
        fig.tight_layout()
        fig.savefig(results_dir / "fig_dpo.png", dpi=150)
        plt.close(fig)

    print(f"Saved checkpoint → {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_path", default="checkpoints/sft")
    p.add_argument("--out_dir", default="checkpoints/dpo")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-6)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--log_every", type=int, default=250)
    p.add_argument(
        "--save_every",
        type=int,
        default=250,
        help="Save policy + optimizer every N steps (for Colab reconnect)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue from checkpoints/dpo/train_state.pt if present",
    )
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
