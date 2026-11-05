"""Stage 1 – Supervised fine-tuning on human TL;DR summaries."""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from common import (
    decode_summary,
    generate_summaries,
    get_device,
    masked_mean,
    set_seed,
    token_logprobs,
)
from starter.batching import collate_prompts, collate_sft
from starter.config import MAX_SUMMARY_TOKENS, MODEL_NAME, SEED
from starter.data import load_eval_posts, load_sft_splits
from starter.models import load_causal_lm, load_tokenizer


def sft_loss(model, batch):
    token_lp, summary_mask = token_logprobs(
        model, batch["input_ids"], batch["attention_mask"], batch["prompt_lens"]
    )
    # mean negative log-likelihood over summary tokens only
    return -masked_mean(token_lp, summary_mask)


@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        loss = sft_loss(model, batch)
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def verify_summary_mask(model, tokenizer, batch, device):
    """Correctness check: decoding mask==1 tokens == summary + EOS."""
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    _, summary_mask = token_logprobs(
        model, batch["input_ids"], batch["attention_mask"], batch["prompt_lens"]
    )
    # mask is (B, L-1) aligned with tokens at positions 1..L-1
    ids = batch["input_ids"]
    print("\n=== SFT mask check (first example) ===")
    mask0 = summary_mask[0].bool()
    # tokens corresponding to mask positions: ids[0, 1:]
    selected = ids[0, 1:][mask0]
    decoded = tokenizer.decode(selected)
    print("Decoded where mask==1:", repr(decoded))
    print("(Expect: summary text followed by EOS token string)")


@torch.no_grad()
def sample_eval_summaries(model, tokenizer, eval_posts, device, n=5):
    model.eval()
    subset = eval_posts[:n]
    loader = DataLoader(
        subset,
        batch_size=n,
        collate_fn=partial(collate_prompts, pad_id=tokenizer.pad_token_id),
    )
    batch = next(iter(loader))
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    sequences, prompt_len, gen_mask = generate_summaries(
        model,
        tokenizer,
        input_ids,
        attention_mask,
        max_new_tokens=MAX_SUMMARY_TOKENS,
        do_sample=False,
    )
    texts = decode_summary(tokenizer, sequences, prompt_len, gen_mask)
    rows = []
    for i, t in enumerate(texts):
        rows.append(
            {
                "prompt": subset[i]["prompt"],
                "reference": subset[i]["summary"],
                "generated": t,
            }
        )
    return rows


def train(args):
    set_seed(SEED)
    device = get_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    tokenizer = load_tokenizer()
    print("Loading SFT data…")
    sft_train, sft_val = load_sft_splits(tokenizer)

    # Task 1.2 stats
    prompt_lens = [len(x["prompt_ids"]) for x in sft_train]
    summary_lens = [len(x["summary_ids"]) for x in sft_train]
    data_stats = {
        "sft_train": len(sft_train),
        "sft_val": len(sft_val),
        "mean_prompt_len": sum(prompt_lens) / len(prompt_lens),
        "max_prompt_len": max(prompt_lens),
        "mean_summary_len": sum(summary_lens) / len(summary_lens),
        "max_summary_len": max(summary_lens),
    }
    print("Data stats:", data_stats)
    with open(results_dir / "table1_data.json", "w", encoding="utf-8") as f:
        json.dump(data_stats, f, indent=2)

    train_loader = DataLoader(
        sft_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_sft, pad_id=tokenizer.pad_token_id),
        drop_last=True,
    )
    val_loader = DataLoader(
        sft_val,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_sft, pad_id=tokenizer.pad_token_id),
    )

    print(f"Loading base model {MODEL_NAME}…")
    model = load_causal_lm(MODEL_NAME, device=device)
    model.train()

    # mask check on one batch
    check_batch = next(iter(train_loader))
    verify_summary_mask(model, tokenizer, check_batch, device)

    # before-SFT samples
    eval_posts = load_eval_posts(tokenizer)
    before = sample_eval_summaries(model, tokenizer, eval_posts, device, n=5)
    with open(results_dir / "sft_samples_before.json", "w", encoding="utf-8") as f:
        json.dump(before, f, indent=2, ensure_ascii=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    log = {"step": [], "train_loss": [], "val_loss": []}
    step = 0
    t0 = time.time()
    running = 0.0
    running_n = 0

    for epoch in range(args.epochs):
        for batch in train_loader:
            batch = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            loss = sft_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            running += loss.item()
            running_n += 1

            if step % args.log_every == 0:
                train_l = running / running_n
                val_l = eval_loss(model, val_loader, device)
                print(f"step {step}: train_loss={train_l:.4f} val_loss={val_l:.4f}")
                log["step"].append(step)
                log["train_loss"].append(train_l)
                log["val_loss"].append(val_l)
                running, running_n = 0.0, 0

    # final val
    final_val = eval_loss(model, val_loader, device)
    elapsed = time.time() - t0
    print(f"SFT done. final val_loss={final_val:.4f} time={elapsed/60:.1f} min")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    after = sample_eval_summaries(model, tokenizer, eval_posts, device, n=5)
    with open(results_dir / "sft_samples_after.json", "w", encoding="utf-8") as f:
        json.dump(after, f, indent=2, ensure_ascii=False)

    with open(results_dir / "sft_log.json", "w", encoding="utf-8") as f:
        json.dump({**log, "final_val_loss": final_val, "time_sec": elapsed}, f, indent=2)

    if log["step"]:
        plt.figure(figsize=(7, 4))
        plt.plot(log["step"], log["train_loss"], label="train")
        plt.plot(log["step"], log["val_loss"], label="val")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("SFT training / validation loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(results_dir / "fig_sft_loss.png", dpi=150)
        plt.close()

    print(f"Saved checkpoint → {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="checkpoints/sft")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--log_every", type=int, default=250)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
