"""Stage 2 – Reward model with Bradley–Terry pairwise loss."""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import freeze_model, get_device, set_seed
from starter.batching import collate_pref
from starter.config import SEED
from starter.data import load_preference_splits
from starter.models import load_causal_lm, load_tokenizer


class RewardModel(nn.Module):
    """Transformer body from SFT + linear head on last non-pad hidden state."""

    def __init__(self, sft_path: str, device=None):
        super().__init__()
        device = device or get_device()
        base = load_causal_lm(sft_path, device=device)
        # drop LM head; keep transformer body
        self.backbone = base.model
        hidden = base.config.hidden_size
        self.v_head = nn.Linear(hidden, 1)
        # discard unused lm_head weights from memory
        del base.lm_head
        self.to(device)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
        )
        hidden = out.last_hidden_state  # (B, L, H)
        # last REAL token via attention_mask (pad == eos for this tokenizer)
        lengths = attention_mask.sum(dim=1) - 1  # index of last real token
        B = input_ids.shape[0]
        gather_idx = lengths.view(B, 1, 1).expand(-1, 1, hidden.shape[-1])
        last = hidden.gather(1, gather_idx).squeeze(1)  # (B, H)
        return self.v_head(last).squeeze(-1)  # (B,)

    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "backbone": self.backbone.state_dict(),
                "v_head": self.v_head.state_dict(),
                "hidden_size": self.v_head.in_features,
            },
            path / "reward.pt",
        )

    @classmethod
    def load(cls, sft_path: str, reward_dir: str, device=None):
        device = device or get_device()
        rm = cls(sft_path, device=device)
        state = torch.load(Path(reward_dir) / "reward.pt", map_location=device)
        rm.backbone.load_state_dict(state["backbone"])
        rm.v_head.load_state_dict(state["v_head"])
        return rm.to(device)


def bradley_terry_loss(r_w, r_l):
    """L = -E[log σ(r_w - r_l)]"""
    return -F.logsigmoid(r_w - r_l).mean()


@torch.no_grad()
def pairwise_accuracy(rm, loader, device):
    rm.eval()
    correct, total = 0, 0
    for batch in loader:
        chosen_ids = batch["chosen_ids"].to(device)
        chosen_mask = batch["chosen_mask"].to(device)
        rejected_ids = batch["rejected_ids"].to(device)
        rejected_mask = batch["rejected_mask"].to(device)
        r_w = rm(chosen_ids, chosen_mask)
        r_l = rm(rejected_ids, rejected_mask)
        correct += (r_w > r_l).sum().item()
        total += r_w.shape[0]
    rm.train()
    return correct / max(total, 1)


def train(args):
    set_seed(SEED)
    device = get_device()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    tokenizer = load_tokenizer()
    print("Loading preference data…")
    pref_train, pref_val = load_preference_splits(tokenizer)
    print(f"Pref train/val: {len(pref_train)} / {len(pref_val)}")

    train_loader = DataLoader(
        pref_train,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=partial(collate_pref, pad_id=tokenizer.pad_token_id),
        drop_last=True,
    )
    val_loader = DataLoader(
        pref_val,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_pref, pad_id=tokenizer.pad_token_id),
    )

    print(f"Building reward model from {args.sft_path}…")
    rm = RewardModel(args.sft_path, device=device)
    rm.train()
    optimizer = torch.optim.AdamW(rm.parameters(), lr=args.lr)

    log = {"step": [], "train_loss": [], "val_acc": []}
    step = 0
    t0 = time.time()
    running, running_n = 0.0, 0

    for epoch in range(args.epochs):
        for batch in train_loader:
            chosen_ids = batch["chosen_ids"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            r_w = rm(chosen_ids, chosen_mask)
            r_l = rm(rejected_ids, rejected_mask)
            loss = bradley_terry_loss(r_w, r_l)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            optimizer.step()

            step += 1
            running += loss.item()
            running_n += 1

            if step % args.log_every == 0:
                train_l = running / running_n
                acc = pairwise_accuracy(rm, val_loader, device)
                print(f"step {step}: loss={train_l:.4f} val_pairwise_acc={acc:.4f}")
                log["step"].append(step)
                log["train_loss"].append(train_l)
                log["val_acc"].append(acc)
                running, running_n = 0.0, 0

    final_acc = pairwise_accuracy(rm, val_loader, device)
    elapsed = time.time() - t0
    print(f"Reward done. val pairwise acc={final_acc:.4f} time={elapsed/60:.1f} min")

    out = Path(args.out_dir)
    rm.save(out)
    # also keep a pointer to the SFT backbone path for loading later
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump({"sft_path": args.sft_path, "val_pairwise_acc": final_acc}, f, indent=2)

    with open(results_dir / "reward_log.json", "w", encoding="utf-8") as f:
        json.dump(
            {**log, "final_val_acc": final_acc, "time_sec": elapsed}, f, indent=2
        )

    if log["step"]:
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(log["step"], log["train_loss"], color="C0", label="train loss")
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss", color="C0")
        ax2 = ax1.twinx()
        ax2.plot(log["step"], log["val_acc"], color="C1", label="val acc")
        ax2.set_ylabel("pairwise acc", color="C1")
        ax1.set_title("Reward model training")
        fig.tight_layout()
        fig.savefig(results_dir / "fig_reward.png", dpi=150)
        plt.close(fig)

    print(f"Saved checkpoint → {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_path", default="checkpoints/sft")
    p.add_argument("--out_dir", default="checkpoints/reward")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--log_every", type=int, default=250)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
