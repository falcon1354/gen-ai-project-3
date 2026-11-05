"""Stage 3 – PPO with KL-constrained per-token rewards."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from common import (
    freeze_model,
    generate_summaries_chunked,
    get_device,
    logits_to_token_logprobs,
    masked_mean,
    set_seed,
)
from starter.batching import collate_prompts
from starter.config import MAX_SUMMARY_TOKENS, SEED
from starter.data import load_sft_splits
from starter.models import load_causal_lm, load_tokenizer
from train_reward import RewardModel

# reduce CUDA allocator fragmentation on free T4
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


class PolicyWithValue(nn.Module):
    """Causal LM policy + linear value head on last hidden states."""

    def __init__(self, sft_path: str, device=None):
        super().__init__()
        device = device or get_device()
        self.lm = load_causal_lm(sft_path, device=device)
        hidden = self.lm.config.hidden_size
        self.value_head = nn.Linear(hidden, 1)
        self.to(device)

    def forward(self, input_ids, attention_mask):
        # only last hidden state — avoid keeping every layer's activations
        out = self.lm.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        values = self.value_head(hidden).squeeze(-1)  # (B, L)
        logits = self.lm.lm_head(hidden)
        return logits, values

    def save_policy(self, path: str | Path):
        """Save the causal LM only (no value head), as required."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.lm.save_pretrained(path)


def logprobs_from_logits(logits, input_ids):
    return logits_to_token_logprobs(logits, input_ids)


def entropy_from_logits(logits, response_mask):
    """Mean entropy over response tokens only (no full-batch V softmax)."""
    shift = logits[:, :-1, :]
    flat = shift[response_mask.bool()]
    if flat.numel() == 0:
        return shift.new_zeros(())
    lp = F.log_softmax(flat.float(), dim=-1)
    ent = -(lp.exp() * lp).sum(dim=-1)
    return ent.mean()


def _micro_slices(B: int, micro: int):
    return [(s, min(s + micro, B)) for s in range(0, B, micro)]


@torch.no_grad()
def policy_logp_and_values(policy, sequences, full_mask, micro: int = 2):
    """Chunked policy forward → (logp_old, values_old) without huge logits."""
    B = sequences.shape[0]
    logps, values = [], []
    for s, e in _micro_slices(B, micro):
        logits, vals = policy(sequences[s:e], full_mask[s:e])
        logps.append(logprobs_from_logits(logits, sequences[s:e]))
        values.append(vals[:, :-1])
        del logits, vals
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(logps, 0), torch.cat(values, 0)


@torch.no_grad()
def ref_logprobs(ref, sequences, full_mask, micro: int = 2):
    B = sequences.shape[0]
    logps = []
    for s, e in _micro_slices(B, micro):
        out = ref(input_ids=sequences[s:e], attention_mask=full_mask[s:e])
        logps.append(logprobs_from_logits(out.logits, sequences[s:e]))
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(logps, 0)


@torch.no_grad()
def reward_scores(reward_model, sequences, full_mask, micro: int = 4):
    B = sequences.shape[0]
    scores = []
    for s, e in _micro_slices(B, micro):
        scores.append(reward_model(sequences[s:e], full_mask[s:e]))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(scores, 0)


@torch.no_grad()
def rollout(
    policy,
    ref,
    reward_model,
    tokenizer,
    prompts_batch,
    device,
    beta_kl,
    forward_micro: int = 2,
    gen_chunk: int = 4,
):
    """Sample summaries and store old logprobs / values / rewards (no grads)."""
    policy.eval()
    input_ids = prompts_batch["input_ids"].to(device)
    attention_mask = prompts_batch["attention_mask"].to(device)

    sequences, prompt_len, full_mask = generate_summaries_chunked(
        policy.lm,
        tokenizer,
        input_ids,
        attention_mask,
        max_new_tokens=MAX_SUMMARY_TOKENS,
        do_sample=True,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        chunk_size=gen_chunk,
    )

    B, full_L = sequences.shape
    positions = torch.arange(full_L - 1, device=device).unsqueeze(0).expand(B, -1)
    token_idx = positions + 1
    response_mask = ((token_idx >= prompt_len) & (full_mask[:, 1:] == 1)).long()

    logp_old, values_old = policy_logp_and_values(
        policy, sequences, full_mask, micro=forward_micro
    )
    logp_ref = ref_logprobs(ref, sequences, full_mask, micro=forward_micro)
    r_phi = reward_scores(reward_model, sequences, full_mask, micro=4)

    kl_token = logp_old - logp_ref
    rewards = -beta_kl * kl_token

    resp_lens = response_mask.sum(dim=1)
    for i in range(B):
        if resp_lens[i] == 0:
            continue
        last_token_idx = int((full_mask[i] == 1).nonzero(as_tuple=False)[-1].item())
        last_action_pos = last_token_idx - 1
        if last_action_pos >= 0:
            rewards[i, last_action_pos] = rewards[i, last_action_pos] + r_phi[i]

    returns = torch.zeros_like(rewards)
    for i in range(B):
        running = 0.0
        for t in range(full_L - 2, -1, -1):
            if response_mask[i, t] == 1:
                running = rewards[i, t].item() + running
                returns[i, t] = running

    advantages = returns - values_old
    adv_flat = advantages[response_mask.bool()]
    if adv_flat.numel() > 1:
        adv_mean = adv_flat.mean()
        adv_std = adv_flat.std().clamp_min(1e-8)
        advantages = (advantages - adv_mean) / adv_std

    kl_per_seq = (kl_token * response_mask.float()).sum(dim=-1)
    mean_kl = kl_per_seq.mean().item()
    mean_r = r_phi.mean().item()
    mean_len = resp_lens.float().mean().item()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "sequences": sequences,
        "full_mask": full_mask,
        "response_mask": response_mask,
        "logp_old": logp_old,
        "values_old": values_old,
        "returns": returns,
        "advantages": advantages,
        "mean_r": mean_r,
        "mean_kl": mean_kl,
        "mean_len": mean_len,
    }


def ppo_update(policy, optimizer, roll, eps, c_v, c_h, device, micro_batches=8):
    """One PPO epoch; process rollout in small micro-batches for T4."""
    sequences = roll["sequences"]
    full_mask = roll["full_mask"]
    response_mask = roll["response_mask"]
    logp_old = roll["logp_old"]
    returns = roll["returns"]
    advantages = roll["advantages"]

    B = sequences.shape[0]
    chunk = max(1, math.ceil(B / micro_batches))
    chunks = [list(range(i, min(i + chunk, B))) for i in range(0, B, chunk)]

    optimizer.zero_grad(set_to_none=True)
    total_pg = total_vf = total_ent = 0.0
    clip_hits = clip_total = 0
    ratio_first = []

    for chunk_idx in chunks:
        idx = torch.tensor(chunk_idx, device=device, dtype=torch.long)
        seq = sequences.index_select(0, idx)
        mask = full_mask.index_select(0, idx)
        resp = response_mask.index_select(0, idx)
        lp_old = logp_old.index_select(0, idx)
        ret = returns.index_select(0, idx)
        adv = advantages.index_select(0, idx)

        logits, values_all = policy(seq, mask)
        logp = logprobs_from_logits(logits, seq)
        values = values_all[:, :-1]

        ratio = (logp - lp_old).exp()
        ratio_first.append(ratio.detach()[resp.bool()])

        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * adv
        pg = torch.min(unclipped, clipped)
        L_ppo = masked_mean(pg, resp)
        L_v = masked_mean((values - ret) ** 2, resp)
        H = entropy_from_logits(logits, resp)

        loss = -L_ppo + c_v * L_v - c_h * H
        (loss * (len(chunk_idx) / B)).backward()

        total_pg += L_ppo.item() * len(chunk_idx)
        total_vf += L_v.item() * len(chunk_idx)
        total_ent += H.item() * len(chunk_idx)

        outside = ((ratio < 1.0 - eps) | (ratio > 1.0 + eps)) & resp.bool()
        clip_hits += outside.sum().item()
        clip_total += resp.sum().item()

        del logits, values_all, logp, values, ratio, unclipped, clipped, pg, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()

    all_ratios = torch.cat(ratio_first) if ratio_first else torch.tensor([1.0])
    return {
        "L_ppo": total_pg / B,
        "L_v": total_vf / B,
        "H": total_ent / B,
        "clip_frac": clip_hits / max(clip_total, 1),
        "mean_ratio": all_ratios.mean().item(),
    }


def train(args):
    set_seed(SEED)
    device = get_device()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    tokenizer = load_tokenizer()
    sft_train, _ = load_sft_splits(tokenizer)
    prompt_loader = DataLoader(
        sft_train,
        batch_size=args.prompts_per_rollout,
        sampler=RandomSampler(
            sft_train,
            replacement=True,
            num_samples=args.iterations * args.prompts_per_rollout,
        ),
        collate_fn=partial(collate_prompts, pad_id=tokenizer.pad_token_id),
    )

    print("Loading policy (+value), reference, reward…")
    policy = PolicyWithValue(args.sft_path, device=device)
    ref = freeze_model(load_causal_lm(args.sft_path, device=device, dtype=torch.float16))
    reward_model = RewardModel.load(args.sft_path, args.reward_path, device=device)
    freeze_model(reward_model)
    reward_model.half()

    assert all(not p.requires_grad for p in ref.parameters())
    assert all(not p.requires_grad for p in reward_model.parameters())
    print("Frozen check OK: π_ref and r_φ have requires_grad=False")

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    history = {
        "iter": [],
        "mean_r": [],
        "mean_kl": [],
        "mean_len": [],
        "clip_frac": [],
        "L_ppo": [],
        "L_v": [],
        "H": [],
    }

    t0 = time.time()
    prompt_iter = iter(prompt_loader)

    for it in range(args.start_iter, args.iterations + 1):
        prompts_batch = next(prompt_iter)
        with torch.no_grad():
            roll = rollout(
                policy,
                ref,
                reward_model,
                tokenizer,
                prompts_batch,
                device,
                args.beta_kl,
                forward_micro=args.forward_micro,
                gen_chunk=args.gen_chunk,
            )

        if it == 1:
            print(f"PPO iter 1 KL to ref (expect ~0): {roll['mean_kl']:.6f}")

        clip_fracs = []
        last_stats = None
        for k in range(args.ppo_epochs):
            stats = ppo_update(
                policy,
                optimizer,
                roll,
                eps=args.eps,
                c_v=args.c_v,
                c_h=args.c_h,
                device=device,
                micro_batches=args.micro_batches,
            )
            clip_fracs.append(stats["clip_frac"])
            last_stats = stats
            if it == 1 and k == 0:
                print(
                    f"PPO iter 1 epoch 1 mean ratio (expect ~1): {stats['mean_ratio']:.6f}, "
                    f"clip_frac (expect 0): {stats['clip_frac']:.6f}"
                )

        history["iter"].append(it)
        history["mean_r"].append(roll["mean_r"])
        history["mean_kl"].append(roll["mean_kl"])
        history["mean_len"].append(roll["mean_len"])
        history["clip_frac"].append(sum(clip_fracs) / len(clip_fracs))
        history["L_ppo"].append(last_stats["L_ppo"])
        history["L_v"].append(last_stats["L_v"])
        history["H"].append(last_stats["H"])

        print(
            f"iter {it:03d}/{args.iterations}: "
            f"rφ={roll['mean_r']:.3f} KL={roll['mean_kl']:.3f} "
            f"len={roll['mean_len']:.1f} clip={history['clip_frac'][-1]:.3f} "
            f"Lppo={last_stats['L_ppo']:.3f} Lv={last_stats['L_v']:.3f} H={last_stats['H']:.3f}"
        )

        del roll
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if it % 20 == 0:
            with open(results_dir / "ppo_log.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            # intermediate policy save (helps after Colab disconnect)
            policy.save_policy(args.out_dir)
            tokenizer.save_pretrained(args.out_dir)
            print(f"  checkpoint saved at iter {it} → {args.out_dir}")

    elapsed = time.time() - t0
    out = Path(args.out_dir)
    policy.save_policy(out)
    tokenizer.save_pretrained(out)

    history["time_sec"] = elapsed
    with open(results_dir / "ppo_log.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(history["iter"], history["mean_r"])
    axes[0, 0].set_title("mean rφ")
    axes[0, 1].plot(history["iter"], history["mean_kl"])
    axes[0, 1].set_title("mean KL")
    axes[1, 0].plot(history["iter"], history["mean_len"])
    axes[1, 0].set_title("mean summary length")
    axes[1, 1].plot(history["iter"], history["clip_frac"])
    axes[1, 1].set_title("clip fraction")
    for ax in axes.flat:
        ax.set_xlabel("iteration")
    fig.suptitle("PPO training curves")
    fig.tight_layout()
    fig.savefig(results_dir / "fig_ppo.png", dpi=150)
    plt.close(fig)

    print(f"PPO done in {elapsed/60:.1f} min. Saved → {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_path", default="checkpoints/sft")
    p.add_argument("--reward_path", default="checkpoints/reward")
    p.add_argument("--out_dir", default="checkpoints/ppo")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--eps", type=float, default=0.2)
    p.add_argument("--beta_kl", type=float, default=0.05)
    p.add_argument("--c_v", type=float, default=0.1)
    p.add_argument("--c_h", type=float, default=0.01)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--prompts_per_rollout", type=int, default=16)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--start_iter", type=int, default=1)
    p.add_argument("--micro_batches", type=int, default=8)
    p.add_argument("--forward_micro", type=int, default=2)
    p.add_argument("--gen_chunk", type=int, default=4)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
