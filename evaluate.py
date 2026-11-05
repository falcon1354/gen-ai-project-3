"""Task 6 – Compare SFT, PPO and DPO on 300 held-out posts."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import torch
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader

from common import (
    decode_summary,
    freeze_model,
    generate_summaries,
    get_device,
    set_seed,
    token_logprobs,
)
from starter.batching import collate_prompts
from starter.config import MAX_SUMMARY_TOKENS, SEED
from starter.data import load_eval_posts
from starter.models import load_causal_lm, load_tokenizer
from train_reward import RewardModel


@torch.no_grad()
def generate_all(model, tokenizer, posts, device, batch_size=8):
    model.eval()
    loader = DataLoader(
        posts,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_prompts, pad_id=tokenizer.pad_token_id),
    )
    summaries = []
    all_ids = []
    all_masks = []
    prompt_lens_list = []
    for batch in loader:
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
        summaries.extend(texts)
        all_ids.append(sequences.cpu())
        all_masks.append(gen_mask.cpu())
        prompt_lens_list.append(torch.full((sequences.shape[0],), prompt_len))
    return summaries, all_ids, all_masks, prompt_lens_list


@torch.no_grad()
def score_batch(rm, sequences, masks, device):
    sequences = sequences.to(device)
    masks = masks.to(device)
    return rm(sequences, masks).cpu()


@torch.no_grad()
def kl_to_ref(policy, ref, sequences, masks, prompt_len, device):
    """Mean over posts of sum_t (log πθ - log πref) on the method's own summary."""
    sequences = sequences.to(device)
    masks = masks.to(device)
    B = sequences.shape[0]
    prompt_lens = torch.full((B,), prompt_len, device=device)
    lp_pol, summary_mask = token_logprobs(policy, sequences, masks, prompt_lens)
    lp_ref, _ = token_logprobs(ref, sequences, masks, prompt_lens)
    kl = ((lp_pol - lp_ref) * summary_mask.float()).sum(dim=-1)
    lengths = summary_mask.sum(dim=-1).float()
    return kl.cpu(), lengths.cpu()


def evaluate(args):
    set_seed(SEED)
    device = get_device()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    tokenizer = load_tokenizer()
    posts = load_eval_posts(tokenizer)
    print(f"Eval posts: {len(posts)}")

    print("Loading models…")
    sft = freeze_model(load_causal_lm(args.sft_path, device=device))
    ppo = freeze_model(load_causal_lm(args.ppo_path, device=device))
    dpo = freeze_model(load_causal_lm(args.dpo_path, device=device))
    rm = freeze_model(RewardModel.load(args.sft_path, args.reward_path, device=device))
    # KL reference = SFT
    ref = sft

    policies = {"SFT": sft, "PPO": ppo, "DPO": dpo}
    generations = {}

    for name, model in policies.items():
        print(f"Generating with {name}…")
        texts, id_chunks, mask_chunks, pl_chunks = generate_all(
            model, tokenizer, posts, device, batch_size=args.batch_size
        )
        generations[name] = {
            "texts": texts,
            "id_chunks": id_chunks,
            "mask_chunks": mask_chunks,
            "pl_chunks": pl_chunks,
        }

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    metrics = {}
    sft_rewards = None

    rows_out = []

    for name in ["SFT", "PPO", "DPO"]:
        gen = generations[name]
        rewards_list = []
        kl_list = []
        len_list = []
        rouge_list = []

        offset = 0
        for ids, masks, pls in zip(
            gen["id_chunks"], gen["mask_chunks"], gen["pl_chunks"]
        ):
            r = score_batch(rm, ids, masks, device)
            rewards_list.append(r)
            prompt_len = int(pls[0].item())
            kl, lengths = kl_to_ref(policies[name], ref, ids, masks, prompt_len, device)
            # For SFT, KL to itself should be ~0
            kl_list.append(kl)
            len_list.append(lengths)
            offset += ids.shape[0]

        rewards = torch.cat(rewards_list)
        kls = torch.cat(kl_list)
        lengths = torch.cat(len_list)

        for i, text in enumerate(gen["texts"]):
            ref_sum = posts[i]["summary"]
            rouge_list.append(scorer.score(ref_sum, text)["rougeL"].fmeasure)

        mean_r = rewards.mean().item()
        mean_kl = kls.mean().item()
        mean_len = lengths.mean().item()
        mean_rouge = sum(rouge_list) / len(rouge_list)

        if name == "SFT":
            sft_rewards = rewards
            win_rate = None
        else:
            win_rate = (rewards > sft_rewards).float().mean().item()

        metrics[name] = {
            "mean_r": mean_r,
            "win_rate_vs_sft": win_rate,
            "kl_to_sft": mean_kl,
            "mean_length": mean_len,
            "rougeL": mean_rouge,
        }
        print(name, metrics[name])

    # Save jsonl of all summaries
    out_path = results_dir / "eval_summaries.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for i, post in enumerate(posts):
            row = {
                "prompt": post["prompt"],
                "reference": post["summary"],
                "SFT": generations["SFT"]["texts"][i],
                "PPO": generations["PPO"]["texts"][i],
                "DPO": generations["DPO"]["texts"][i],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(results_dir / "table3_comparison.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Qualitative samples (first 10)
    samples_md = ["# Qualitative samples (10 evaluation posts)\n"]
    for i in range(min(10, len(posts))):
        samples_md.append(f"## Post {i+1}\n")
        samples_md.append(f"**Prompt (tail):** …{posts[i]['prompt'][-400:]}\n")
        samples_md.append(f"**Human:** {posts[i]['summary']}\n")
        samples_md.append(f"**SFT:** {generations['SFT']['texts'][i]}\n")
        samples_md.append(f"**PPO:** {generations['PPO']['texts'][i]}\n")
        samples_md.append(f"**DPO:** {generations['DPO']['texts'][i]}\n")
    (results_dir / "samples.md").write_text("\n".join(samples_md), encoding="utf-8")

    print(f"Wrote {out_path}")
    return metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_path", default="checkpoints/sft")
    p.add_argument("--ppo_path", default="checkpoints/ppo")
    p.add_argument("--dpo_path", default="checkpoints/dpo")
    p.add_argument("--reward_path", default="checkpoints/reward")
    p.add_argument("--batch_size", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
