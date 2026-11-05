"""Gradio front end: paste a Reddit post → SFT / PPO / DPO summaries + rφ scores."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gradio as gr
import torch

# allow importing from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import decode_summary, freeze_model, generate_summaries, get_device
from starter.batching import collate_prompts
from starter.config import MAX_SUMMARY_TOKENS
from starter.data import clean_post
from starter.models import load_causal_lm, load_tokenizer
from train_reward import RewardModel


class SummarizerApp:
    def __init__(self, sft_path, ppo_path, dpo_path, reward_path):
        self.device = get_device()
        self.tok = load_tokenizer()
        print("Loading checkpoints…")
        self.sft = freeze_model(load_causal_lm(sft_path, device=self.device))
        self.ppo = freeze_model(load_causal_lm(ppo_path, device=self.device))
        self.dpo = freeze_model(load_causal_lm(dpo_path, device=self.device))
        self.rm = freeze_model(RewardModel.load(sft_path, reward_path, device=self.device))
        self.policies = {"SFT": self.sft, "PPO": self.ppo, "DPO": self.dpo}

    @torch.no_grad()
    def summarize_one(self, model, prompt_text: str):
        # clean_post strips any existing TL;DR and appends a single marker
        prompt = clean_post(prompt_text.strip())
        ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        batch = collate_prompts([{"prompt_ids": ids}], pad_id=self.tok.pad_token_id)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        sequences, prompt_len, gen_mask = generate_summaries(
            model,
            self.tok,
            input_ids,
            attention_mask,
            max_new_tokens=MAX_SUMMARY_TOKENS,
            do_sample=False,
        )
        text = decode_summary(self.tok, sequences, prompt_len, gen_mask)[0]
        score = float(self.rm(sequences, gen_mask)[0].item())
        return text, score

    def run(self, post: str):
        if not post or not post.strip():
            return "Paste a post.", "", "", "", "", ""
        results = {}
        for name, model in self.policies.items():
            text, score = self.summarize_one(model, post.strip())
            results[name] = (text, f"{score:.3f}")
        return (
            results["SFT"][0],
            results["SFT"][1],
            results["PPO"][0],
            results["PPO"][1],
            results["DPO"][0],
            results["DPO"][1],
        )


def build_ui(app: SummarizerApp):
    with gr.Blocks(title="TL;DR – SFT vs PPO vs DPO") as demo:
        gr.Markdown(
            "# Learning to Summarize from Human Feedback\n"
            "Paste a Reddit-style post. The three policies (SFT, PPO, DPO) each "
            "write a one-sentence TL;DR and are scored by the trained reward model."
        )
        inp = gr.Textbox(
            lines=12,
            label="Reddit post",
            placeholder="SUBREDDIT: r/…\nTITLE: …\nPOST: …\n",
        )
        btn = gr.Button("Summarize", variant="primary")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### SFT")
                sft_sum = gr.Textbox(label="Summary", lines=4)
                sft_r = gr.Textbox(label="Reward rφ")
            with gr.Column():
                gr.Markdown("### PPO")
                ppo_sum = gr.Textbox(label="Summary", lines=4)
                ppo_r = gr.Textbox(label="Reward rφ")
            with gr.Column():
                gr.Markdown("### DPO")
                dpo_sum = gr.Textbox(label="Summary", lines=4)
                dpo_r = gr.Textbox(label="Reward rφ")
        btn.click(
            app.run,
            inputs=[inp],
            outputs=[sft_sum, sft_r, ppo_sum, ppo_r, dpo_sum, dpo_r],
        )
    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_path", default="checkpoints/sft")
    p.add_argument("--ppo_path", default="checkpoints/ppo")
    p.add_argument("--dpo_path", default="checkpoints/dpo")
    p.add_argument("--reward_path", default="checkpoints/reward")
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    app = SummarizerApp(args.sft_path, args.ppo_path, args.dpo_path, args.reward_path)
    demo = build_ui(app)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
