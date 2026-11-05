# Learning to Summarize from Human Feedback

Assignment 3 – Generative AI (Fall 2026).  
Pipeline: **SFT → Reward model → PPO** and a parallel **DPO** branch, all on `HuggingFaceTB/SmolLM2-135M`, implemented in PyTorch (no TRL / PEFT / Trainer).

Paper: [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325)

## Repo layout

```
starter/           # given data / model loading (unchanged intent)
common.py          # token_logprobs, summary mask, generation helpers
train_sft.py       # Stage 1
train_reward.py    # Stage 2
train_ppo.py       # Stage 3
train_dpo.py       # Stage 4
evaluate.py        # Task 6 comparison on 300 posts
app/app.py         # Gradio front end
results/           # logs, tables, figures, analysis, human eval
notebooks/colab_train.ipynb
scripts/
```

## Environment

```bash
pip install -r requirements.txt
```

Use **Google Colab free T4** (or Kaggle). CPU will work for debugging tiny runs only.

## How to train on Colab (your part)

1. Push this repo to GitHub **or** zip-upload the folder to Google Drive.
2. Open `notebooks/colab_train.ipynb` in Colab.
3. **Runtime → GPU (T4)**.
4. Edit `PROJECT = '/content/drive/MyDrive/GenAI_Project3'` to your path.
5. Run cells **in order**. Do not skip stages; each needs the previous checkpoint.

| Cell / command | Approx. T4 time | Output |
|----------------|-----------------|--------|
| `python -m starter.check_starter` | 5–15 min (download) | subset sizes |
| `python scripts/check_token_logprobs.py` | minutes | mask decode check |
| `python train_sft.py` | ~1–2 h | `checkpoints/sft/` |
| `python train_reward.py` | ~1–2 h | `checkpoints/reward/` |
| `python train_ppo.py` | ~2–4 h | `checkpoints/ppo/` |
| `python train_dpo.py` | ~1–2 h | `checkpoints/dpo/` |
| `python evaluate.py` | ~20–40 min | `results/eval_summaries.jsonl`, Table 3 |
| `python app/app.py --share` | — | Gradio URL + screenshot for README |

**Tip:** Colab can disconnect. Keep `checkpoints/` on Drive so you can resume from the next stage. Do **not** commit checkpoints to GitHub — upload them to Hugging Face or Drive and paste the link below.

### Checkpoint links (fill after upload)

- SFT: _TODO_
- Reward: _TODO_
- PPO: _TODO_
- DPO: _TODO_

## Correctness checks (paste outputs into this README after runs)

- **SFT mask:** decoding positions where `summary_mask == 1` equals summary + EOS.
- **Frozen models:** every parameter of `π_ref` and `r_φ` has `requires_grad=False` in PPO/DPO (printed by the scripts).
- **PPO, first iteration:** KL to reference ≈ 0.
- **PPO, first of K epochs:** mean ratio `r_t(θ) ≈ 1`, clip fraction ≈ 0.
- **DPO, first step:** loss ≈ `log 2 ≈ 0.6931`.
- **Padding:** prompts for `generate` are left-padded; reward model uses `attention_mask` for the last real token (never `input_ids != pad_id`, because pad == EOS here).

## Front end

```bash
python app/app.py --sft_path checkpoints/sft --ppo_path checkpoints/ppo \
  --dpo_path checkpoints/dpo --reward_path checkpoints/reward
```

Paste a Reddit-style post; see SFT / PPO / DPO summaries side by side with reward-model scores.  
Screenshot: `results/frontend_screenshot.png` _(add after you run it)_.

## Human evaluation

```bash
python scripts/prepare_human_eval.py
```

Both members independently pick the best of A/B/C in `results/human_eval_blind.csv` (order is shuffled). Map back with `results/human_eval_key.json` and fill `results/human_eval.csv` for Table 4.

## Allowed / forbidden libraries

**Allowed:** `torch`, `transformers` (`AutoTokenizer`, `AutoModelForCausalLM`, `generate`), `datasets`, `rouge-score`, `numpy`, `matplotlib`, starter code, Gradio.

**Forbidden:** `trl`, `peft`, `trlx`, OpenRLHF, DeepSpeed-Chat, Hugging Face `Trainer`, `AutoModelForSequenceClassification`.

## Hyperparameters (from the manual)

| Stage | Key settings |
|-------|----------------|
| SFT | 1 epoch, AdamW 5e-5, bs 16, grad clip 1.0 |
| Reward | 1 epoch, AdamW 1e-5, bs 8 pairs |
| PPO | 200 iters × 16 prompts, K=4, ε=0.2, β_KL=0.05, c_V=0.1, c_H=0.01, lr 1e-5 |
| DPO | 1 epoch, β=0.1, AdamW 2e-6, bs 8 pairs |

## Communication deliverables (non-code)

- Medium blog (800–1500 words) + LinkedIn post linking blog + repo.
- Fill `results/analysis.md` (Task 7).
- Commits from **both** team members on GitHub.

## Team

- Member 1: _(name / roll)_
- Member 2: _(name / roll)_
