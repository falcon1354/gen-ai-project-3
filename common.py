"""Shared helpers used by every training / evaluation stage."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def summary_token_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
) -> torch.Tensor:
    """Mask of shape (B, L-1) that is 1 exactly on summary token positions.

    Position j in the (L-1) mask corresponds to token input_ids[:, j+1]
    (the token whose log-prob comes from logits at index j).
    """
    B, L = input_ids.shape
    device = input_ids.device
    # positions 0..L-2 in the mask align with tokens at indices 1..L-1
    positions = torch.arange(L - 1, device=device).unsqueeze(0).expand(B, -1)
    # token index being predicted
    token_idx = positions + 1
    pl = prompt_lens.to(device).unsqueeze(1)
    is_summary = token_idx >= pl
    is_real = attention_mask[:, 1:] == 1
    return (is_summary & is_real).long()


def logits_to_token_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Memory-friendly log π(x_t | x_<t) for t >= 1, shape (B, L-1).

    Uses cross-entropy instead of materializing a full (B, L, V) log-softmax,
    which OOMs on T4 with long Reddit prompts + large SmolLM vocab.
    """
    # logits[:, t] predicts input_ids[:, t+1]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    B, T, V = shift_logits.shape
    nll = F.cross_entropy(
        shift_logits.reshape(-1, V),
        shift_labels.reshape(-1),
        reduction="none",
    ).view(B, T)
    return -nll


def token_logprobs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor | None = None,
):
    """Return per-token log π(x_t | x_<t) for t >= 1, shape (B, L-1),
    and a summary mask that is 1 on summary tokens (not prompt, not padding).

    If prompt_lens is None, the summary mask is all zeros (caller must supply
    prompt_lens for a useful mask).
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    token_lp = logits_to_token_logprobs(outputs.logits, input_ids)

    if prompt_lens is None:
        summary_mask = torch.zeros_like(token_lp, dtype=torch.long)
    else:
        summary_mask = summary_token_mask(input_ids, attention_mask, prompt_lens)
    return token_lp, summary_mask


def slice_batch(batch: dict, start: int, end: int) -> dict:
    """Slice a collated batch along the batch dimension."""
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v[start:end]
        else:
            out[k] = v
    return out


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of values where mask == 1. Returns 0 if mask is empty."""
    mask_f = mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    return (values * mask_f).sum() / denom


def sequence_logprob(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lens: torch.Tensor,
) -> torch.Tensor:
    """Sum of token log-probs over summary tokens only. Shape (B,)."""
    token_lp, summary_mask = token_logprobs(
        model, input_ids, attention_mask, prompt_lens
    )
    mask_f = summary_mask.float()
    return (token_lp * mask_f).sum(dim=-1)


@torch.no_grad()
def generate_summaries(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int = 48,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
):
    """Generate continuation after left-padded prompts. Returns
    (full_sequences, prompt_len, gen_attention_mask) where prompt_len is the
    common prompt width (= input_ids.shape[1]).
    """
    model.eval()
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        # top_k=0 means disabled in the assignment; omit the arg
        if top_k and top_k > 0:
            gen_kwargs["top_k"] = top_k

    sequences = model.generate(**gen_kwargs)
    prompt_len = input_ids.shape[1]
    # Build attention mask for the full sequence: original mask + 1s for new tokens,
    # but zero out everything after the first EOS in the generated part.
    B, full_L = sequences.shape
    device = sequences.device
    new_mask = torch.ones(B, full_L, dtype=torch.long, device=device)
    new_mask[:, :prompt_len] = attention_mask
    eos_id = tokenizer.eos_token_id
    for i in range(B):
        gen = sequences[i, prompt_len:]
        eos_pos = (gen == eos_id).nonzero(as_tuple=False)
        if len(eos_pos) > 0:
            # keep EOS itself; zero everything after it
            cut = prompt_len + int(eos_pos[0].item()) + 1
            if cut < full_L:
                new_mask[i, cut:] = 0
                # also zero the ids after EOS for cleanliness (optional)
                sequences[i, cut:] = tokenizer.pad_token_id
    return sequences, prompt_len, new_mask


def decode_summary(tokenizer, sequences, prompt_len, attention_mask=None):
    """Decode generated summary tokens (after prompt), stopping at EOS."""
    texts = []
    for i in range(sequences.shape[0]):
        gen_ids = sequences[i, prompt_len:]
        if attention_mask is not None:
            gen_ids = gen_ids[attention_mask[i, prompt_len:] == 1]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        texts.append(text)
    return texts


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def freeze_model(model):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def set_seed(seed: int = 42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
