"""Task 1.3 – verify token_logprobs + summary mask on one SFT batch."""

import sys
from functools import partial
from pathlib import Path

# repo root must be on sys.path when launched as scripts/...
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from common import set_seed, token_logprobs
from starter.batching import collate_sft
from starter.config import MODEL_NAME, SEED
from starter.data import load_sft_splits
from starter.models import load_causal_lm, load_tokenizer


def main():
    set_seed(SEED)
    tok = load_tokenizer()
    sft_train, _ = load_sft_splits(tok)
    loader = DataLoader(
        sft_train,
        batch_size=2,
        shuffle=False,
        collate_fn=partial(collate_sft, pad_id=tok.pad_token_id),
    )
    batch = next(iter(loader))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_causal_lm(MODEL_NAME, device=device)
    model.eval()

    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    with torch.no_grad():
        lp, mask = token_logprobs(
            model, batch["input_ids"], batch["attention_mask"], batch["prompt_lens"]
        )

    print("logprobs shape:", tuple(lp.shape), "mask shape:", tuple(mask.shape))
    selected = batch["input_ids"][0, 1:][mask[0].bool()]
    print("Decoded summary tokens (mask==1):", repr(tok.decode(selected)))
    print("Expected summary + EOS:", repr(sft_train[0]["summary"] + tok.eos_token))


if __name__ == "__main__":
    main()
