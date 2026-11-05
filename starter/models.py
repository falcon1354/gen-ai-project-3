import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from config import MODEL_NAME
except ImportError:
    from starter.config import MODEL_NAME


def load_tokenizer(name=MODEL_NAME):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:  # SmolLM2 has no pad token
        tok.pad_token = tok.eos_token
    return tok


def load_causal_lm(name_or_path=MODEL_NAME, device=None, dtype=None):
    """Plain pretrained causal LM (no heads added). Use it for the SFT
    policy, the frozen reference model, and as the backbone of the
    reward model and of the PPO value head.

    Keep weights in float32 by default so AdamW + GradScaler work. Memory
    is controlled via micro-batching / autocast in the train scripts.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=dtype)
    return model.to(device)
