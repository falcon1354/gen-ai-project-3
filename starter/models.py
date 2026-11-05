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
    reward model and of the PPO value head."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # fp16 on CUDA roughly halves activation/logit memory on a free T4
    if dtype is None:
        if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
            dtype = torch.float16
        else:
            dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(name_or_path, torch_dtype=dtype)
    return model.to(device)
