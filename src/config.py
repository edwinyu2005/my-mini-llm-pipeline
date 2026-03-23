import torch
from dataclasses import dataclass


@dataclass
class GPTConfig:
    """ Configuration parameters for the GPT model and training loop """

    # --- Architecture Defaults ---
    vocab_size: int = 65
    block_size: int = 256
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.2

    # --- Training Defaults ---
    batch_size: int = 64
    max_iters: int = 5000
    learning_rate: float = 3e-4

    # --- Device Autodetection ---
    # Prioritize CUDA for your Linux/RTX 3070 setup, fallback to others
    device: str = (
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu'
    )
