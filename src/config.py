from dataclasses import dataclass


@dataclass
class GPTConfig:
    # Default parameters suitable for a local RTX 3070 and a tiny dataset
    vocab_size: int = 65      # Default for tiny shakespeare character level
    block_size: int = 256     # Maximum context length
    n_embd: int = 384         # Embedding dimension
    n_head: int = 6           # Number of attention heads
    n_layer: int = 6          # Number of transformer blocks
    dropout: float = 0.2      # Dropout rate for regularization
