import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """ One head of self-attention """
    def __init__(self, config, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        # Causal Mask
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape  # C is n_embd or d_model
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # HINT 1: Compute attention scores ("affinities")
        # Formula: (Q * K^T) / sqrt(d_k)
        # Remember to transpose the last two dimensions of 'k'
        # Attention Weights
        wei = q @ k.transpose(-2, -1) / (k.shape[-1] ** 0.5)  # (B, T, T)

        # HINT 2: Apply the causal mask to hide future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, -torch.inf)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        # HINT 3: Multiply the attention weights by the Values
        out = wei @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # 1. Input x: The initial features from the previous layer or embedding
        # Shape of x: (B, T, n_embd)

        # 2. Parallel Computation & Concatenation
        # [h(x) for h in self.heads] runs 'x' through each Head independently.
        # Each 'h(x)' returns a tensor of shape: (B, T, head_size)
        # We have 'config.n_head' number of such tensors in a list.

        # torch.cat(..., dim=-1) joins them along the last dimension (head_size).
        # Calculation: head_size * n_head = n_embd
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_embd)

        # 3. Final Linear Projection (self.proj)
        # This mixes the concatenated features across all heads.
        # Even though the shape is already (B, T, n_embd), this step is crucial
        # so heads can communicate with each other before exiting the block.
        out = self.proj(out)  # (B, T, n_embd)

        # 4. Regularization (Dropout)
        # Randomly zeroes some elements of the tensor with probability p during training
        out = self.dropout(out)  # (B, T, n_embd)

        return out


class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity """
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            # Why 4 * n_embd?
            # 1. Capacity: Attention is for "communication" between tokens.
            #    This MLP is for "computation" and memorizing world knowledge.
            #    Expanding the dimension acts as a large key-value memory bank.
            # 2. Non-linearity: Projecting features into a higher-dimensional space
            #    makes it easier for the ReLU activation to disentangle complex patterns,
            #    before projecting them back down to the residual stream (n_embd).
            # 3. Empirical: The original "Attention Is All You Need" paper used
            #    d_model=512 and d_ff=2048 (exactly 4x). It became the industry standard.
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            # Project back to the original embedding dimension to allow residual connections
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        # x shape: (B, T, n_embd) -> (B, T, 4 * n_embd) -> (B, T, n_embd)
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        # HINT 5: Apply residual connections around the sub-layers
        # In modern GPT architectures, LayerNorm is applied BEFORE the sub-layers (Pre-Norm)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class ToyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Token Embedding: Translates integer word IDs into dense vectors
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)

        # 2. Position Embedding: Gives the model a sense of sequence order
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)

        # 3. The Core Brain: Stacking multiple Transformer blocks
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])

        # 4. Final LayerNorm: Cleans up the signal before the final prediction
        self.ln_f = nn.LayerNorm(config.n_embd)

        # 5. Language Modeling Head: Projects the feature back to vocabulary size
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Retrieve the embeddings for the tokens and their positions
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))

        # HINT 6 Answer: Combine meaning and position
        # Broadcast semantics: pos_emb (T, C) is added to tok_emb (B, T, C)
        x = tok_emb + pos_emb

        # Pass through the deep neural network (Transformer Blocks)
        x = self.blocks(x)

        # Final normalization before outputting probabilities
        x = self.ln_f(x)

        # Predict the next token logits
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            # Reshape logits and targets for PyTorch's cross_entropy function
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
