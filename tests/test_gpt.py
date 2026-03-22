import torch
import pytest
from src.config import GPTConfig
from src.model.gpt import Head, MultiHeadAttention, FeedForward, Block, ToyGPT


@pytest.fixture
def config():
    # Use a tiny configuration specifically for fast unit testing
    return GPTConfig(
        vocab_size=100,
        block_size=16,
        n_embd=32,
        n_head=4,
        n_layer=2,
        dropout=0.0
    )


def test_head_shape(config):
    # head_size is derived from n_embd and n_head
    head_size = config.n_embd // config.n_head
    model = Head(config, head_size)
    x = torch.randn(2, config.block_size, config.n_embd)
    out = model(x)

    # Check if the output shape is correctly transformed to (B, T, head_size)
    assert out.shape == (2, config.block_size, head_size)


def test_multi_head_attention_shape(config):
    model = MultiHeadAttention(config)
    x = torch.randn(2, config.block_size, config.n_embd)
    out = model(x)

    # Check if the output shape is restored to (B, T, n_embd) after concatenation
    assert out.shape == (2, config.block_size, config.n_embd)


def test_feed_forward_shape(config):
    model = FeedForward(config)
    x = torch.randn(2, config.block_size, config.n_embd)
    out = model(x)

    # FeedForward should not change the input shape
    assert out.shape == (2, config.block_size, config.n_embd)


def test_block_shape(config):
    model = Block(config)
    x = torch.randn(2, config.block_size, config.n_embd)
    out = model(x)

    # The entire Block should maintain the (B, T, n_embd) shape
    assert out.shape == (2, config.block_size, config.n_embd)


def test_toygpt_forward_without_targets(config):
    model = ToyGPT(config)
    # Input indices shape: (Batch, Time)
    idx = torch.randint(0, config.vocab_size, (2, config.block_size))
    logits, loss = model(idx)

    # Logits should have shape (Batch, Time, Vocab_size)
    assert logits.shape == (2, config.block_size, config.vocab_size)
    assert loss is None


def test_toygpt_forward_with_targets(config):
    model = ToyGPT(config)
    idx = torch.randint(0, config.vocab_size, (2, config.block_size))
    targets = torch.randint(0, config.vocab_size, (2, config.block_size))
    _, loss = model(idx, targets=targets)

    # When targets are provided, loss must be calculated and returned as a scalar
    assert loss is not None
    assert loss.item() > 0
