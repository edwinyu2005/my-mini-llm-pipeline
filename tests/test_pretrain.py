import torch
from scripts.pretrain import get_batch
from src.model.gpt import ToyGPT
from src.config import GPTConfig


def test_get_batch():
    vocab_size = 10
    data = torch.randint(0, vocab_size, (100,))
    block_size = 8
    batch_size = 4

    x, y = get_batch(data, block_size, batch_size, device='cpu')

    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)
    # Check the "shift" logic: y[i] should be x[i+1]
    # For the first sequence in batch, check if y follows x shifted by 1
    assert torch.equal(x[0, 1:], y[0, :-1])


def test_one_train_step():
    # Test if a single forward/backward pass works without crashing
    config = GPTConfig(vocab_size=10, block_size=8, n_embd=16, n_head=2, n_layer=1)
    model = ToyGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    data = torch.randint(0, 10, (50,))
    xb, yb = get_batch(data, config.block_size, batch_size=2, device='cpu')

    # Forward
    _, loss = model(xb, yb)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss is not None
    assert loss.item() > 0
