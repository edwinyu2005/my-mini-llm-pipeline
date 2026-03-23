import numpy as np
import torch
import pytest
from scripts.pretrain import get_batch
from src.config import GPTConfig
from src.model.gpt import ToyGPT


@pytest.fixture
def dummy_data_dir(tmp_path):
    """ Create dummy binary files in a temporary directory for testing """
    train_data = np.arange(100, dtype=np.uint16)
    train_data.tofile(tmp_path / "train.bin")
    return str(tmp_path)


def test_get_batch(dummy_data_dir):
    """ Test the get_batch function with temporary binary data """
    config = GPTConfig(block_size=8, batch_size=4, device='cpu')
    x, y = get_batch('train', config, data_dir=dummy_data_dir)

    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert torch.equal(x[0, 1:], y[0, :-1])


def test_one_train_step(dummy_data_dir):
    """ Test a single training step to ensure no silent failures """
    config = GPTConfig(
        vocab_size=100,
        block_size=8,
        n_embd=16,
        n_head=2,
        n_layer=1,
        device='cpu'
    )
    model = ToyGPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    xb, yb = get_batch('train', config, data_dir=dummy_data_dir)
    _, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss is not None
    assert loss.item() > 0
