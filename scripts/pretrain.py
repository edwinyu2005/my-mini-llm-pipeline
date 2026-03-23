import numpy as np
import os
import torch
from src.config import GPTConfig
from src.model.gpt import ToyGPT


def get_batch(split, config, data_dir=None):
    """ Fetch a batch of data from the binary files using memmap """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '../data/shakespeare')

    filename = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(filename, dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))

    x = torch.stack([
        torch.from_numpy((data[i:i + config.block_size]).astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy((data[i + 1:i + config.block_size + 1]).astype(np.int64))
        for i in ix
    ])

    return x.to(config.device), y.to(config.device)


def main():
    """ Main training loop with RTX 3070 optimizations """
    # Enable Tensor Cores for Ampere architecture (RTX 3070)
    torch.set_float32_matmul_precision('high')

    config = GPTConfig()
    print(f"Training on device: {config.device}")

    model = ToyGPT(config).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for iter_num in range(config.max_iters):
        xb, yb = get_batch('train', config)

        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter_num % 500 == 0:
            print(f"Step {iter_num}: Training Loss {loss.item():.4f}")


if __name__ == '__main__':
    main()
