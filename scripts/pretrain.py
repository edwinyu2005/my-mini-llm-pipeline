import numpy as np
import os
import time
import torch
from src.config import GPTConfig
from src.model.gpt import ToyGPT


# Global cache to keep the memory map open across iterations
_DATA_CACHE = {}


def get_batch(split, config, data_dir=None):
    """ Fetch a batch of data, heavily optimized for I/O and GPU transfer """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), '../data/shakespeare')

    # 1. Initialize memmap ONLY ONCE per split to eliminate disk I/O bottleneck
    if split not in _DATA_CACHE:
        filename = os.path.join(data_dir, f'{split}.bin')
        _DATA_CACHE[split] = np.memmap(filename, dtype=np.uint16, mode='r')

    data = _DATA_CACHE[split]
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))

    # 2. Extract arrays and convert to PyTorch tensors
    x = torch.stack([
        torch.from_numpy((data[i:i + config.block_size]).astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy((data[i + 1:i + config.block_size + 1]).astype(np.int64))
        for i in ix
    ])

    # 3. Pin memory first, THEN transfer asynchronously to the GPU
    x_pinned = x.pin_memory()
    y_pinned = y.pin_memory()

    return (
        x_pinned.to(config.device, non_blocking=True),
        y_pinned.to(config.device, non_blocking=True)
    )


@torch.no_grad()
def estimate_loss(model, config, eval_iters=50):
    """ Evaluate the model on both train and val splits to monitor overfitting """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, config)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main():
    """ Main training loop with rich telemetry and validation """
    torch.set_float32_matmul_precision('high')
    config = GPTConfig()

    print(f"Initializing model on {config.device}...")
    model = ToyGPT(config).to(config.device)

    print("Compiling model... (Expect a ~30s pause on the first step)")
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Calculate tokens per batch for throughput metrics
    tokens_per_batch = config.batch_size * config.block_size

    t0 = time.time()
    epoch_t0 = time.time()
    for iter_num in range(config.max_iters):

        # Periodically evaluate the loss on train and val sets
        if iter_num % 500 == 0 or iter_num == config.max_iters - 1:
            losses = estimate_loss(model, config)
            epoch_dt = time.time() - epoch_t0
            print(f"\n--- Step {iter_num} Validation ---")
            print(f"Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
            if iter_num > 0:
                print(f"Epoch Duration (last 500 steps): {epoch_dt:.2f}s")
            print("---------------------------------\n")
            epoch_t0 = time.time()

        xb, yb = get_batch('train', config)

        _, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Print rich telemetry every 100 steps
        if iter_num % 100 == 0 and iter_num > 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1  # reset timer

            # Calculate metrics
            tokens_processed = tokens_per_batch * 100
            tok_sec = tokens_processed / dt

            print(
                f"Step {iter_num:4d} | "
                f"Loss: {loss.item():.4f} | "
                f"Time/100s: {dt:.2f}s | "
                f"Throughput: {tok_sec:,.2f} tok/s"
            )

    torch.save(model.state_dict(), 'gpt_shakespeare.pth')
    print("\nTraining complete. Model weights saved to gpt_shakespeare.pth")


if __name__ == '__main__':
    main()
