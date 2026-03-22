import torch
from src.config import GPTConfig
from src.model.gpt import ToyGPT


def get_batch(data, block_size, batch_size, device):
    """ Generate a small batch of data of inputs x and targets y """
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # HINT 1: Slice the input and target sequences from the data
    # Target 'y' is simply the input 'x' shifted by one position to the right
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x.to(device), y.to(device)


def main():
    # Hyperparameters for local RTX 3070
    batch_size = 32
    max_iters = 5000
    learning_rate = 1e-3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create dummy dataset to test the pipeline (replace with real data later)
    vocab_size = 65
    dummy_data = torch.randint(0, vocab_size, (10000,))

    config = GPTConfig(vocab_size=vocab_size)
    model = ToyGPT(config).to(device)

    # HINT 2: Initialize the AdamW optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter_num in range(max_iters):
        xb, yb = get_batch(dummy_data, config.block_size, batch_size, device)

        # HINT 3: Forward pass
        _, loss = model(xb, yb)

        # HINT 4: Zero gradients, backward pass, and step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_num % 500 == 0:
            print(f"Step {iter_num}: Loss {loss.item():.4f}")


if __name__ == '__main__':
    main()
