import os
import pickle
import requests
import numpy as np


def download_data(url, filepath):
    """ Download the raw text file from the provided URL """
    if not os.path.exists(filepath):
        print(f"Downloading {url}...")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(requests.get(url).text)


def prepare_shakespeare():
    """ Process the raw text into binary shards for training """
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

    download_data(data_url, input_file_path)

    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()
    print(f"Length of dataset in characters: {len(data):,}")

    # Create vocabulary mapping
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"Vocab size: {vocab_size}")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    # Split into train (90%) and val (10%)
    n = len(data)
    train_data = data[:int(n * 0.9)]
    val_data = data[int(n * 0.9):]

    # Convert to numpy arrays of uint16
    train_ids = np.array(encode(train_data), dtype=np.uint16)
    val_ids = np.array(encode(val_data), dtype=np.uint16)

    # Save to binary files
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

    # Save meta information for decoding during inference
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print("Data preparation complete. Binary files and meta.pkl saved.")


if __name__ == '__main__':
    prepare_shakespeare()
