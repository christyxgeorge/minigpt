"""class for managing data from the tiny shakespeare dataset"""
import torch
from minigpt.loaders.base import BaseDataset


class TinyShakespeareCharData(BaseDataset):
    def __init__(self, data_dir, verbose=False):
        super().__init__(data_dir, "tiny_shakespeare.txt", verbose=verbose)

    def load_token_ids(self) -> tuple[list[int], list[int]]:
        """Load token IDs from file"""
        if self.verbose:
            print("=" * 100)
            print("Loading Data...")
            print("=" * 100)
        with open(self.filename, "r", encoding="utf-8") as f:
            self.text = f.read()

        # Unique Characters in the text
        self.vocab_chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.vocab_chars)
        self.stoi = {ch: i for i, ch in enumerate(self.vocab_chars)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab_chars)}

        token_ids = self.encode(self.text)

        # Split in train, val
        tv_split = int(0.9 * len(token_ids))
        train_ids = token_ids[:tv_split]
        val_ids = token_ids[tv_split:]
        return train_ids, val_ids

    def encode(self, s) -> list[int]:
        """encode a string to a list of integers"""
        return [self.stoi[c] for c in s]

    def decode(self, l) -> str:
        """decode a list of integers back to a string"""
        return "".join([self.itos[i] for i in l])
