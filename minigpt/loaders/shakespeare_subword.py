"""class for managing data from the tiny shakespeare dataset"""

import tiktoken
from minigpt.loaders.loader_base import BaseDataset


class TinyShakespeareWordData(BaseDataset):
    @property
    def name(self) -> str:
        """Return the dataset name"""
        return "Tiny Shakespeare (Word tokens)"

    def load_token_ids(self) -> tuple[list[int], list[int]]:
        """Load Token IDs from Dataset"""
        if self.verbose:
            print("=" * 100)
            print("Loading Data...")
            print("=" * 100)

        with open(self.filename, "r") as f:
            self.text = f.read()

        # Split in train, val
        tv_split = int(0.9 * len(self.text))
        train_text = self.text[:tv_split]
        val_text = self.text[tv_split:]

        # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
        self.vocab_size = 50304

        # encode with tiktoken gpt2 bpe
        self.enc = tiktoken.get_encoding("gpt2")
        train_ids = self.enc.encode_ordinary(train_text)
        val_ids = self.enc.encode_ordinary(val_text)

        return train_ids, val_ids

    def encode(self, s) -> list[int]:
        """encode a string to a list of integers"""
        return self.enc.encode_ordinary(s)

    def decode(self, l) -> str:
        """decode a list of integers back to a string"""
        return "".join(self.enc.decode(l))
