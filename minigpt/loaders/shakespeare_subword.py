"""class for managing data from the tiny shakespeare dataset"""

import requests  # type: ignore
import tiktoken
from minigpt.loaders.loader_base import BaseDataset


class TinyShakespeareWordData(BaseDataset):
    @property
    def name(self) -> str:
        """Return the dataset name"""
        return "Tiny Shakespeare (Word tokens)"

    def download(self):
        # download the tiny shakespeare dataset
        input_file_path = self.data_dir / "tiny_shakespeare.txt"
        if not input_file_path.exists():
            data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            with open(input_file_path, "w") as f:
                f.write(requests.get(data_url).text)  # nosec

    def get_metadata(self):
        """Get metadata to save alongwith train/val.bin"""
        return {"vocab_size": self.vocab_size}

    def load_metadata(self, metadata):
        """Load metadata saved alongwith train/val.bin"""
        self.vocab_size = metadata["vocab_size"]

    def load_token_ids(self) -> tuple[list[int], list[int]]:
        """Load Token IDs from Dataset"""
        if self.verbose:
            print("=" * 100)
            print("Loading Data [{self.filename}]...")
            print("=" * 100)

        with open(self.filename, "r") as f:
            text = f.read()

        # Split in train, val
        tv_split = int(0.9 * len(text))
        train_text = text[:tv_split]
        val_text = text[tv_split:]

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
