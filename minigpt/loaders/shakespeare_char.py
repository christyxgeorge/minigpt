"""class for managing data from the tiny shakespeare dataset"""
import requests  # type: ignore
from minigpt.loaders.loader_base import BaseDataset


class TinyShakespeareCharData(BaseDataset):
    @property
    def name(self) -> str:
        """Return the dataset name"""
        return "Tiny Shakespeare (Character tokens)"

    def download(self):
        # download the tiny shakespeare dataset
        input_file_path = self.data_dir / "tiny_shakespeare.txt"
        if not input_file_path.exists():
            data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            with open(input_file_path, "w") as f:
                f.write(requests.get(data_url).text)  # nosec

    def get_metadata(self):
        """Get metadata to save alongwith train/val.bin"""
        return {"vocab_chars": self.vocab_chars, "vocab_size": self.vocab_size}

    def load_metadata(self, metadata):
        """Load metadata saved alongwith train/val.bin"""
        self.vocab_chars = metadata["vocab_chars"]
        self.vocab_size = metadata["vocab_size"]
        self.stoi = {ch: i for i, ch in enumerate(self.vocab_chars)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab_chars)}

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
