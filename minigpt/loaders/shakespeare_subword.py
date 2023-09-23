"""class for managing data from the tiny shakespeare dataset"""

import tiktoken
from minigpt.loaders.text_dataset import TextDataset

# GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
GPT2_VOCAB_SIZE = 50304


class TinyShakespeareWordData(TextDataset):
    def __init__(self, args):
        self.vocab_size = GPT2_VOCAB_SIZE
        super().__init__(args, "tiny_shakespeare.txt")

    @classmethod
    def get_vocab_size(cls, _source, _vocab_soure: str | None = None):
        """Get the vocab size based on the source"""
        return GPT2_VOCAB_SIZE

    @property
    def name(self) -> str:
        """Return the dataset name"""
        return "Tiny Shakespeare (Word tokens)"

    def is_prepared(self) -> bool:
        """Check if the data(bin_files) have been prepared"""
        bin_files_exist = (self.work_dir / self.train_bin).exists() and (
            self.work_dir / self.val_bin
        ).exists()
        return bin_files_exist

    def download(self, force=False):
        # download the tiny shakespeare dataset
        # data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        data_url = (
            "https://raw.githubusercontent.com/christyxgeorge/datasets/main/tiny_shakespeare.txt"
        )
        self.download_file(data_url, self.filename)

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
