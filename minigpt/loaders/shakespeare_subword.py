"""class for managing data from the tiny shakespeare dataset"""

import tiktoken
from minigpt.loaders.base_dataset import GPT2_VOCAB_SIZE
from minigpt.loaders.text_dataset import TextDataset


class TinyShakespeareWordData(TextDataset):
    def __init__(self, args):
        super().__init__(args, "tiny_shakespeare.txt")
        self.vocab_size = GPT2_VOCAB_SIZE
        # encode with tiktoken gpt2 bpe
        self.enc = tiktoken.get_encoding("gpt2")

    @classmethod
    def get_vocab_size(cls, source, model_id):
        """Get the vocab size based on the source"""
        return GPT2_VOCAB_SIZE

    @property
    def name(self) -> str:
        """Return the dataset name"""
        return "Tiny Shakespeare (Word tokens)"

    def download(self, force=False):
        # download the tiny shakespeare dataset
        # data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        data_url = (
            "https://raw.githubusercontent.com/christyxgeorge/datasets/main/tiny_shakespeare.txt"
        )
        self.download_file(data_url, self.filename)

    def get_token_ids(self) -> tuple[list[int], list[int]]:
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

        train_ids = self.enc.encode_ordinary(train_text)
        val_ids = self.enc.encode_ordinary(val_text)

        return train_ids, val_ids

    def encode(self, s) -> list[int]:
        """encode a string to a list of integers"""
        return self.enc.encode_ordinary(s)

    def decode(self, l) -> str:
        """decode a list of integers back to a string"""
        return "".join(self.enc.decode(l))
