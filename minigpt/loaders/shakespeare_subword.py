"""class for managing data from the tiny shakespeare dataset"""
import tiktoken
import torch

# import numpy as np
from minigpt.loaders.base import TextDataBase


class TextDataTinyShakespeare(TextDataBase):
    def __init__(self, root_dir, verbose=False):
        super().__init__(root_dir, f"{root_dir}/data/tiny_shakespeare.txt", verbose=verbose)
        self.load_data()

    def load_data(self):
        """Load Data from file"""
        if self.verbose:
            print("================================================================")
            print("Loading Data...")
            print("================================================================")

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

        self.train_data = torch.tensor(train_ids, dtype=torch.long)
        self.val_data = torch.tensor(val_ids, dtype=torch.long)
        if self.verbose:
            print(f"    Text Length = {len(self.text)}")
            print(
                f"    Training len = {len(train_ids):,} tokens, Validation len = {len(val_ids):,} tokens"
            )
            print("================================================================")

        # export to bin files
        # train_ids = np.array(train_ids, dtype=np.uint16)
        # val_ids = np.array(val_ids, dtype=np.uint16)
        # train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
        # val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

    def encode(self, s) -> list[int]:
        """encode a string to a list of integers"""
        return self.enc.encode_ordinary(s)

    def decode(self, l) -> str:
        """decode a list of integers back to a string"""
        return "".join(self.enc.decode(l))
