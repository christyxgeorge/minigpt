"""class for managing data from the tiny shakespeare dataset"""
import torch
from minigpt.loaders.base import TextDataBase


class CharDataTinyShakespeare(TextDataBase):
    def __init__(self, root_dir, verbose=False):
        super().__init__(f"{root_dir}/data/tiny_shakespeare.txt", verbose=verbose)
        self.load_data()

    def load_data(self):
        """Load Data from file"""
        if self.verbose:
            print("================================================================")
            print("Loading Data...")
            print("================================================================")
        with open(self.filename, "r", encoding="utf-8") as f:
            self.text = f.read()

        # Unique Characters in the text
        self.vocab_chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.vocab_chars)
        self.stoi = {ch: i for i, ch in enumerate(self.vocab_chars)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab_chars)}

        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)
        # print(self.data[:100])  ## Looks like 0 = \n, 1 = ' '

        # Split in train, val
        tv_split = int(0.9 * len(self.data))
        self.train_data = self.data[:tv_split]
        self.val_data = self.data[tv_split:]
        if self.verbose:
            print(f"    Text Length = {len(self.text)}")
            print(f"    Vocab Size = {self.vocab_size} => [{','.join(self.vocab_chars)}]")
            print(f"    Data Shape: {self.data.shape}, dtype: {self.data.dtype}")
            print(
                f"    Training len = {len(self.train_data)}, Validation len = {len(self.val_data)}"
            )
            print("================================================================")

    def encode(self, s) -> list[int]:
        """encode a string to a list of integers"""
        return [self.stoi[c] for c in s]

    def decode(self, l) -> str:
        """decode a list of integers back to a string"""
        return "".join([self.itos[i] for i in l])
