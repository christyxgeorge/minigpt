"""Base load to handle text data"""
from abc import ABC, abstractmethod

import torch
from torch.nn import functional as F


class TextDataBase(ABC):
    """class for keeping text data"""

    filename: str

    def __init__(self, filename):
        self.filename = filename

    @abstractmethod
    def load_data(self):
        """Load Data from file"""

    @abstractmethod
    def encode(self, s) -> list[int]:
        """encode a string to a list of integers"""

    @abstractmethod
    def decode(self, l) -> str:
        """decode a list of integers back to a string"""

    @classmethod
    def get_loader(cls, type):
        if type == "shakespeare_char":
            from .shakespeare_char import TextDataTinyShakespeare

            return TextDataTinyShakespeare()
        else:
            print("Unknown Data Type: {type}")
            return None

    def generate_text(self, model, cfg, num_tokens=200):
        print("==================================================================")
        print("  Generating Text...")
        print("==================================================================")
        ## Create the initial 'text' to generate the continuation --> Using 0 = \n
        idx = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
        tokens = self.generate(model, cfg, idx, num_tokens=num_tokens)
        print(self.decode(tokens[0].tolist()))
        print("==================================================================")

    def generate(self, model, cfg, idx, num_tokens):
        """Generate next `num_tokens` tokens, idx --> B x T"""
        for _i in range(num_tokens):
            # print(f"Generating {i} token...")
            # crop idx to the last block size tokens [Remove extra tokens from the beginning!]
            # because positional encodings are defined only upto block_size
            idx_cond = idx[:, -cfg.block_size :]
            # get the predictions/losses
            logits, _loss = model(cfg.device, idx_cond)  ## logits --> B x T x C
            # focus on last time step (only the last character)
            logits = logits[:, -1, :]  ## --> B x C
            # counts = logits.exp() # counts, equivalent to N
            # probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
            probs = F.softmax(logits, dim=-1)  ## --> B x C
            # Sample from the probability distribution to get the next idx.
            idx_next = torch.multinomial(probs, num_samples=1)  ## --> B x 1
            idx = torch.cat((idx, idx_next), dim=1)  ## --> B x T+1
        return idx
