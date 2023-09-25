"""Basic Bigram Language Model, Uses only the last token to predict the next one"""

from dataclasses import asdict, dataclass

import torch.nn as nn

from .base_model import BaseLanguageModel

# Note, in the `making gpt` video, the author uses a batch size of 32, and 16000 iterations
# The final loss is 2.576. Our loss after 16K iterations is approximately 2.482


@dataclass
class BigramModelArgs:
    # Hyperparameters for the Bigram Language model
    block_size: int = 8
    batch_size: int = 32
    learning_rate: float = 1e-3
    decay_lr: bool = False


class BigramLanguageModel(BaseLanguageModel):
    """Basic Bigram Language Model, Uses only the last token to predict the next one"""

    def __init__(self, cfg):
        super().__init__(cfg)
        # Each token gets the logits for the next token from the lookup table
        if cfg.vocab_size > 200:
            raise ValueError(f"Vocab Size [{cfg.vocab_size}] is too high! Should be <= 200")
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.vocab_size)

    @staticmethod
    def fixed_params():
        """Return a dict of fixed params for the model, Empty by default"""
        return asdict(BigramModelArgs())

    def forward(self, idx, targets=None):
        """idx, targets --> B x T (batch_size x block_size)"""
        logits = self.token_embedding_table(idx)  # B x T x C [vocab_size]
        loss = self.compute_loss(logits, targets)
        return logits, loss
