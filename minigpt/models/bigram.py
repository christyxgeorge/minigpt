"""Basic Bigram Language Model, Uses only the last token to predict the next one"""

import torch.nn as nn

from .model_base import LanguageModelBase


class BigramLanguageModel(LanguageModelBase):
    """Basic Bigram Language Model, Uses only the last token to predict the next one"""

    def __init__(self, cfg):
        super().__init__()
        # Each token gets the logits for the next token from the lookup table
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.vocab_size)

    def forward(self, _device, idx, targets=None):
        """idx, targets --> B x T (batch_size x block_size)"""
        logits = self.token_embedding_table(idx)  # B x T x C [vocab_size]
        loss = self.compute_loss(logits, targets)
        return logits, loss
