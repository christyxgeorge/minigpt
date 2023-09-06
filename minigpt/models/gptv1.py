"""GPT Language Model v1: With a single attention head of `n_embed / 2`"""

import torch
import torch.nn as nn

from .blocks import AttentionHead
from .model_base import LanguageModelBase


class GPTLanguageModelv1(LanguageModelBase):
    def __init__(self, cfg):
        super().__init__()
        # Each token gets the logits for the next token from the lookup table
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embed)
        self.sa_head = AttentionHead(cfg, cfg.n_embed // 2)
        self.lm_head = nn.Linear(cfg.n_embed // 2, cfg.vocab_size)

    def forward(self, device, idx, targets=None):
        # idx, targets --> B x T (batch_size x block_size)
        B, T = idx.shape
        token_embedding = self.token_embedding_table(idx)  # B x T x C (n_embed)
        position_embedding = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T x C)

        ## B x T x C (position_embedding gets broadcasted for each batch)
        x = token_embedding + position_embedding
        x = self.sa_head(x)  # Apply a single attention head
        logits = self.lm_head(x)  # B x T x vocab_size
        loss = self.compute_loss(logits, targets)
        return logits, loss
