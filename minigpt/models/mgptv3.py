"""GPT Language Model v3: With multiple attention heads and Feed forward"""
import torch
import torch.nn as nn

from .base_model import BaseLanguageModel
from .blocks import FeedForward, MultiHeadAttention


class GPTLanguageModelv3(BaseLanguageModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embed)
        ## 4 heads of 8-dimensional self-attention
        self.sa_heads = MultiHeadAttention(cfg)
        self.ffwd = FeedForward(cfg)
        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size)

    def forward(self, idx, targets=None):
        # idx, targets --> B x T (batch_size x block_size)
        B, T = idx.shape
        token_embedding = self.token_embedding_table(idx)  # B x T x C (n_embed)
        pos = torch.arange(T, device=self.cfg.device)
        position_embedding = self.position_embedding_table(pos)  # (T x C)
        x = (
            token_embedding + position_embedding
        )  ## B x T x C (position_embedding gets broadcasted for each batch)
        x = self.sa_heads(x)  # Apply the attention heads
        x = self.ffwd(x)  # B x T x C
        logits = self.lm_head(x)  # B x T x vocab_size
        loss = self.compute_loss(logits, targets)
        return logits, loss
