"""
GPT Language Model v6: With multiple transformer blocks
with residual connections and projections into to the FF layer
and LayerNorms and Dropouts appropriately
References:
Dropout: https://jmlr.org/papers/v15/srivastava14a.html#:~:text=Dropout%20is%20a%20technique%20for,number%20of%20different%20thinned%20networks.
"""
import torch
import torch.nn as nn

from .blocks import ResidualTransformerBlockDropout
from .model_base import LanguageModelBase


class GPTLanguageModelv6(LanguageModelBase):
    def __init__(self, cfg):
        super().__init__()
        # Each token gets the logits for the next token from the lookup table
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embed)
        # self.blocks = nn.Sequential(
        #     ResidualTransformerBlockDropout(cfg),
        #     ResidualTransformerBlockDropout(cfg),
        #     ResidualTransformerBlockDropout(cfg),
        #     ResidualTransformerBlockDropout(cfg),
        #     nn.LayerNorm(n_embed),
        # )
        self.blocks = nn.Sequential(
            *[ResidualTransformerBlockDropout(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_f = nn.LayerNorm(cfg.n_embed)
        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size)

    def forward(self, device, idx, targets=None):
        # idx, targets --> B x T (batch_size x block_size)
        B, T = idx.shape
        token_embedding = self.token_embedding_table(idx)  # B x T x C (n_embed)
        position_embedding = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T x C)
        x = (
            token_embedding + position_embedding
        )  ## B x T x C (position_embedding gets broadcasted for each batch)
        x = self.blocks(x)
        x = self.ln_f(x)  # B x T x C
        logits = self.lm_head(x)  # B x T x vocab_size
        loss = self.compute_loss(logits, targets)
        return logits, loss