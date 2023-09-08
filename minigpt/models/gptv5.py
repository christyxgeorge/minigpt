"""
GPT Language Model v5: With multiple transformer blocks
with residual connections and projections into to the FF layer
"""
import torch
import torch.nn as nn

from .blocks import AttentionHead
from .model_base import LanguageModelBase


class MultiHeadAttentionProjection(nn.Module):
    """Multiple Attention heads - And with projection"""

    def __init__(self, cfg):
        super().__init__()
        head_size = cfg.n_embed // cfg.n_heads
        self.heads = nn.ModuleList([AttentionHead(cfg, head_size) for _ in range(cfg.n_heads)])
        self.proj = nn.Linear(cfg.n_embed, cfg.n_embed)

    def forward(self, x):
        # Concatenate on the `channel` dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForwardProjection(nn.Module):
    """Positional Feed Forward - With projection into the residual layer"""

    def __init__(self, cfg):
        super().__init__()
        # According to the `Attention is all you need` the inner FF layer has a multiplier of 4.
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embed, 4 * cfg.n_embed),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embed, cfg.n_embed),
        )

    def forward(self, x):
        return self.net(x)


class ResidualTransformerBlockProjection(nn.Module):
    """Transformer Block: Communication followed by Computation - With Residual Connections"""

    def __init__(self, cfg):
        super().__init__()
        self.sa = MultiHeadAttentionProjection(cfg)
        self.ffwd = FeedForwardProjection(cfg)

    def forward(self, x):
        x = x + self.sa(x)  # Apply the attention heads
        x = x + self.ffwd(x)  # B x T x C # Positional feed-forward
        return x


class GPTLanguageModelv5(LanguageModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Each token gets the logits for the next token from the lookup table
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embed)
        self.blocks = nn.Sequential(
            ResidualTransformerBlockProjection(cfg),
            ResidualTransformerBlockProjection(cfg),
            ResidualTransformerBlockProjection(cfg),
            ResidualTransformerBlockProjection(cfg),
        )
        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size)

    def forward(self, idx, targets=None):
        # idx, targets --> B x T (batch_size x block_size)
        B, T = idx.shape
        token_embedding = self.token_embedding_table(idx)  # B x T x C (n_embed)
        position_embedding = self.position_embedding_table(
            torch.arange(T, device=self.cfg.device)
        )  # (T x C)
        x = (
            token_embedding + position_embedding
        )  ## B x T x C (position_embedding gets broadcasted for each batch)
        x = self.blocks(x)
        logits = self.lm_head(x)  # B x T x vocab_size
        loss = self.compute_loss(logits, targets)
        return logits, loss
