"""
GPT Language Model v4: With multiple transformer blocks
This does not perform too well. As deep networks suffer from optimization issues.
To fix this, we need to add residual connections
"""
import torch
import torch.nn as nn

from .base_model import BaseLanguageModel
from .blocks import FeedForward, MultiHeadAttention


class TransformerBlock(nn.Module):
    """Transformer Block: Communication followed by Computation"""

    def __init__(self, cfg):
        super().__init__()
        self.sa = MultiHeadAttention(cfg)
        self.ffwd = FeedForward(cfg)

    def forward(self, x):
        x = self.sa(x)  # Apply the attention heads
        x = self.ffwd(x)  # B x T x C # Positional feed-forward
        return x


class GPTLanguageModelv4(BaseLanguageModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Each token gets the logits for the next token from the lookup table
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embed)
        self.blocks = nn.Sequential(
            TransformerBlock(cfg),
            TransformerBlock(cfg),
            TransformerBlock(cfg),
            TransformerBlock(cfg),
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
