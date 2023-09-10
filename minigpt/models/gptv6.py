"""
GPT Language Model v6: With multiple transformer blocks
with residual connections and projections into to the FF layer
and LayerNorms and Dropouts appropriately
References:
Dropout: https://jmlr.org/papers/v15/srivastava14a.html#:~:text=Dropout%20is%20a%20technique%20for,number%20of%20different%20thinned%20networks.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from .blocks import FeedForwardDropout
from .model_base import LanguageModelBase


class AttentionHeadDropout(nn.Module):
    """Single Attention head - with Dropout"""

    def __init__(self, cfg, head_size):
        super().__init__()
        self.key = nn.Linear(cfg.n_embed, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embed, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embed, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        )  ## T x T

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # B x T x head_size (hs)
        q = self.query(x)  # B x T x head_size (hs)

        wei = q @ k.transpose(-2, -1) * C**-0.5  # Scaled attention
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)  # B x T x head_size (hs)
        out = wei @ v
        return out


class MultiHeadAttentionDropOut(nn.Module):
    """Multiple Attention heads - And with projection, dropout"""

    def __init__(self, cfg):
        super().__init__()
        head_size = cfg.n_embed // cfg.n_heads
        self.heads = nn.ModuleList(
            [AttentionHeadDropout(cfg, head_size) for _ in range(cfg.n_heads)]
        )
        self.proj = nn.Linear(cfg.n_embed, cfg.n_embed)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        # Concatenate on the `channel` dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class ResidualTransformerBlockDropout(nn.Module):
    """Transformer Block: Communication followed by Computation - With Residual Connections"""

    """The layer norm we apply is called pre-norm. Slighly different from the original paper"""

    def __init__(self, cfg):
        super().__init__()
        self.sa = MultiHeadAttentionDropOut(cfg)
        self.ffwd = FeedForwardDropout(cfg)
        # Although we implemented the layer norm below, we use the torch version of it!
        # Normalize each token. (mean of the `n_embed` channels)
        self.ln1 = nn.LayerNorm(cfg.n_embed)
        self.ln2 = nn.LayerNorm(cfg.n_embed)

    def forward(self, x):
        # Apply the attention heads on the pre-norm'ed `x`
        x = x + self.sa(self.ln1(x))
        # B x T x C # Positional feed-forward on the pre-norm'ed `x`
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModelv6(LanguageModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
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
        x = self.ln_f(x)  # B x T x C
        logits = self.lm_head(x)  # B x T x vocab_size
        loss = self.compute_loss(logits, targets)
        return logits, loss
