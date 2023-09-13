"""
GPT Language Model v7: Compute the Multiple heads in parallel
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from .blocks import FeedForwardDropout
from .model_base import LanguageModelBase


class MultiHeadAttentionLlama2(nn.Module):
    """Multiple Attention heads - with Dropout - In parallel (split/combine)"""

    def __init__(self, cfg):
        super().__init__()
        self.key = nn.Linear(cfg.n_embed, cfg.n_embed, bias=False)
        self.query = nn.Linear(cfg.n_embed, cfg.n_embed, bias=False)
        self.value = nn.Linear(cfg.n_embed, cfg.n_embed, bias=False)
        self.att_dropout = nn.Dropout(cfg.dropout)
        self.residual_dropout = nn.Dropout(cfg.dropout)

        self.proj = nn.Linear(cfg.n_embed, cfg.n_embed)

        # Local Variables
        self.flash = False  # hasattr(torch.nn.functional, "scaled_dot_product_attention")
        self.head_size = cfg.n_embed // cfg.n_heads
        self.num_heads = cfg.n_heads
        self.cfg = cfg

        if not self.flash:
            self.register_buffer(
                "tril",
                torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
                    1, 1, cfg.block_size, cfg.block_size
                ),
            )  ## T x T

    def split_heads(self, x):
        # Reshape the input to have num_heads for multi-head attention
        B, T, _ = x.size()  # batch_size, seq_length, d_model
        return x.view(B, T, self.num_heads, self.head_size).transpose(1, 2)

    def combine_heads(self, x):
        # Combine the multiple heads back to original shape
        B, _, T, _ = x.size()
        return x.transpose(1, 2).contiguous().view(B, T, self.cfg.n_embed)

    def forward(self, x):
        _, T, C = x.shape
        k = self.split_heads(self.key(x))  # B x num_heads x T x head_size (hs)
        q = self.split_heads(self.query(x))  # B x num_heads x T x head_size (hs)

        if self.flash:
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.cfg.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            att = q @ k.transpose(-2, -1) * C**-0.5  # Scaled attention
            att = att.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.att_dropout(att)
            v = self.split_heads(self.value(x))  # B x num_heads x T x head_size (hs)
            out = att @ v
        out = self.combine_heads(out)
        return self.residual_dropout(self.proj(out))


class ResidualTransformerBlockLlama2(nn.Module):
    """Transformer Block: Communication followed by Computation - With Residual Connections"""

    """The layer norm we apply is called pre-norm. Slighly different from the original paper"""

    def __init__(self, cfg):
        super().__init__()
        self.sa = MultiHeadAttentionLlama2(cfg)
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


class GPTLanguageModelLlama2(LanguageModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        # Each token gets the logits for the next token from the lookup table
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embed)
        self.blocks = nn.Sequential(
            *[ResidualTransformerBlockLlama2(cfg) for _ in range(cfg.n_layers)]
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
