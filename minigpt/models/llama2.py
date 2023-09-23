"""
GPT Language Model v7: Compute the Multiple heads in parallel
"""
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from .base_model import BaseLanguageModel
from .blocks import FeedForwardDropout


@dataclass
class ModelArgs:
    # default hyperparameters for the Llama 7B model
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    # Number of KV Heads (Grouped Query Attention)
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    max_seq_len: int = 2048  # Same as block size
    dropout: float = 0.0


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim  # nosec
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])  # nosec
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class MultiHeadAttention(nn.Module):
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
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
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


class TransformerBlock(nn.Module):
    """Transformer Block: Communication followed by Computation - With Residual Connections"""

    """The layer norm we apply is called pre-norm. Slighly different from the original paper"""

    def __init__(self, layer_id, cfg):
        super().__init__()
        self.layer_id = layer_id
        self.sa = MultiHeadAttention(cfg)
        self.ffwd = FeedForwardDropout(cfg)
        # Although we implemented the layer norm below, we use the torch version of it!
        # Normalize each token. (mean of the `n_embed` channels)
        self.attention_norm = nn.LayerNorm(cfg.n_embed)
        self.ln2 = nn.LayerNorm(cfg.n_embed)

    def forward(self, x, freqs_cos, freqs_sin):
        # Apply the attention heads on the pre-norm'ed `x`
        x = x + self.sa(self.ln1(x))
        # B x T x C # Positional feed-forward on the pre-norm'ed `x`
        x = x + self.ffwd(self.ln2(x))
        return x


class Llama2LanguageModel(BaseLanguageModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(cfg.n_layers):
            self.layers.append(TransformerBlock(layer_id, cfg))
        self.norm = RMSNorm(cfg.n_embed, eps=cfg.norm_eps)
        self.output = nn.Linear(cfg.n_embed, cfg.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        # https://paperswithcode.com/method/weight-tying
        self.tok_embeddings.weight = self.output.weight

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.params.dim // cfg.n_heads, self.params.max_seq_len
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("w3.weight") or pn.endswith("wo.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_layers))

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx, targets --> B x T (batch_size x block_size)
        B, T = idx.shape
        h = self.tok_embeddings(idx)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:T]
        freqs_sin = self.freqs_sin[:T]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)

        logits = self.output(h)  # B x T x vocab_size
        # loss = self.compute_loss(logits, targets)
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)
            self.last_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(h[:, [-1], :])  # note: using list [-1] to preserve the time dim
            self.last_loss = None

        return logits, self.last_loss
