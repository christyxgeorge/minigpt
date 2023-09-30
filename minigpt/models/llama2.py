"""
LLAM2 Model. Implemented from Scratch
"""
import logging
import math
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

from .base_model import BaseLanguageModel

logger = logging.getLogger(__name__)


@dataclass
class Llama2ModelArgs:
    # Hyperparameters for the trained Llama 7B model
    n_embed: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    # Number of KV Heads (Grouped Query Attention)
    n_kv_heads: Optional[int] = None
    hidden_dim: Optional[int] = None
    multiple_of: int = 256  # MLP hidden layer size will be multiple of
    norm_eps: float = 1e-5
    block_size: int = 2048  # a.k.a max_seq_len
    decay_lr: bool = False
    bias: bool = False


class RMSNorm(torch.nn.Module):
    """
    Layer norm which focusses on rescaling and not recentering.
    So, we dont use the variance and mean, but the RMS
    RMS = sqrt(mean(x^2) + eps)
    x = x / RMS * gamma
    gamma is a learnable parameter
    Advantages: less computational than LayerNorm, Works well!
    """

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


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # ======== TODO: Why are we doing this? Transformer paper uses 4?
        if cfg.hidden_dim is None:
            cfg.hidden_dim = 4 * cfg.n_embed
            cfg.hidden_dim = int(2 * cfg.hidden_dim / 3)
            cfg.hidden_dim = cfg.multiple_of * (
                (cfg.hidden_dim + cfg.multiple_of - 1) // cfg.multiple_of
            )
        self.w1 = nn.Linear(cfg.n_embed, cfg.hidden_dim, bias=cfg.bias)
        self.w2 = nn.Linear(cfg.hidden_dim, cfg.n_embed, bias=cfg.bias)
        self.w3 = nn.Linear(cfg.n_embed, cfg.hidden_dim, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class MultiHeadAttention(nn.Module):
    """Multiple Attention heads - with Dropout - In parallel (split/combine)"""

    def __init__(self, cfg):
        super().__init__()

        # TODO: ==== Why do we need this?
        model_parallel_size = 1
        self.n_local_heads = cfg.n_heads // model_parallel_size
        self.n_local_kv_heads = cfg.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = cfg.n_embed // cfg.n_heads
        # self.wq = nn.Linear(cfg.n_embed, cfg.n_heads * self.head_dim, bias=cfg.bias)
        # self.wk = nn.Linear(cfg.n_embed, cfg.n_kv_heads * self.head_dim, bias=cfg.bias)
        # self.wv = nn.Linear(cfg.n_embed, cfg.n_kv_heads * self.head_dim, bias=cfg.bias)
        # self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.n_embed, bias=cfg.bias)
        # TODO: ==== Why do we need this?

        self.wk = nn.Linear(cfg.n_embed, cfg.n_embed, bias=cfg.bias)
        self.wq = nn.Linear(cfg.n_embed, cfg.n_embed, bias=cfg.bias)
        self.wv = nn.Linear(cfg.n_embed, cfg.n_embed, bias=cfg.bias)

        self.attention_dropout = nn.Dropout(cfg.dropout)
        self.residual_dropout = nn.Dropout(cfg.dropout)

        self.wo = nn.Linear(cfg.n_embed, cfg.n_embed, bias=cfg.bias)  # Final Output Projection

        # Local Variables
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        self.head_size = cfg.n_embed // cfg.n_heads
        self.num_heads = cfg.n_heads
        self.cfg = cfg

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, cfg.block_size, cfg.block_size), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(
        self,
        x,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        B, T, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(B, T, self.n_local_heads, self.head_dim)
        xk = xk.view(B, T, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(B, T, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (B, T, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (B, T, n_local_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (B, n_local_heads, T, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # make heads into a batch dimension
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=None,
                dropout_p=self.cfg.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, "mask")  # nosec
            scores = scores + self.mask[:, :, :T, :T]  # type: ignore # (B, n_local_heads, T, cache_len + T)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attention_dropout(scores)
            output = torch.matmul(scores, xv)  # (B, n_local_heads, T, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(B, T, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.residual_dropout(output)
        return output


class TransformerBlock(nn.Module):
    """
    Transformer Block: Communication followed by Computation - With Residual Connections
    The layer norm we apply is called pre-norm. Slighly different from the original paper
    """

    def __init__(self, layer_id, cfg):
        super().__init__()
        self.layer_id = layer_id
        self.attention = MultiHeadAttention(cfg)
        self.feed_forward = FeedForward(cfg)
        self.attention_norm = RMSNorm(cfg.n_embed, eps=cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.n_embed, eps=cfg.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Llama2LanguageModel(BaseLanguageModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        # Ensure that bias is False
        cfg.update_hparams(bias=False)

        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(cfg.n_layers):
            self.layers.append(TransformerBlock(layer_id, cfg))
        self.norm = RMSNorm(cfg.n_embed, eps=cfg.norm_eps)
        self.output = nn.Linear(cfg.n_embed, cfg.vocab_size, bias=cfg.bias)

        # share the unembedding parameters with the embedding parameters
        # https://paperswithcode.com/method/weight-tying
        self.tok_embeddings.weight = self.output.weight

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(cfg.n_embed // cfg.n_heads, cfg.block_size)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("w3.weight") or pn.endswith("wo.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

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
