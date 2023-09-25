"""GPT2 Pretrained Language Model"""

import logging
import math
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

from .base_model import BaseLanguageModel
from .blocks import LayerNorm

PRETRAINED_MODELS = {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

logger = logging.getLogger(__name__)


@dataclass
class GPT2ModelArgs:
    # hyperparameters for the trained GPT2 model
    n_embed: int = 768  # Embedding dimension
    n_layers: int = 12
    n_heads: int = 12
    block_size: int = 1024
    bias: bool = False
    batch_size: int = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
    gradient_accumulation_steps: int = 5 * 8  # used to simulate larger batch sizes
    learning_rate = 6e-4  # max learning rate
    min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # max_iters = 600000 # total number of training iterations


class CausalSelfAttention(nn.Module):
    """Multiple Attention heads - with Dropout - In parallel (split/combine)"""

    def __init__(self, cfg):
        super().__init__()
        self.key = nn.Linear(cfg.n_embed, cfg.n_embed, bias=cfg.bias)
        self.query = nn.Linear(cfg.n_embed, cfg.n_embed, bias=cfg.bias)
        self.value = nn.Linear(cfg.n_embed, cfg.n_embed, bias=cfg.bias)
        self.att_dropout = nn.Dropout(cfg.dropout)
        self.residual_dropout = nn.Dropout(cfg.dropout)

        self.proj = nn.Linear(cfg.n_embed, cfg.n_embed)

        # Local Variables
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        self.head_size = cfg.n_embed // cfg.n_heads
        self.num_heads = cfg.n_heads
        self.cfg = cfg

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
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
        v = self.split_heads(self.value(x))  # B x num_heads x T x head_size (hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.cfg.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = q @ k.transpose(-2, -1) * C**-0.5  # Scaled attention
            att = att.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.att_dropout(att)
            out = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out = self.combine_heads(out)
        return self.residual_dropout(self.proj(out))


class MLP(nn.Module):
    """
    Positional Feed Forward - With projection into the residual layer
    And a dropout layer
    For use in GPT2 Architecture - which uses GELU
    """

    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embed, 4 * cfg.n_embed, bias=True)  # cfg.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * cfg.n_embed, cfg.n_embed, bias=True)  # cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Block: Communication followed by Computation - With Residual Connections"""

    """The layer norm we apply is called pre-norm. Slighly different from the original paper"""

    def __init__(self, cfg):
        super().__init__()
        # We use the LayerNorm with the optional bias (GPT2 models dont use bias)
        self.ln1 = LayerNorm(cfg.n_embed, cfg.bias)
        self.sa = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.n_embed, cfg.bias)
        self.ffwd = MLP(cfg)

    def forward(self, x):
        # Apply the attention heads on the pre-norm'ed `x`
        x = x + self.sa(self.ln1(x))
        # B x T x C # Positional feed-forward on the pre-norm'ed `x`
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT2PretainedModel(BaseLanguageModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg.vocab_size, cfg.n_embed),
                wpe=nn.Embedding(cfg.block_size, cfg.n_embed),
                drop=nn.Dropout(cfg.dropout),
                h=nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)]),
                ln_f=LayerNorm(cfg.n_embed, bias=cfg.bias),
            )
        )
        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size, bias=cfg.bias)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # Reference: https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """Forward pass ==> Call the forward pass for the pre-trained model"""
        device = idx.device

        # idx, targets --> B x T (batch_size x block_size)
        B, T = idx.shape
        token_emb = self.transformer.wte(idx)  # B x T x C (n_embed)
        pos = torch.arange(0, T, dtype=torch.long, device=device)  # shape (t)
        position_emb = self.transformer.wpe(pos)  # (T x C)
        ## B x T x C (position_embedding gets broadcasted for each batch)
        x = self.transformer.drop(token_emb + position_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)  # B x T x C
        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)  # B x T x vocab_size
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None
        return logits, loss

    @staticmethod
    def fixed_params():
        """Return a dict of fixed params for the model"""
        return asdict(GPT2ModelArgs())

    @staticmethod
    def from_pretrained(cfg):
        model_type = cfg.pretrained_model
        logger.info(f"loading weights for pretrained gpt: {model_type}")

        # n_layers, n_heads and n_embed are determined from model_type
        # only dropout can be overridden see more notes below
        config_args = {
            "gpt2": dict(n_layers=12, n_heads=12, n_embed=768),  # 124M params
            "gpt2-medium": dict(n_layers=24, n_heads=16, n_embed=1024),  # 350M params
            "gpt2-large": dict(n_layers=36, n_heads=20, n_embed=1280),  # 774M params
            "gpt2-xl": dict(n_layers=48, n_heads=25, n_embed=1600),  # 1558M params
        }[model_type]

        logger.info("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints

        # create a from-scratch initialized `gpt2_pretrained` model
        cfg.update_hparams(**config_args)
        model = GPT2PretainedModel(cfg)

        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"  # nosec
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape  # nosec
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape  # nosec
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
