## Blocks needed by the models
import torch
import torch.nn as nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
    """Single Attention head With a specific head size"""

    def __init__(self, cfg, head_size):
        super().__init__()
        self.key = nn.Linear(cfg.n_embed, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embed, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embed, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        )  ## T x T

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # B x T x head_size (hs)
        q = self.query(x)  # B x T x head_size (hs)

        wei = q @ k.transpose(-2, -1) * C**-0.5  # Scaled attention
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)  # B x T x head_size (hs)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple Attention heads"""

    def __init__(self, cfg):
        super().__init__()
        head_size = cfg.n_embed // cfg.n_heads
        self.heads = nn.ModuleList([AttentionHead(cfg, head_size) for _ in range(cfg.n_heads)])

    def forward(self, x):
        # Concatenate on the `channel` dimension
        return torch.cat([h(x) for h in self.heads], dim=-1)


class MLP(nn.Module):
    """MLP for use in GPT2 Architecture - which uses GELU"""

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


class FeedForward(nn.Module):
    """Positional Feed Forward"""

    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg.n_embed, cfg.n_embed), nn.ReLU())

    def forward(self, x):
        return self.net(x)


class FeedForwardDropout(nn.Module):
    """
    Positional Feed Forward - With projection into the residual layer
    And a dropout layer
    """

    def __init__(self, cfg):
        super().__init__()
        # According to the `Attention is all you need` the inner FF layer has a multiplier of 4.
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embed, 4 * cfg.n_embed),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embed, cfg.n_embed),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class BatchNorm1D:
    """Batch Norm - from the makemore series"""

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.ones(dim)
        # buffers (trained with a running `momentum update` to keep track of running mean/variance)
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        """Calculate the forward pass"""
        if self.training:
            x_mean = x.mean(0, keepdim=True)  # batch mean
            x_var = x.var(0, keepdim=True)  # batch variance
        else:
            x_mean = self.running_mean
            x_var = self.running_var
        xhat = (x - x_mean) / torch.sqrt(x_var + self.eps)  # normalize to unit variance
        out = self.gamma * xhat + self.beta
        if self.training:
            with torch.no_grad():
                self.running_mean = (
                    1 - self.momentum
                ) * self.running_mean + self.momentum * x_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * x_var
        return out

    def parameters(self):
        return [self.gamma, self.beta]


class LayerNorm1D:
    """Layer Norm -- Adapted from the above BatchNorm1D"""

    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        # parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.ones(dim)

    def __call__(self, x):
        """Calculate the forward pass"""
        x_mean = x.mean(1, keepdim=True)  # layer mean (Here dim is 1)
        x_var = x.var(1, keepdim=True)  # layer variance
        xhat = (x - x_mean) / torch.sqrt(x_var + self.eps)  # normalize to unit variance
        out = self.gamma * xhat + self.beta
        return out

    def parameters(self):
        return [self.gamma, self.beta]


class LayerNorm(nn.Module):
    """
    LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False
    Needed in the GPT2 Architecture
    """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
