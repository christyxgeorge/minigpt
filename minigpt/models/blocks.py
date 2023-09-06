## Blocks needed by the models
import torch
import torch.nn as nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
    """Single Attention head"""

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


class AttentionHeadDropout(nn.Module):
    """Single Attention head"""

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


class MultiHeadAttention(nn.Module):
    """Multiple Attention heads"""

    def __init__(self, cfg):
        super().__init__()
        head_size = cfg.n_embed // cfg.n_heads
        self.heads = nn.ModuleList([AttentionHead(cfg, head_size) for _ in range(cfg.n_heads)])

    def forward(self, x):
        # Concatenate on the `channel` dimension
        return torch.cat([h(x) for h in self.heads], dim=-1)


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
        out = self.proj(out)
        return out


class MultiHeadAttentionParallel(nn.Module):
    """Multiple Attention heads running in parallel - And with projection, dropout"""

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
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """Positional Feed Forward"""

    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cfg.n_embed, cfg.n_embed), nn.ReLU())

    def forward(self, x):
        return self.net(x)


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
        x = x + self.sa(self.ln1(x) if self.ln1 else x)
        # B x T x C # Positional feed-forward on the pre-norm'ed `x`
        x = x + self.ffwd(self.ln2(x) if self.ln2 else x)
        return x


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
