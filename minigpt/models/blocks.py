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
        out = self.dropout(self.proj(out))
        return out


class MLP(nn.Module):
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


class MultiHeadAttentionParallel(nn.Module):
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
        x = x + self.sa(self.ln1(x))
        # B x T x C # Positional feed-forward on the pre-norm'ed `x`
        x = x + self.ffwd(self.ln2(x))
        return x


class ResidualTransformerBlockParallel(nn.Module):
    """Transformer Block: Communication followed by Computation - With Residual Connections"""

    """The layer norm we apply is called pre-norm. Slighly different from the original paper"""

    def __init__(self, cfg):
        super().__init__()
        self.sa = MultiHeadAttentionParallel(cfg)
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
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
