"""The Language Model Classes"""
import torch
import torch.nn as nn
from torch.nn import functional as F


class LanguageModelBase(nn.Module):
    def compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """Compute loss for the predicted logits v/s the targets"""
        if targets is None:
            loss = None
        else:
            # loss = F.cross_entropy(logits, targets) # Does not work because pytorch needs B * C * T for multi-dimensional array
            # So, Reshaping so that cross_entropy works
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return loss
