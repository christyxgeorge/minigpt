""""Hyperparameters"""
from dataclasses import asdict, dataclass
from datetime import datetime

import torch
import torch.nn as nn
from minigpt.models import (
    GPTLanguageModelv1,
    GPTLanguageModelv2,
    GPTLanguageModelv3,
    GPTLanguageModelv4,
    GPTLanguageModelv5,
    GPTLanguageModelv6,
    GPTLanguageModelv7,
)
from minigpt.models.bigram import BigramLanguageModel

MODELS = {
    1: GPTLanguageModelv1,
    2: GPTLanguageModelv2,
    3: GPTLanguageModelv3,
    4: GPTLanguageModelv4,
    5: GPTLanguageModelv5,
    6: GPTLanguageModelv6,
    7: GPTLanguageModelv7,
}


@dataclass
class ModelConfig:
    # asdict returns ony fields with type annotations!
    vocab_size: int

    model_id: int = 0  ## Model Version to use
    batch_size: int = 4  ## Number of independent sequences processed in parallel
    block_size: int = 8  ## Context Length for the prediction
    n_embed: int = 32  ## Dimension of the embedding
    n_layers: int = 4
    n_heads: int = 4
    max_iters: int = 3000
    learning_rate: float = 1e-3
    dropout: float = 0.2

    device: str = "cpu"
    eval_interval: int = 200
    eval_iters: int = 200

    verbose: bool = False

    def __post_init__(self):
        # Setup Device and Evaluation Parameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def dict(self):
        x = {k: str(v) for k, v in asdict(self).items()}
        x["model_name"] = self.model_name()
        return x

    def get_model(self, tdata) -> nn.Module:
        model_cls = MODELS.get(self.model_id, BigramLanguageModel)
        model_params = {"cfg": self}
        m = model_cls(**model_params)
        return m

    def model_name(self):
        return MODELS.get(self.model_id, BigramLanguageModel).__name__

    def __repr__(self):
        return f"{self.device}|{self.model_id}|{datetime.now().isoformat().replace(':', '.')}"


# ------------------------------------------------------------
# Hyper-paramaeters -- v6 (Will definitely require GPU/TPU)
# ------------------------------------------------------------
# batch_size = 64  ## Number of independent sequences processed in parallel
# block_size = 256  ## Context Length for the prediction
# n_embed = 384  ## Dimension of the embedding
# n_layers = 6
# n_heads = 6
# max_iters = 5000
# learning_rate = 3e-4
# dropout = 0.2
