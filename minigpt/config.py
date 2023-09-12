""""Hyperparameters"""
import hashlib
import json
import logging
import os
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime

import psutil
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

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    # Need type annotations for all fields. asdict returns ony fields with type annotations!
    vocab_size: int
    data_dir: pathlib.PosixPath  # Data directory for the input files
    out_dir: pathlib.PosixPath
    source: str
    verbose: bool = False

    model_id: int = 0  ## Model Version to use
    batch_size: int = 4  ## Number of independent sequences processed in parallel
    block_size: int = 8  ## Context Length for the prediction
    n_embed: int = 32  ## Dimension of the embedding
    n_layers: int = 4
    n_heads: int = 4
    max_iters: int = 3000
    learning_rate: float = 1e-3
    dropout: float = 0.2

    gradient_accumulation_steps: int = 5 * 8  # used to simulate larger batch sizes

    # optimizer settings
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

    # learning rate decay settings
    decay_lr: bool = False  # whether to decay the learning rate
    warmup_iters: int = 2000  # how many steps to warm up for
    lr_decay_iters: int = 600000  # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    compile: bool = False  ## use PyTorch 2.0 to compile the model to be faster
    profile: bool = False  # use pytorch profiler, or just simple benchmarking?

    device_type: str = "cpu"
    device = None

    use_ddp: bool = False
    local_rank: int = 0
    ddp_device: str = "gloo"  ## "xla" for TPU, "nccl" for CUDA, "gloo" for CPU

    eval_interval: int = 200
    eval_iters: int = 200
    eval_only = False  # if True, script exits right after the first eval

    wandb: str = "off"  # "on", "overwrite", "off"

    def __post_init__(self):
        # Setup Device and Evaluation Parameters
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_type + f":{self.local_rank}")
        self.ddp_device = "nccl" if torch.cuda.is_available() else "gloo"
        if self.device_type == "cpu":
            n_cores = psutil.cpu_count(logical=False)
            os.environ["OMP_NUM_THREADS"] = f"{n_cores}"
            # torch.set_num_interop_threads()  # Inter-op parallelism
            # torch.set_num_threads()  # Intra-op parallelism
        # logger.info(f"config = {self.dict()}, local_rank = {self.local_rank}")

    @property
    def wandb_log(self) -> bool:
        return self.wandb != "off"

    @property
    def model_name(self) -> str:
        return MODELS.get(self.model_id, BigramLanguageModel).__name__

    @property
    def device_count(self):
        if self.device_type == "cuda":
            n_gpus = torch.cuda.device_count()
            return n_gpus
        return psutil.cpu_count(logical=False)

    @property
    def run_id(self):
        if self.wandb == "overwrite":
            # Create ID for a specific model/source/config
            hash = hashlib.md5(
                json.dumps(self.dict(), sort_keys=True).encode("utf-8")
            ).hexdigest()  # nosec
            return f"{self.device_type}|{self.model_name.lower()}|{self.source.lower()}|{hash}"
        # Dont over-write, create unique id for each run...
        date_str = datetime.now().strftime("%d%b|%H%M%S.%f")[:-3]
        return f"{self.device_type}|{self.model_id}|{date_str}"

    def dict(self) -> dict[str, str]:
        x = {k: str(v) for k, v in asdict(self).items()}
        x["model_name"] = self.model_name
        return x

    def get_model(self) -> nn.Module:
        model_cls = MODELS.get(self.model_id, BigramLanguageModel)
        model_params = {"cfg": self}
        m = model_cls(**model_params)
        return m.to(self.device)

    @staticmethod
    def num_devices() -> int:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        if device_type == "cuda":
            n_gpus = torch.cuda.device_count()
            return n_gpus
        return psutil.cpu_count(logical=False)

    @staticmethod
    def modelname_fromid(model_id):
        return MODELS.get(model_id, BigramLanguageModel).__name__

    @staticmethod
    def default_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
