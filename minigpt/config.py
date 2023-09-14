""""Hyperparameters"""
import hashlib
import json
import logging
import os
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime

import psutil
import semver
import torch
import torch.nn as nn
from minigpt.models import (
    GPTLanguageModelLlama2,
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
    "b": BigramLanguageModel,
    "m1": GPTLanguageModelv1,
    "m2": GPTLanguageModelv2,
    "m3": GPTLanguageModelv3,
    "m4": GPTLanguageModelv4,
    "m5": GPTLanguageModelv5,
    "m6": GPTLanguageModelv6,
    "m7": GPTLanguageModelv7,
    "l2": GPTLanguageModelLlama2,
}

# Note: P100 does not support Mixed Precision...
# Check https://www.xcelerit.com/computing-benchmarks/insights/benchmarks-deep-learning-nvidia-p100-vs-v100-gpu/
F16_TFLOPS = {"TPU v2": 100, "A100": 312, "V100": 112, "P100": 18.7, "T4": 65}

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    # Need type annotations for all fields. asdict returns ony fields with type annotations!
    vocab_size: int
    work_dir: pathlib.PosixPath  # Working directory
    source: str
    verbose: bool = False

    model_id: str = "b"  ## Model Version to use
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

    use_ddp: bool = False
    local_rank: int = 0
    world_size: int = 0
    ddp_device: str = "gloo"  ## "xla" for TPU, "nccl" for CUDA, "gloo" for CPU
    device_type: str = "cpu"
    device = None

    eval_interval: int = 200
    eval_iters: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval

    wandb: str = "off"  # "on", "overwrite", "off"
    vocab_source: str = "llama2"  # used while tokenizing `tiny stories` dataset
    resume: bool = False  # If we want to resume the previous training

    def __post_init__(self):
        # Setup Device and Evaluation Parameters
        if torch.cuda.is_available():
            self.device_type = "cuda"
            self.ddp_device = "nccl"
            if torch.cuda.device_count() > 1:
                self.use_ddp = True
        else:
            self.device_type = "cpu"
            self.ddp_device = "gloo"

        ver = semver.Version.parse(torch.__version__)
        self.compile = True if self.compile and ver.major >= 2 else False
        if self.device_type == "cpu":
            self.device = torch.device(self.device_type)
        else:
            self.device = torch.device(self.device_type + f":{self.local_rank}")
        if self.device_type == "cpu":
            n_cores = psutil.cpu_count(logical=False)
            os.environ["OMP_NUM_THREADS"] = f"{n_cores}"
            # torch.set_num_interop_threads()  # Inter-op parallelism
            # torch.set_num_threads()  # Intra-op parallelism

        # Setup cuda device
        if self.device_type == "cuda":
            device = f"{self.device_type}:{self.local_rank}"
            torch.cuda.set_device(device)
            logger.info(f"[{self.local_rank}] Setting CUDA device to {device}")

        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        initial_grad_accum_steps = self.gradient_accumulation_steps
        # assert self.gradient_accumulation_steps % self.world_size == 0  # nosec
        self.gradient_accumulation_steps //= self.world_size
        tokens_per_iter = (
            self.gradient_accumulation_steps * self.world_size * self.batch_size * self.block_size
        )
        if self.local_rank == 0:  # Master process
            if self.gradient_accumulation_steps != initial_grad_accum_steps:  # Has changed
                logger.info(
                    f"world size = {self.world_size}, "
                    f"gradient_accumulation_steps changed from {initial_grad_accum_steps} to {self.gradient_accumulation_steps} * {self.world_size}, "
                )
            else:
                logger.info(
                    f"world size = {self.world_size}, "
                    f"gradient_accumulation_steps = {self.gradient_accumulation_steps}, "
                )
                logger.info(f"tokens per iteration [per process] will be: {tokens_per_iter:,}")

    @property
    def is_wandb_enabled(self) -> bool:
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
