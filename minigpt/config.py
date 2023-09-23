""""Hyperparameters"""
import hashlib
import json
import logging
import os
import pathlib
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Optional

import psutil
import semver
import torch
from minigpt.loaders.base_dataset import BaseDataset
from minigpt.models.base_model import BaseLanguageModel

# Note: P100 does not support Mixed Precision...
# Check https://www.xcelerit.com/computing-benchmarks/insights/benchmarks-deep-learning-nvidia-p100-vs-v100-gpu/
F16_TFLOPS = {"TPU v2": 100, "A100": 312, "V100": 112, "P100": 18.7, "T4": 65}

logger = logging.getLogger(__name__)


@dataclass
class CommonConfig:
    """Common Configs for all models"""

    work_dir: pathlib.PosixPath  # Working directory
    source: str
    model_id: str = "bg"  ## Model Version to use
    vocab_source: str = "llama2"  # Tokenizer needs to be used for the tiny stories dataset
    vocab_size: int = 0  # Needs to be initialized from the dataset
    verbose: bool = False

    def __post_init__(self):
        # Vocab_size and Vocab source
        self.vocab_size = BaseDataset.get_vocab_size(self.source, self.vocab_source)

    @property
    def common_params(self) -> dict[str, Any]:
        return dict(
            work_dir=self.work_dir,
            source=self.source,
            vocab_size=self.vocab_size,
            vocab_source=self.vocab_source,
            model_id=self.model_id,
            verbose=self.verbose,
        )


@dataclass
class TrainerConfig(CommonConfig):
    """Trainer Configs for all models"""

    # Need type annotations for all fields. asdict returns ony fields with type annotations!
    batch_size: int = 4  ## Number of independent sequences processed in parallel

    # Hyper-parameters
    block_size: int = 8  ## Context Length for the prediction
    n_embed: int = 32  ## Dimension of the embedding
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.2  # for pretraining 0 is good, for finetuning try 0.1+
    bias: bool = True

    # settings for learning rate decay, gradient accumulation and max iterations
    learning_rate: float = 1e-3
    decay_lr: bool = False  # whether to decay the learning rate
    lr_decay_iters: int = 600000  # should be ~= max_iters per Chinchilla
    min_lr: float = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    warmup_iters: int = 2000  # how many steps to warm up for before starting lr decay
    max_iters: int = 3000
    gradient_accumulation_steps: int = 5 * 8  # used to simulate larger batch sizes

    # optimizer settings
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    norm_eps: float = 1e-5  # TODO: Check where it is used!
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0

    # other device related config!
    use_ddp: bool = False
    local_rank: int = 0
    world_size: int = 1
    ddp_device: str = "gloo"  ## "xla" for TPU, "nccl" for CUDA, "gloo" for CPU
    ddp_port: int = 12355  # Port to be used for DDP
    device_type: str = "cpu"
    device = None  # torch.device

    # How often do we evaluate (eval_intervals) & how many batches do we process (eval_iters)
    eval_interval: int = 200
    eval_iters: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval

    wandb: str = "off"  # "on", "overwrite", "off"
    pretrained_model: str = "gpt-2"  ## If we need to load a pretrained model
    resume: bool = False  # If we want to resume the previous training
    compile: bool = False  ## use PyTorch 2.0 to compile the model to be faster
    profile: bool = False  # TODO: use pytorch profiler, or just simple benchmarking?
    log_interval: int = 40  # How often do we want to write to the training log

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

        self.setup_gradient_accumulation()

    def setup_gradient_accumulation(self):
        """Setup gradient accumulation steps"""
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        initial_grad_accum_steps = self.gradient_accumulation_steps
        # assert self.gradient_accumulation_steps % self.world_size == 0  # nosec
        ## can also do math.ceil(accum_steps / world_size) instead of int div.
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
        return BaseLanguageModel.model_name(self.model_id)

    @property
    def device_count(self):
        if self.device_type == "cuda":
            n_gpus = torch.cuda.device_count()
            return n_gpus
        return psutil.cpu_count(logical=False)

    @property
    def hparams(self) -> dict[str, float]:
        """Returns a dict of hyper-params to be stored in the model checkpoint"""
        return dict(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_embed=self.n_embed,
            block_size=self.block_size,
            bias=self.bias,
            vocab_size=self.vocab_size,
            dropout=self.dropout,
        )

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

    def update_hparams(self, **hparams):
        # sample.__dict__.update(hparams)
        master_process = self.local_rank == 0
        for key, value in hparams.items():
            if hasattr(self, key):
                curr_value = getattr(self, key)
                if master_process and curr_value != value:
                    logger.info(f"Hyperparameter {key} Changed from {curr_value} to {value}")
                setattr(self, key, value)

    @staticmethod
    def num_devices() -> int:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        if device_type == "cuda":
            n_gpus = torch.cuda.device_count()
            return n_gpus
        return psutil.cpu_count(logical=False)

    @staticmethod
    def default_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class GeneratorConfig(CommonConfig):
    """Config for generating text"""

    tokens: int = 1000
    start_with: Optional[str] = None
    temperature: float = 0.7
