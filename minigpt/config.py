""""Hyperparameters"""
import hashlib
import json
import os
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
    device_type: str = "cpu"
    device = None
    ddp_device = "gloo"  ## "xla" for TPU, "nccl" for CUDA, "gloo" for CPU

    eval_interval: int = 200
    eval_iters: int = 200

    verbose: bool = False
    source: str = "s_char"  ## Text Source 's_char', 's_word', 't_stories'
    wandb_log: bool = False

    def __post_init__(self):
        # Setup Device and Evaluation Parameters
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_type)
        self.ddp_device = "nccl" if torch.cuda.is_available() else "gloo"
        if self.device_type == "cpu":
            n_cores = psutil.cpu_count(logical=False)
            os.environ["OMP_NUM_THREADS"] = f"{n_cores}"
            # torch.set_num_interop_threads()  # Inter-op parallelism
            # torch.set_num_threads()  # Intra-op parallelism
        self.print_device_info()

    @property
    def model_name(self) -> str:
        return MODELS.get(self.model_id, BigramLanguageModel).__name__

    @property
    def device_count(self):
        if self.device_type == "cuda":
            n_gpus = torch.cuda.device_count()
            return n_gpus
        return 1

    def dict(self):
        x = {k: str(v) for k, v in asdict(self).items()}
        x["model_name"] = self.model_name
        return x

    def get_model(self) -> nn.Module:
        model_cls = MODELS.get(self.model_id, BigramLanguageModel)
        model_params = {"cfg": self}
        m = model_cls(**model_params)
        return m.to(self.device)

    @staticmethod
    def modelname_fromid(model_id):
        return MODELS.get(model_id, BigramLanguageModel).__name__

    @staticmethod
    def default_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __repr__(self):
        overwrite = True
        if overwrite:
            # Create ID for a specific model/source/config
            hash = hashlib.md5(
                json.dumps(self.dict(), sort_keys=True).encode("utf-8")
            ).hexdigest()  # nosec
            return f"{self.device}|{self.model_name.lower()}|{self.source.lower()}|{hash}"
        # Create ID for each run... So that there
        return f"{self.device}|{self.model_id}|{datetime.now().isoformat().replace(':', '.')}"

    def print_device_info(self):
        if self.device_type == "cpu":
            print(torch.__config__.parallel_info())
        elif self.device_type == "cuda":
            # torch.cuda.device(0) # -> <torch.cuda.device at 0x7efce0b03be0>
            print("CUDA Devices Info:")
            num_devices = torch.cuda.device_count()
            print(f"  Count: {num_devices}")
            print(f"  Current Device = {torch.cuda.current_device()}")
            print(f"  bfloat16 supported: {torch.cuda.is_bf16_supported()}")
            for device_id in range(num_devices):
                print(f"  Device Name[{device_id}]: {torch.cuda.get_device_name(device_id)}")
            print("Memory Usage:")
            for device_id in range(num_devices):
                print(
                    f"  Allocated[{device_id}]:",
                    round(torch.cuda.memory_allocated(device_id) / 1024**3, 1),
                    "GB",
                )
                print(
                    f"  Cached[{device_id}]:   ",
                    round(torch.cuda.memory_reserved(device_id) / 1024**3, 1),
                    "GB",
                )


# ------------------------------------------------------------
# Final Hyper-paramaeters (Will definitely require GPU/TPU)
# ------------------------------------------------------------
# batch_size = 64  ## Number of independent sequences processed in parallel
# block_size = 256  ## Context Length for the prediction
# n_embed = 384  ## Dimension of the embedding
# n_layers = 6
# n_heads = 6
# max_iters = 5000
# learning_rate = 3e-4
# dropout = 0.2
