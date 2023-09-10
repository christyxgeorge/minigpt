"""Base load to handle text data"""
from __future__ import annotations

import logging
import pathlib
from abc import ABC, abstractmethod

import numpy as np
import torch

logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    """Base dataset class for handling text data"""

    data_dir: pathlib.PosixPath
    filename: str
    verbose: bool = False
    train_data = None
    val_data = None
    prepared: bool = False

    # Prepared binary files.
    train_bin: str = "train.bin"  # train file, if prepared
    val_bin: str = "val.bin"

    def __init__(self, data_dir, filename, verbose=False):
        self.data_dir = data_dir
        self.filename = data_dir / filename
        self.verbose = verbose
        bin_files_exist = (data_dir / self.train_bin).exists() and (
            data_dir / self.val_bin
        ).exists()
        self.prepared = True if bin_files_exist else False
        if not self.prepared:
            logger.info("bin files not found, loading data!")
            self.load_data()

    @abstractmethod
    def load_token_ids(self) -> tuple[list[int], list[int]]:
        """Load Token IDs from Dataset"""

    @abstractmethod
    def encode(self, s) -> list[int]:
        """encode a string to a list of integers"""

    @abstractmethod
    def decode(self, l) -> str:
        """decode a list of integers back to a string"""

    def prepare(self):
        """Prepare train.bin and val.bin files"""
        train_ids, val_ids = self.load_token_ids()
        # export to bin files
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)
        train_ids.tofile(self.data_dir, "train.bin")
        val_ids.tofile(self.data_dir, "val.bin")
        self.prepared = True
        return True

    def load_data(self):
        """Load Data from file"""
        if self.prepared:
            # poor man's data loader [Load from train.bin, val.bin]
            self.train_data = np.memmap(self.data_dir / self.train_bin, dtype=np.uint16, mode="r")
            self.val_data = np.memmap(self.data_dir / self.val_bin, dtype=np.uint16, mode="r")
        else:
            train_ids, val_ids = self.load_token_ids()
            self.train_data = torch.tensor(train_ids, dtype=torch.long)
            self.val_data = torch.tensor(val_ids, dtype=torch.long)
        if self.verbose:
            print(f"    Text Length = {len(self.text)}")
            print(f"    Vocab Size = {self.vocab_size} => [{','.join(self.vocab_chars)}]")
            print(
                f"    Training len = {len(self.train_data)}, Validation len = {self.len(self.val_data)}"
            )
            print("=" * 100)

    @classmethod
    def get_loader(cls, type, root_dir, verbose=False) -> BaseDataset | None:
        if type == "s_char":
            from .shakespeare_char import TinyShakespeareCharData

            return TinyShakespeareCharData(root_dir, verbose=verbose)
        elif type == "s_word":
            from .shakespeare_subword import TinyShakespeareWordData

            return TinyShakespeareWordData(root_dir, verbose=verbose)
        else:
            print(f"Unknown TextData Type: {type}")
            raise ValueError(f"Unknown TextData Type: {type} - Should be `s_char` or `s_word`")

    def get_batch(self, cfg, split) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch Size = Number of sequences being processed in parallel!"""
        return self.get_batch_data(cfg, split) if self.prepared else self.get_batch_bin(cfg, split)

    def get_batch_data(self, cfg, split) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch Size = Number of sequences being processed in parallel!"""
        assert self.train_data is not None and self.val_data is not None  # nosec
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(
            len(data) - cfg.block_size, (cfg.batch_size,)
        )  # Generate `batch_size` random offsets
        x = torch.stack(
            [data[i : i + cfg.block_size] for i in ix]
        )  # Each sample is stacked as a row!
        y = torch.stack([data[i + 1 : i + cfg.block_size + 1] for i in ix])
        x, y = x.to(cfg.device), y.to(cfg.device)
        return x, y

    def get_batch_bin(self, cfg, split) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch Size = Number of sequences being processed in parallel!"""

        # poor man's data loader [Need to enable this when we use train.bin, val.bin]
        # data_dir = os.path.join('data', dataset)
        # train_data = np.memmap(os.path.join(data_dir, self.train_bin), dtype=np.uint16, mode='r')
        # val_data = np.memmap(os.path.join(data_dir, self.val_bin), dtype=np.uint16, mode='r')

        assert self.train_data is not None and self.val_data is not None  # nosec
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(
            len(data) - cfg.block_size, (cfg.batch_size,)
        )  # Generate `batch_size` random offsets
        x = torch.stack(
            [data[i : i + cfg.block_size] for i in ix]
        )  # Each sample is stacked as a row!
        y = torch.stack([data[i + 1 : i + cfg.block_size + 1] for i in ix])
        if cfg.device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(cfg.device, non_blocking=True), y.pin_memory().to(
                cfg.device, non_blocking=True
            )
        else:
            x, y = x.to(cfg.device), y.to(cfg.device)
        return x, y
