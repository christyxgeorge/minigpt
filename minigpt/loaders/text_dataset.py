"""Base load to handle text data"""
from __future__ import annotations

import logging
import pickle  # nosec
from abc import ABC, abstractmethod

import numpy as np
import torch
from minigpt.loaders.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class TextDataset(BaseDataset):
    """
    Text dataset class for handling text data from a single file
    """

    filename: str

    # Prepared binary files.
    train_bin: str = "train.bin"  # train file, if prepared
    val_bin: str = "val.bin"

    def __init__(self, args, filename):
        self.filename = args.work_dir / filename

        # Setup internal variables before calling super().__init__()
        super().__init__(args)

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the dataset name"""

    @abstractmethod
    def is_prepared(self) -> bool:
        """Check if the data(bin_files) have been prepared"""

    @abstractmethod
    def load_token_ids(self) -> tuple[list[int], list[int]]:
        """Load Token IDs from Dataset"""

    @abstractmethod
    def encode(self, s) -> list[int]:
        """encode a string to a list of integers"""

    @abstractmethod
    def decode(self, l) -> str:
        """decode a list of integers back to a string"""

    @abstractmethod
    def download(self, force=False):
        """download the dataset"""

    @abstractmethod
    def get_metadata(self):
        """Get metadata to save alongwith train/val.bin"""

    @abstractmethod
    def load_metadata(self, metadata):
        """Load metadata saved alongwith train/val.bin"""

    def prepare(self, force=False):
        """Create train.bin and val.bin files"""
        self.download(force=force)  # Download the file, if not available
        train_ids, val_ids = self.load_token_ids()
        # export to bin files
        train_ids = np.array(train_ids, dtype=np.int32)
        val_ids = np.array(val_ids, dtype=np.int32)
        train_ids.tofile(self.work_dir / self.train_bin)
        val_ids.tofile(self.work_dir / self.val_bin)
        metadata = self.get_metadata()
        with open(self.metadata_file, "wb") as pklfile:
            pickle.dump(metadata, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        self.prepared = True
        return True

    def load_data(self):
        """Load Data from file"""
        if self.prepared:
            # poor man's data loader [Load from train.bin, val.bin]
            self.train_data = np.memmap(self.work_dir / self.train_bin, dtype=np.int32, mode="r")
            self.val_data = np.memmap(self.work_dir / self.val_bin, dtype=np.int32, mode="r")
            with open(self.metadata_file, "rb") as pklfile:
                metadata = pickle.load(pklfile)  # nosec
                self.load_metadata(metadata)
        else:
            self.download()  # Check and download the file
            train_ids, val_ids = self.load_token_ids()
            self.train_data = torch.tensor(train_ids, dtype=torch.long)
            self.val_data = torch.tensor(val_ids, dtype=torch.long)

        # Create data tensor!
        if self.verbose:
            print(f"    Vocab Size = {self.vocab_size}")
            print(
                f"    Training len = {len(self.train_data)}, Validation len = {len(self.val_data)}"
            )
            print("=" * 100)

    def get_batch(self, cfg, split) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch Size = Number of sequences being processed in parallel!"""
        return self.get_batch_bin(cfg, split) if self.prepared else self.get_batch_data(cfg, split)

    def get_batch_data(self, cfg, split) -> tuple[torch.Tensor, torch.Tensor]:
        """Load from the created data tensors - Inefficient compared to preparing numpy files"""
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
        """Load from the numpy memmap of train/val.bin"""
        """Batch Size = Number of sequences being processed in parallel!"""

        assert self.train_data is not None and self.val_data is not None  # nosec
        data = self.train_data if split == "train" else self.val_data
        # Generate `batch_size` random offsets
        ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
        # Create data from numpy np.memmap
        x = torch.stack(
            [torch.from_numpy((data[i : i + cfg.block_size]).astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [torch.from_numpy((data[i + 1 : i + 1 + cfg.block_size]).astype(np.int64)) for i in ix]
        )
        if cfg.device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x = x.pin_memory().to(cfg.device, non_blocking=True)
            y = y.pin_memory().to(cfg.device, non_blocking=True)
        else:
            x, y = x.to(cfg.device), y.to(cfg.device)
        return x, y
