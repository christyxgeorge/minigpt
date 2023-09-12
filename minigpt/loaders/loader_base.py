"""Base load to handle text data"""
from __future__ import annotations

import importlib
import logging
import os
import pathlib
import pickle  # nosec
import shutil
import zipfile
from abc import ABC, abstractmethod

import numpy as np
import requests  # type: ignore
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

LOADERS = {
    "s_char": {"cls": "TinyShakespeareCharData", "filename": "tiny_shakespeare.txt"},
    "s_word": {"cls": "TinyShakespeareWordData", "filename": "tiny_shakespeare.txt"},
    "m_song": {"cls": "SpotifyMillionSongsData", "filename": "spotify_millsongdata.csv"},
    "t_stories": {"cls": "TinyStoriesData", "filename": "spotify_millsongdata.csv"},
}
DEFAULT_LOADER = "s_char"


class BaseDataset(ABC):
    """Base dataset class for handling text data"""

    data_dir: pathlib.PosixPath
    bin_dir: pathlib.PosixPath  # Location of train_bin, val_bin
    filename: str
    verbose: bool = False
    train_data = None
    val_data = None
    prepared: bool = False

    # Prepared binary files.
    train_bin: str = "train.bin"  # train file, if prepared
    val_bin: str = "val.bin"

    def __init__(self, src, data_dir, verbose=False):
        self.data_dir = data_dir
        cls_info = LOADERS.get(src)
        filename = cls_info["filename"]
        self.filename = data_dir / filename
        self.bin_dir = data_dir / src
        self.bin_dir.mkdir(parents=True, exist_ok=True)  # Create bin_dir if not available
        self.metadata_file = self.bin_dir / "metadata.pkl"
        self.verbose = verbose
        bin_files_exist = (self.bin_dir / self.train_bin).exists() and (
            self.bin_dir / self.val_bin
        ).exists()
        self.prepared = True if bin_files_exist else False

    @staticmethod
    def default_loader():
        return DEFAULT_LOADER

    @staticmethod
    def loaders():
        return LOADERS.keys()

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the dataset name"""

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
    def download(self):
        """download the dataset"""

    @abstractmethod
    def get_metadata(self):
        """Get metadata to save alongwith train/val.bin"""

    @abstractmethod
    def load_metadata(self, metadata):
        """Load metadata saved alongwith train/val.bin"""

    def prepare(self):
        """Create train.bin and val.bin files"""
        self.download()  # Download the file, if not available
        train_ids, val_ids = self.load_token_ids()
        # export to bin files
        train_ids = np.array(train_ids, dtype=np.int32)
        val_ids = np.array(val_ids, dtype=np.int32)
        train_ids.tofile(self.bin_dir / self.train_bin)
        val_ids.tofile(self.bin_dir / self.val_bin)
        metadata = self.get_metadata()
        with open(self.metadata_file, "wb") as pklfile:
            pickle.dump(metadata, pklfile, protocol=pickle.HIGHEST_PROTOCOL)
        self.prepared = True
        return True

    def load_data(self):
        """Load Data from file"""
        if self.prepared:
            # poor man's data loader [Load from train.bin, val.bin]
            self.train_data = np.memmap(self.bin_dir / self.train_bin, dtype=np.int32, mode="r")
            self.val_data = np.memmap(self.bin_dir / self.val_bin, dtype=np.int32, mode="r")
            with open(self.metadata_file, "rb") as pklfile:
                metadata = pickle.load(pklfile)  # nosec
                self.load_metadata(metadata)
        else:
            train_ids, val_ids = self.load_token_ids()
            self.train_data = torch.tensor(train_ids, dtype=torch.long)
            self.val_data = torch.tensor(val_ids, dtype=torch.long)

        # Create data tensor!
        if self.verbose:
            print(f"    Text Length = {len(self.text)}")
            print(f"    Vocab Size = {self.vocab_size} => [{','.join(self.vocab_chars)}]")
            print(
                f"    Training len = {len(self.train_data)}, Validation len = {self.len(self.val_data)}"
            )
            print("=" * 100)

    @staticmethod
    def get_loader(src, data_dir, verbose=False, load=False) -> BaseDataset:
        current_package = importlib.import_module(__package__)
        cls_info = LOADERS.get(src)
        if cls_info:
            cls = getattr(current_package, cls_info["cls"])
            loader = cls(src, data_dir, verbose=verbose)
            if load:
                loader.load_data()
            return loader
        else:
            print(f"Unknown Dataset Source: {src}")
            raise ValueError(f"Unknown Dataset Source: {src} - Should be one of {LOADERS.keys()}")

    def get_batch(self, cfg, split) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch Size = Number of sequences being processed in parallel!"""
        return self.get_batch_bin(cfg, split) if self.prepared else self.get_batch_data(cfg, split)

    def get_batch_data(self, cfg, split) -> tuple[torch.Tensor, torch.Tensor]:
        """Load from the created data tensors"""
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
        # Create data from numpy npmemmap
        x = torch.stack(
            [torch.from_numpy((data[i : i + cfg.block_size]).astype(np.int64)) for i in ix]
        )
        y = torch.stack(
            [torch.from_numpy((data[i + 1 : i + 1 + cfg.block_size]).astype(np.int64)) for i in ix]
        )
        if cfg.device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(cfg.device, non_blocking=True), y.pin_memory().to(
                cfg.device, non_blocking=True
            )
        else:
            x, y = x.to(cfg.device), y.to(cfg.device)
        return x, y

    # Common utility
    def download_file(self, url: str, fname: str, chunk_size=1024):
        """Helper function to download a file from a given url"""
        resp = requests.get(url, stream=True)  # nosec
        total = int(resp.headers.get("content-length", 0))
        with open(fname, "wb") as file, tqdm(
            desc=fname,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)

    def download_kaggle(self, user, dataset_name, file_name):
        # download the spotify million songs dataset
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore

        input_file_path = self.data_dir / file_name
        if not input_file_path.exists():
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(f"{user}/{dataset_name}", path=self.data_dir)

            zip_file = self.data_dir / f"{dataset_name}.zip"
            print(f"Unpacking {zip_file}...")
            with zipfile.ZipFile(zip_file) as z:
                with z.open(file_name) as zf, open(input_file_path, "wb") as f:
                    shutil.copyfileobj(zf, f)

            os.unlink(zip_file)

    def download_url(self, filename, data_url):
        # download the file from the URL
        input_file_path = self.data_dir / filename
        if not input_file_path.exists():
            with open(input_file_path, "w") as f:
                f.write(requests.get(data_url).text)  # nosec
