"""Base load to handle text data"""
from __future__ import annotations

import importlib
import logging
import os
import pathlib
import shutil
import zipfile
from abc import ABC, abstractmethod
from ast import Not

import numpy as np
import requests  # type: ignore
import torch
from tqdm.auto import tqdm  # Choose tqdm or tqdm_notebook based on env

logger = logging.getLogger(__name__)

LOADERS = {
    "s_char": "TinyShakespeareCharData",
    "s_word": "TinyShakespeareWordData",
    "m_song": "SpotifyMillionSongsData",
    "t_stories": "TinyStoriesData",
}
DEFAULT_LOADER = "s_char"

# Vocab Sizes:
LLAMA2_VOCAB_SIZE = 32000  # the Llama 2 tokenizer has 32K tokens
CUSTOM_VOCAB_SIZE = 2048
# GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
GPT2_VOCAB_SIZE = 50304


class BaseDataset(ABC):
    """Base dataset class for handling text data"""

    work_dir: pathlib.PosixPath
    verbose: bool = False
    train_data = None
    val_data = None
    prepared: bool = False

    def __init__(self, args):
        self.work_dir = args.work_dir
        self.verbose = args.verbose
        self.prepared = self.is_prepared()
        if self.verbose:
            prep_text = "is" if self.prepared else "is not"
            logger.info(
                f"Text Dataset {args.source} [{self.__class__.__name__}] {prep_text} prepared"
            )

    @classmethod
    def get_vocab_size(cls, source, model_id):
        """Get the vocab size based on the source"""
        current_package = importlib.import_module(__package__)
        cls_name = LOADERS.get(source) or "TextDataset"
        cls = getattr(current_package, cls_name)
        return cls.get_vocab_size(source, model_id)

    @staticmethod
    def default_loader():
        return DEFAULT_LOADER

    @staticmethod
    def loaders():
        return list(LOADERS.keys())

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the dataset name"""

    @abstractmethod
    def is_prepared(self) -> bool:
        """Check if the data(bin_files) have been prepared"""

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
    def prepare(self, force=False):
        """Create binary token files"""

    @abstractmethod
    def load_data(self):
        """Load Data from file"""

    @abstractmethod
    def get_batch(self, cfg, split) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of data"""

    @staticmethod
    def get_loader(args, load=False) -> BaseDataset:
        current_package = importlib.import_module(__package__)
        cls_name = LOADERS.get(args.source)
        if cls_name:
            cls = getattr(current_package, cls_name)
            loader = cls(args)
            if load:
                loader.load_data()
            return loader
        else:
            error_msg = f"Unknown Dataset Source: {args.source} - Use one of {LOADERS.keys()}"
            logger.warn(error_msg)
            raise ValueError(error_msg)

    # Common utility
    def download_file(self, url: str, file_name: pathlib.PosixPath, chunk_size=1024):
        """Helper function to download a file from a given url"""
        if not file_name.exists():
            resp = requests.get(url, stream=True)  # nosec
            total = int(resp.headers.get("content-length", 0))
            with open(file_name, "wb") as file, tqdm(
                desc=str(file_name),
                total=total,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in resp.iter_content(chunk_size=chunk_size):
                    size = file.write(data)
                    bar.update(size)

    def download_kaggle(self, user: str, dataset_name: str, file_name: pathlib.PosixPath):
        # download the spotify million songs dataset
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore

        if not file_name.exists():
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(f"{user}/{dataset_name}", path=self.work_dir)

            zip_file = self.work_dir / f"{dataset_name}.zip"
            print(f"Unpacking {zip_file}...")
            with zipfile.ZipFile(zip_file) as z:
                with z.open(str(file_name)) as zf, open(file_name, "wb") as f:
                    shutil.copyfileobj(zf, f)

            os.unlink(zip_file)
        else:
            print("File {file_name} exists, Not downloading")
