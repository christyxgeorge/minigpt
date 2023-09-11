"""class for managing data from the tiny stories dataset"""
import glob
import json
import os
from pathlib import Path

from minigpt.loaders.loader_base import BaseDataset

# This is adapated from github/llama2.c/tinystories.py
# https://github.com/karpathy/llama2.c/blob/master/tinystories.py


class TinyStoriesData(BaseDataset):
    def download(self):
        """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
        os.makedirs(self.data_dir, exist_ok=True)

        # download the TinyStories dataset, unless it's already downloaded
        data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
        data_filename = self.data_dir / "TinyStories_all_data.tar.gz"
        if not data_filename.exists():
            print(f"Downloading {data_url} to {data_filename}...")
            self.download_file(data_url, data_filename)
        else:
            print(f"{data_filename} already exists, skipping download...")

        # unpack the tar.gz file into all the data shards (json files)
        data_dir = self.data_dir / "TinyStories_all_data"
        if not data_dir.exists():
            os.makedirs(data_dir, exist_ok=True)
            print(f"Unpacking {data_filename}...")
            os.system(f"tar -xzf {data_filename} -C {data_dir}")  # nosec
        else:
            print(f"{data_dir} already exists, skipping unpacking...")

        # print a single example just for debugging and such
        shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        with open(shard_filenames[0], "r") as f:
            data = json.load(f)
        print("Download done.")
        print(f"Number of shards: {len(shard_filenames)}")
        print(f"Example story:\n{data[0]}")

    def get_metadata(self):
        """Get metadata to save alongwith train/val.bin"""
        return {"vocab_size": self.vocab_size}

    def load_metadata(self, metadata):
        """Load metadata saved alongwith train/val.bin"""
        self.vocab_size = metadata["vocab_size"]
