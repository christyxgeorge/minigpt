"""class for managing data from the tiny shakespeare dataset"""

import pandas as pd
import tiktoken
from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
from minigpt.loaders.loader_base import BaseDataset


class SpotifyMillionSongsData(BaseDataset):
    @property
    def name(self) -> str:
        """Return the dataset name"""
        return "Spotify Million Songs"

    def download(self):
        # download the spotify million songs dataset
        file_name = "spotify_millsongdata.csv"
        dataset_name = "spotify-million-song-dataset"
        self.download_kaggle("notshrirang", dataset_name, file_name)

    def get_metadata(self):
        """Get metadata to save alongwith train/val.bin"""
        return {"vocab_size": self.vocab_size}

    def load_metadata(self, metadata):
        """Load metadata saved alongwith train/val.bin"""
        self.enc = tiktoken.get_encoding("gpt2")
        self.vocab_size = metadata["vocab_size"]
        print("Loaded metadata from file")

    def load_token_ids(self) -> tuple[list[int], list[int]]:
        """Load Token IDs from Dataset"""
        if self.verbose:
            print("=" * 100)
            print(f"Loading Data [{self.filename}]...")
            print("=" * 100)

        df = pd.read_csv(self.filename, on_bad_lines="warn")
        text = df["text"].str.cat(sep="\n")

        # Split in train, val
        tv_split = int(0.9 * len(text))
        train_text = text[:tv_split]
        val_text = text[tv_split:]

        # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
        self.vocab_size = 50304

        # encode with tiktoken gpt2 bpe
        self.enc = tiktoken.get_encoding("gpt2")
        train_ids = self.enc.encode_ordinary(train_text)
        val_ids = self.enc.encode_ordinary(val_text)

        return train_ids, val_ids

    def encode(self, s) -> list[int]:
        """encode a string to a list of integers"""
        return self.enc.encode_ordinary(s)

    def decode(self, l) -> str:
        """decode a list of integers back to a string"""
        return "".join(self.enc.decode(l))
