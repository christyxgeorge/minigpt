"""class for managing data from the tiny shakespeare dataset"""

import pandas as pd
import tiktoken
from minigpt.loaders.text_dataset import TextDataset

# GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
GPT2_VOCAB_SIZE = 50304


class SpotifyMillionSongsData(TextDataset):
    def __init__(self, args):
        super().__init__(args, "spotify_millsongdata.csv")
        self.vocab_size = GPT2_VOCAB_SIZE
        self.enc = tiktoken.get_encoding("gpt2")

    @classmethod
    def get_vocab_size(cls, _source, _vocab_soure: str | None = None):
        """Get the vocab size based on the source"""
        return GPT2_VOCAB_SIZE

    @property
    def name(self) -> str:
        """Return the dataset name"""
        return "Spotify Million Songs"

    def is_prepared(self) -> bool:
        """Check if the data(bin_files) have been prepared"""
        bin_files_exist = (self.work_dir / self.train_bin).exists() and (
            self.work_dir / self.val_bin
        ).exists()
        return bin_files_exist

    def download(self, force=False):
        # download the spotify million songs dataset (git lfs link)
        data_url = "https://media.githubusercontent.com/media/christyxgeorge/datasets/main/spotify_millsongdata.csv"
        self.download_file(data_url, self.filename)

    def download_kaggle(self):
        # download the spotify million songs dataset
        dataset_name = "spotify-million-song-dataset"
        file_name = self.bin_dir / dataset_name
        self.download_kaggle("notshrirang", dataset_name, str(file_name))

    def get_token_ids(self) -> tuple[list[int], list[int]]:
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
