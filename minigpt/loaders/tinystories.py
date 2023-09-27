"""class for managing data from the tiny stories dataset"""
import argparse
import glob
import json
import logging
import os
import random
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import sentencepiece as spm
import tiktoken
import torch
import torch.distributed as dist
from minigpt.loaders.base_dataset import (
    CUSTOM_VOCAB_SIZE,
    GPT2_VOCAB_SIZE,
    LLAMA2_VOCAB_SIZE,
    BaseDataset,
)
from minigpt.loaders.tokenizer import Tokenizer
from minigpt.models.base_model import BaseLanguageModel
from tqdm.auto import tqdm  # Choose tqdm or tqdm_notebook based on env

logger = logging.getLogger(__name__)
NUM_DATA_FILES = 50

# This is adapated from github/llama2.c/tinystories.py
# https://github.com/karpathy/llama2.c/blob/master/tinystories.py


class TinyStoriesData(BaseDataset):
    def __init__(self, args):
        self.iter_batches = {"train": None, "val": None}
        self.iter_val_batches = None
        if args.model_id == "l2":
            self.vocab_size = LLAMA2_VOCAB_SIZE
            self.tokenizer_path = "llama2"
        elif args.model_id == "g2":
            self.vocab_size = GPT2_VOCAB_SIZE
            self.tokenizer_path = "gpt2"
        else:
            self.vocab_size = CUSTOM_VOCAB_SIZE
            self.tokenizer_path = f"tok{self.vocab_size}"

        # .bin files will be saved into the tokenizer directory, create it once here
        self.bin_dir = args.work_dir / self.tokenizer_path
        os.makedirs(self.bin_dir, exist_ok=True)

        # Setup internal variables before calling super().__init__()
        super().__init__(args)

    @classmethod
    def get_vocab_size(cls, source, model_id):
        """Get the vocab size based on the source"""
        if model_id == "l2":
            return LLAMA2_VOCAB_SIZE
        elif model_id == "g2":
            return GPT2_VOCAB_SIZE
        else:
            return CUSTOM_VOCAB_SIZE

    @property
    def name(self) -> str:
        """Return the dataset name"""
        return "Tiny Stories"

    def is_prepared(self) -> bool:
        """Check if the data(bin_files) have been prepared"""
        data_glob = os.path.join(self.bin_dir, "data*.bin")
        files = glob.glob(data_glob)
        logger.info(f"glob = {data_glob}, Files = {len(files)} / {NUM_DATA_FILES}")
        return len(files) == NUM_DATA_FILES

    def download(self, force=False):
        """Downloads the TinyStories dataset to work_dir"""
        start_time = time.time()
        os.makedirs(self.work_dir, exist_ok=True)

        # download the TinyStories dataset, unless it's already downloaded
        data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
        data_filename = self.work_dir / "TinyStories_all_data.tar.gz"

        # unpack the tar.gz file into all the data shards (json files)
        data_dir = self.work_dir / "TinyStories_all_data"
        if not data_dir.exists():
            if not data_filename.exists():
                logger.info(f"Downloading {data_url} to {data_filename}...")
                self.download_file(data_url, data_filename)
            else:
                logger.info(f"{data_filename} already exists, skipping download...")
            data_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Unpacking {data_filename}...")
            # os.system(f"tar -xzf {data_filename} -C {data_dir}")  # nosec
            with tarfile.open(name=data_filename, mode="r:gz") as tar:
                for member in tqdm(iterable=tar.getmembers(), total=len(tar.getmembers())):
                    tar.extract(member=member, path=data_dir)
            os.remove(data_filename)
            logger.info(f"Deleted {data_filename}")
        else:
            logger.info(f"{data_dir} already exists, skipping unpacking...")

        # print a single example just for debugging and such
        shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        with open(shard_filenames[0], "r") as f:
            data = json.load(f)
        logger.info(f"Number of shards: {len(shard_filenames)}\nExample story:\n{data[0]}")
        elapsed_time = time.time() - start_time
        logger.info(f"Download Done, Time taken = {elapsed_time:.3f} secs")

    def prepare(self, force=False):
        """Create train.bin and val.bin files"""
        self.download(force=force)  # Download the file, if not available
        model_path = str(self.work_dir / f"{self.tokenizer_path}.model")
        if self.tokenizer_path.startswith("tok"):
            if not os.path.exists(model_path):
                self.train_vocab(self.vocab_size)
            else:
                logger.info(f"Tokenizer already trained: {model_path}, skipping")
        self.pretokenize(model_path)

        # We set self.enc after pre-tokenization because 'coreBPE' used by tiktoken
        # cannot be pickled for multiprocessing/ProcessPoolExecutor
        self.enc = self.get_encoder(model_path)

        return True

    def get_encoder(self, model_path):
        if self.tokenizer_path == "llama2":
            enc = Tokenizer(model_path)
        elif self.tokenizer_path == "gpt2":
            enc = tiktoken.get_encoding("gpt2")
        else:  # custom tokenizer (toknnnn)
            enc = Tokenizer(model_path)
        return enc

    def load_data(self):
        """Load Data from file"""
        if self.prepared:
            model_path = str(self.work_dir / f"{self.tokenizer_path}.model")
            self.enc = self.get_encoder(model_path)
        else:
            raise NotImplementedError("Load Data not supported. Use prepare first!")

    def get_batch(self, cfg, split) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch Size = Number of sequences being processed in parallel!"""
        if self.iter_batches[split] is None:
            iter_batches = self.get_iter_batches(cfg)
            self.iter_batches[split] = iter_batches(split=split)

        batch_iter = self.iter_batches[split]
        xb, yb = next(batch_iter)
        return xb, yb

    def get_iter_batches(self, cfg):
        return partial(
            Task.iter_batches,
            batch_size=cfg.batch_size,
            max_seq_len=cfg.block_size,
            vocab_size=self.vocab_size,
            tokenizer_path=self.tokenizer_path,
            bin_dir=self.bin_dir,
            device=cfg.device,
            num_workers=0,
        )

    def encode(self, s) -> list[int]:
        """encode a string to a list of integers"""
        return self.enc.encode(s, bos=True, eos=False)

    def decode(self, l) -> str:
        """decode a list of integers back to a string"""
        return "".join(self.enc.decode(l))

    def train_vocab(self, vocab_size):
        """
        Trains a custom sentencepiece tokenizer on the TinyStories dataset.
        The custom tokenizer files will be saved in `work_dir`/tok{N} directories,
        where N is the vocab size. This is also where the pretok .bin files will go.
        """
        assert vocab_size > 0, "Vocab size must be positive"  # nosec

        start_time = time.time()

        # output file prefix path for sentencepiece
        prefix = self.work_dir / f"tok{vocab_size}"

        # how many shards we'll use for vocab training, kept low for efficiency
        num_shards = 10

        # 1) export a large chunk of text as a single text file tiny.txt
        tiny_file = self.work_dir / "tiny.txt"
        data_dir = self.work_dir / "TinyStories_all_data"
        shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))

        logger.info(
            f"Writing temporary file {tiny_file} with {num_shards} shards / {len(shard_filenames)}..."
        )
        with open(tiny_file, "w", encoding="utf-8") as of:
            for shard in tqdm(shard_filenames[:num_shards]):
                with open(shard, "r") as f:
                    data = json.load(f)
                for example in data:
                    text = example["story"]
                    text = text.strip()
                    of.write(text + "\n")
        logger.info(f"Size is: {os.path.getsize(tiny_file) / 1024 / 1024:.2f} MB")

        # 2) train the sentencepiece model
        logger.info("Will now train the vocab...")
        spm.SentencePieceTrainer.train(
            input=str(tiny_file),
            model_prefix=prefix,
            model_type="bpe",
            vocab_size=vocab_size,
            self_test_sample_size=0,
            input_format="text",
            character_coverage=1.0,
            num_threads=os.cpu_count(),
            split_digits=True,
            allow_whitespace_only_pieces=True,
            byte_fallback=True,
            unk_surface=r" \342\201\207 ",
            normalization_rule_name="identity",
        )

        os.remove(tiny_file)
        elapsed_time = time.time() - start_time
        logger.info(
            f"Trained tokenizer is in {prefix}.model, Time taken = {elapsed_time:.3f} secs"
        )

    def pretokenize(self, model_path):
        # iterate the shards and tokenize all of them one by one
        start_time = time.time()
        data_dir = self.work_dir / "TinyStories_all_data"
        shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        # process all the shards in a process pool
        fun = partial(self.process_shard, model_path=model_path)
        with ProcessPoolExecutor() as executor:
            for result in executor.map(fun, enumerate(shard_filenames)):
                pass  # print(result)
        elapsed_time = time.time() - start_time
        logger.info(f"Pretokenization Done, Time taken = {elapsed_time:.3f} secs")

    def check_file(self, shard):
        shard_basename = os.path.basename(shard)
        bin_basename = shard_basename.replace(".json", ".bin")
        tokenized_filename = os.path.join(self.bin_dir, bin_basename)
        file_exists = os.path.exists(tokenized_filename)
        return tokenized_filename, file_exists

    def process_shard(self, args, model_path):
        shard_id, shard = args
        # calculate the output filename and check if it exists
        tokenized_filename, file_exists = self.check_file(shard)
        if file_exists:
            logger.info(f"Already saved {tokenized_filename}, skipping..")
            return

        enc = self.get_encoder(model_path)
        # Actually create tokens
        with open(shard, "r") as f:
            data = json.load(f)
        all_tokens = []
        for example in tqdm(data, position=shard_id):
            text = example["story"]
            text = text.strip()  # get rid of leading/trailing whitespace
            if self.tokenizer_path == "gpt2":
                tokens = enc.encode_ordinary(text)
            else:
                tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
            all_tokens.extend(tokens)

        # convert to uint16 nparray
        all_tokens = np.array(all_tokens, dtype=np.uint16)

        # write the bytes
        with open(tokenized_filename, "wb") as f:
            f.write(all_tokens.tobytes())

        # calculate the average sequence length (they are separated by BOS=1)
        avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
        logger.info(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")


class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y


class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, tokenizer_path, bin_dir):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.tokenizer_path = tokenizer_path
        self.bin_dir = bin_dir

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        logger.info(f"Created a PretokDataset for {self.split} data with rng seed {seed}")
        shard_filenames = sorted(glob.glob(os.path.join(self.bin_dir, "*.bin")))
        # train/test split. let's use only shard 0 for test split, rest train
        shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        assert len(shard_filenames) > 0, f"No bin files found in {self.bin_dir}"  # nosec
        while True:
            rng.shuffle(shard_filenames)
            for shard in shard_filenames:
                # open the dataset for reading but keep it on disk with memmap
                m = np.memmap(shard, dtype=np.uint16, mode="r")
                num_batches = len(m) // self.max_seq_len
                num_batches -= 1  # drop the last partial batch
                assert num_batches > 0, "this shard is way too small? investigate."  # nosec
                ixs = list(range(num_batches))
                rng.shuffle(ixs)
                for ix in ixs:
                    start = ix * self.max_seq_len
                    end = start + self.max_seq_len + 1
                    # calling .astype will copy the data into a new numpy array, now in RAM
                    chunk = torch.from_numpy((m[start:end]).astype(np.int64))
                    x = chunk[:-1]
                    y = chunk[1:]
                    yield x, y


# CLI for constructing the dataset

if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    python tinystories.py download
    python tinystories.py pretokenize

    To tokenize data with a custom tokenizer we train ourselves with sentencepiece, e.g.:
    python tinystories.py download
    python tinystories.py train_vocab --vocab_size=2048
    python tinystories.py pretokenize --vocab_size=2048
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, choices=["download", "pretokenize", "train_vocab"])
    parser.add_argument("--work-dir", dest="work_dir")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument(
        "-m",
        "--model_id",
        choices=BaseLanguageModel.models(),
        default=BaseLanguageModel.default_model(),
    )
    args = parser.parse_args()

    if not args.work_dir:
        path = Path(__file__)
        root_dir = path.parent.absolute()
        args.work_dir = Path(args.work_dir) if args.work_dir else root_dir / "data"

    args.source = "t_stories"
    # depending on the stage call the appropriate function
    tiny_stories = TinyStoriesData(args)
    if args.stage == "download":
        tiny_stories.download()
    elif args.stage == "train_vocab":
        tiny_stories.train_vocab(vocab_size=tiny_stories.vocab_size)
    # elif args.stage == "pretokenize":
    #     tiny_stories.pretokenize(vocab_size=tiny_stories.vocab_size)
    else:
        raise ValueError(f"Unknown stage {args.stage}")
