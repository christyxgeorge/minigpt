"""Startup Script"""
import argparse
import logging
import os
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values, load_dotenv
from minigpt.generator import GPTGenerator
from minigpt.loaders.loader_base import BaseDataset
from minigpt.trainer import GPTTrainer

# log_format = '%(asctime)s.%(msecs)03d %(message)s'
# log_style: Literal["%", "{", "$"] = '%'
log_format = "{asctime}.{msecs:3.0f} {levelname} [{name}]: {message}"
log_style: Literal["%", "{", "$"] = "{"
logging.basicConfig(format=log_format, level=logging.DEBUG, datefmt="%I:%M:%S", style=log_style)
# print(f"Log Level Before = {logging.getLogger('minigpt').level} // {logging.getLogger().level}")
# logging.getLogger("minigpt").setLevel(logging.DEBUG)  # logging.getLogger().level)
# print(f"Log Level After  = {logging.getLogger('minigpt').level} // {logging.getLogger().level}")
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(prog="minigpt", description="Mini GPT")
    subparsers = parser.add_subparsers(
        title="Training/Generation Options", dest="command", required=True
    )

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("-m", "--model_id", type=int, default=0)
    common_parser.add_argument(
        "-s",
        "--source",
        choices=BaseDataset.loaders(),
    )
    common_parser.add_argument("--work-dir", dest="work_dir")
    common_parser.add_argument("-v", "--verbose", action="store_true", default=False)

    # Sub-parser for getting options to download the data
    down_parser = subparsers.add_parser("download", parents=[common_parser])
    down_parser.add_argument("-f", "--force", action="store_true", default=False)

    # Sub-parser for getting options to prepare the data
    prep_parser = subparsers.add_parser("prepare", parents=[common_parser])
    prep_parser.add_argument("-f", "--force", action="store_true", default=False)

    # Sub-parser for getting options to  train
    train_parser = subparsers.add_parser("train", parents=[common_parser])
    train_parser.add_argument("-b", "--batch", dest="batch_size", type=int, default=4)
    train_parser.add_argument("-k", "--block", dest="block_size", type=int, default=8)
    train_parser.add_argument("-e", "--embedding", dest="n_embed", type=int, default=32)
    train_parser.add_argument("-l", "--layers", dest="n_layers", type=int, default=4)
    train_parser.add_argument("--heads", dest="n_heads", type=int, default=4)
    train_parser.add_argument("-i", "--iterations", dest="max_iters", type=int, default=3000)
    train_parser.add_argument("-r", "--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("-d", "--dropout", type=float, default=0.2)

    train_parser.add_argument("--wandb", dest="wandb", default="off")
    train_parser.add_argument("--no-ddp", dest="use_ddp", action="store_false", default=True)
    train_parser.add_argument("--eval-only", dest="eval_only", action="store_true", default=False)
    train_parser.add_argument("--compile", action="store_true", default=False)
    train_parser.add_argument("--decay-lr", dest="decay_lr", action="store_true", default=False)

    # Sub-parser for getting options to  generate
    gen_parser = subparsers.add_parser("generate", parents=[common_parser])
    gen_parser.add_argument("-t", "--tokens", type=int, default=1000)
    gen_parser.add_argument("--start-with", dest="start_with", default=None)

    args = parser.parse_args()
    # print(f"Args = {args}")
    command = args.command
    delattr(args, "command")  # or del args.command

    if not args.source:
        args.source = BaseDataset.default_loader()

    if command == "train" and args.n_embed % args.n_heads != 0:
        print(
            f"Error: Invalid multiple of heads [{args.n_heads}] to embedding dimensions [{args.n_embed}]"
        )
        exit(-1)
    return command, args


def set_env(verbose=False) -> None:
    """Load Environment Variables..."""

    if verbose:
        cur_dir = os.path.abspath(os.getcwd())
        config = dotenv_values(".env")
        print(f"Current directory = {cur_dir}; Dotenv Values = {config}")

    load_dotenv(".env")


if __name__ == "__main__":
    command, args = get_args()
    set_env(args.verbose)

    path = Path(__file__)
    root_dir = path.parent.absolute() / "data"
    args.work_dir = Path(args.work_dir) if args.work_dir else root_dir
    args.work_dir = args.work_dir / args.source

    ## Create directories if they dont exist
    args.work_dir.mkdir(parents=True, exist_ok=True)

    if command == "download":
        loader = BaseDataset.get_loader(args.source, args.work_dir, verbose=args.verbose)
        loader.download(args.force)
    elif command == "prepare":  ## Create train.bin, val.bin
        loader = BaseDataset.get_loader(args.source, args.work_dir, verbose=args.verbose)
        loader.prepare(args.force)
    elif command == "train":
        GPTTrainer.train(args)
    elif command == "generate":
        GPTGenerator.generate(args)
    else:
        print(
            f"Invalid Command: {command} => Should be one of `download`, `prepare`, `train` or `generate`"
        )
