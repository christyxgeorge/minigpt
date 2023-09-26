"""Startup Script"""
import argparse
import logging
import os
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values, load_dotenv
from minigpt.generator import GPTGenerator
from minigpt.loaders.base_dataset import BaseDataset
from minigpt.models.base_model import BaseLanguageModel
from minigpt.models.gpt_pretrained import PRETRAINED_MODELS
from minigpt.trainer import GPTTrainer

# log_format = '%(asctime)s.%(msecs)03d %(message)s'
# log_style: Literal["%", "{", "$"] = '%'
log_format = "{asctime}.{msecs:03.0f} {levelname} [{name}]: {message}"
log_style: Literal["%", "{", "$"] = "{"
logging.basicConfig(format=log_format, level=logging.DEBUG, datefmt="%I:%M:%S", style=log_style)

# Reduce log level for distributed -- Maybe set TORCH_DISTRIBUTED_DEBUG?
logging.getLogger("torch.distributed").setLevel(logging.WARNING)
logging.getLogger("torch.nn.parallel.distributed").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(prog="minigpt", description="Mini GPT")
    subparsers = parser.add_subparsers(
        title="Training/Generation Options", dest="command", required=True
    )

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "-m",
        "--model_id",
        choices=BaseLanguageModel.models(),
        default=BaseLanguageModel.default_model(),
    )
    common_parser.add_argument(
        "-s", "--source", choices=BaseDataset.loaders(), default=BaseDataset.default_loader()
    )
    common_parser.add_argument("-d", "--work-dir")
    common_parser.add_argument("-p", "--ddp-port", default=argparse.SUPPRESS)
    common_parser.add_argument("-v", "--verbose", action="store_true", default=False)

    # Sub-parser for getting options to download the data
    down_parser = subparsers.add_parser("download", parents=[common_parser])
    down_parser.add_argument("-f", "--force", action="store_true", default=False)

    # Sub-parser for getting options to prepare the data
    prep_parser = subparsers.add_parser("prepare", parents=[common_parser])
    prep_parser.add_argument("-f", "--force", action="store_true", default=False)

    # Sub-parser for getting options to  train [Can be generated from TrainingConfig?]
    train_parser = subparsers.add_parser("train", parents=[common_parser])
    train_parser.add_argument("--batch", dest="batch_size", type=int, default=4)
    train_parser.add_argument("--block", dest="block_size", type=int, default=8)
    train_parser.add_argument("--embed-dim", dest="n_embed", type=int, default=32)
    train_parser.add_argument("--layers", dest="n_layers", type=int, default=4)
    train_parser.add_argument("--heads", dest="n_heads", type=int, default=4)
    train_parser.add_argument("--max-iters", type=int, default=3000)
    train_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("--dropout", type=float, default=0.2)

    train_parser.add_argument("--wandb", action="store_true", default="off")
    train_parser.add_argument("--no-ddp", dest="use_ddp", action="store_false", default=True)
    train_parser.add_argument("--eval-only", action="store_true", default=False)
    train_parser.add_argument("--eval-iters", type=int, default=200)
    train_parser.add_argument("--compile", action="store_true", default=False)
    train_parser.add_argument("--profile", action="store_true", default=False)
    train_parser.add_argument("--decay-lr", action="store_true", default=False)
    train_parser.add_argument("--resume", action="store_true", default=False)
    train_parser.add_argument("--pretrained-model", default="gpt2", choices=PRETRAINED_MODELS)
    train_parser.add_argument("--log-interval", type=int, default=40)

    # Sub-parser for getting options to  generate
    gen_parser = subparsers.add_parser("generate", parents=[common_parser])
    gen_parser.add_argument("--tokens", type=int, default=1000)
    gen_parser.add_argument("--start-with", default=None)
    gen_parser.add_argument("-t", "--temperature", type=float, default=0.7)
    gen_parser.add_argument("-k", "--top-k", type=int, default=200)

    # Sub-parser for getting options to  generate
    ckp_parser = subparsers.add_parser("checkpoints", parents=[common_parser])

    args = parser.parse_args()
    # print(f"Args = {args}")
    command = args.command
    delattr(args, "command")  # or del args.command

    if command == "train" and args.n_embed % args.n_heads != 0:
        print(
            f"Error: Invalid multiple of heads [{args.n_heads}] to embedding dimensions [{args.n_embed}]"
        )
        exit(-1)
    if args.model_id in ["g2", "l2"]:
        # In case of GPT2 or Llama2 models, use only Tinystories
        args.source = "t_stories"

    if command == "train" and args.model_id == "g2" and not args.pretrained_model:
        print(f"Error: Pretrained model name not specified")
        exit(-2)

    return command, args


def set_env(verbose=False) -> None:
    """Load Environment Variables..."""

    if verbose:
        cur_dir = os.path.abspath(os.getcwd())
        config = dotenv_values(".env")
        print(f"Current directory = {cur_dir}; Dotenv Keys = {config.keys()}")

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
        loader = BaseDataset.get_loader(args)
        loader.download(args.force)
    elif command == "prepare":  ## Create train.bin, val.bin
        loader = BaseDataset.get_loader(args)
        loader.prepare(args.force)
    elif command == "train":
        GPTTrainer.train(args)
    elif command == "generate":
        GPTGenerator.generate(args)
    elif command == "checkpoints":
        GPTGenerator.checkpoints(args)
    else:
        print(
            f"Invalid Command: {command} => Should be one of `download`, `prepare`, `train` or `generate`"
        )
