"""Startup Script"""
import argparse
import logging
import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from minigpt.trainer import GPTGenerator, GPTTrainer

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s:%(message)s",
    level=logging.DEBUG,
    datefmt="%I:%M:%S",
)


def get_args():
    parser = argparse.ArgumentParser(prog="minigpt", description="Mini GPT")
    subparsers = parser.add_subparsers(
        title="Training/Generation Options", dest="command", required=True
    )

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("-m", "--model_id", type=int, default=0)
    common_parser.add_argument(
        "-s", "--source", default="s_char", choices=["s_char", "s_word", "tiny_stories"]
    )
    common_parser.add_argument("-v", "--verbose", action="store_true", default=False)

    train_parser = subparsers.add_parser("train", parents=[common_parser])
    train_parser.add_argument("-b", "--batch", dest="batch_size", type=int, default=4)
    train_parser.add_argument("-k", "--block", dest="block_size", type=int, default=8)
    train_parser.add_argument("-e", "--embedding", dest="n_embed", type=int, default=32)
    train_parser.add_argument("-l", "--layers", dest="n_layers", type=int, default=4)
    train_parser.add_argument("--heads", dest="n_heads", type=int, default=4)
    train_parser.add_argument("-i", "--iterations", dest="max_iters", type=int, default=3000)
    train_parser.add_argument("-r", "--learning-rate", type=float, default=1e-3)
    train_parser.add_argument("-d", "--dropout", type=float, default=0.2)

    gen_parser = subparsers.add_parser("generate", parents=[common_parser])
    gen_parser.add_argument("-t", "--tokens", type=int, default=1000)
    args = parser.parse_args()
    # print(f"Args = {args}")
    command = args.command
    delattr(args, "command")

    if command == "train" and args.n_embed % args.n_heads != 0:
        print(
            f"Error: Invalid multiple of heads [{args.n_heads}] to embedding dimensions [{args.n_embed}]"
        )
        exit(-1)
    return command, args


def set_env(verbose=False):
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
    root_dir = path.parent.absolute()

    if command == "train":
        trainer = GPTTrainer(root_dir, root_dir, args)
        trainer.train()
        trainer.generate()
    else:
        generator = GPTGenerator(root_dir, root_dir, args)
        generator.generate()


## ========================================================================================
## Pending things
## 1. Learning Decay
## 2. Resume learning, checkpointing
## 3. GPT2 weights
## 4. crop down the model block size (why?)
## 5. micro_step in range(gradient_accumulation_steps)
## 6. grad_clip
## 7. running_mfu
## 8. creating train.bin, val.bin
## 9. Support for TPU/XLA
## 10. Track gradients on WANDB
## 11. Check torchrun
## ========================================================================================
