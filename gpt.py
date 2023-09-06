"""Startup Script"""
import argparse
import logging
import os

from dotenv import dotenv_values, load_dotenv
from minigpt.trainer import GPTTrainer

logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s:%(message)s",
    level=logging.DEBUG,
    datefmt="%I:%M:%S",
)


def get_args():
    parser = argparse.ArgumentParser(prog="mini-gpt", description="Mini GPT")
    parser.add_argument("-m", "--model_id", type=int, default=0)
    parser.add_argument("-b", "--batch", dest="batch_size", type=int, default=4)
    parser.add_argument("-k", "--block", dest="block_size", type=int, default=8)
    parser.add_argument("-e", "--embedding", dest="n_embed", type=int, default=32)
    parser.add_argument("-l", "--layers", dest="n_layers", type=int, default=4)
    parser.add_argument("--heads", dest="n_heads", type=int, default=4)
    parser.add_argument("-i", "--iterations", dest="max_iters", type=int, default=3000)
    parser.add_argument("-r", "--learning-rate", type=float, default=1e-3)
    parser.add_argument("-d", "--dropout", type=float, default=0.2)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    args = parser.parse_args()
    # print(f"Args = {args.n_embed} % {args.n_heads} = {args.n_embed % args.n_heads}")
    if args.n_embed % args.n_heads != 0:
        print(
            f"Error: Invalid multiple of heads [{args.n_heads}] to embedding dimensions [{args.n_embed}]"
        )
        exit(-1)
    return args


def set_env(verbose=False):
    """Load Environment Variables..."""

    if verbose:
        cur_dir = os.path.abspath(os.getcwd())
        config = dotenv_values(".env")
        print(f"Current directory = {cur_dir}; Dotenv Values = {config}")

    load_dotenv(".env")


if __name__ == "__main__":
    args = get_args()
    set_env(args.verbose)
    trainer = GPTTrainer(args)
    trainer.train()
