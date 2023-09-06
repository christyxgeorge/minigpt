"""MiniGPT Trainer"""

import json
import logging
import time

import torch
import wandb
from minigpt.config import ModelConfig
from minigpt.loaders.base import TextDataBase
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class GPTTrainer:
    def __init__(self, args):
        """Data Loading and Hyperparameters"""
        # Setup the manual Seed to ensure reproducability
        torch.manual_seed(1337)
        self.tdata = TextDataBase.get_loader("shakespeare_char")
        self.cfg = ModelConfig(**vars(args), vocab_size=self.tdata.vocab_size)
        self.model = self.cfg.get_model(self.tdata)

    @torch.no_grad()
    def print_estimate_loss(self, iter, elapsed_time=None):
        xlosses = {}
        ### Put in `inference` mode [Not needed in our case, no different behaviour for our model!]
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.cfg.eval_iters)
            for k in range(self.cfg.eval_iters):
                X, Y = self.get_batch(split)
                _logits, loss = self.model(self.cfg.device, X, Y)
                losses[k] = loss.item()
            xlosses[split] = losses.mean().item()
            self.model.train()

        if not elapsed_time:
            elapsed_str = ""
        else:
            elapsed_str = f", elapsed time = {elapsed_time:.2f} secs"
        logger.info(
            f"step {iter:4d}: train loss = {xlosses['train']}, val loss = {xlosses['val']}{elapsed_str}"
        )
        if elapsed_time:
            xlosses["eval_time"] = elapsed_time
        wandb.log(xlosses)
        return xlosses

    def get_batch(self, split):
        """Batch Size = Number of sequences being processed in parallel!"""
        data = self.tdata.train_data if split == "train" else self.tdata.val_data
        ix = torch.randint(
            len(data) - self.cfg.block_size, (self.cfg.batch_size,)
        )  # Generate `batch_size` random offsets
        x = torch.stack(
            [data[i : i + self.cfg.block_size] for i in ix]
        )  # Each sample is stacked as a row!
        y = torch.stack([data[i + 1 : i + self.cfg.block_size + 1] for i in ix])
        x, y = x.to(self.cfg.device), y.to(self.cfg.device)
        return x, y

    def train(self):
        """Run Loop"""
        # use AdamW instead of torch.optim.SGD
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)

        wandb.init(
            # set the wandb project where this run will be logged
            project="minigpt",
            id=str(self.cfg),
            notes=json.dumps(self.cfg.dict()),
            # starting_step=0,
            # resumed=False,
            config=self.cfg.dict(),  # track hyperparameters and run metadata
            tags=[self.cfg.model_name()],
        )

        print(f"Training: Model = {self.model.__class__.__name__}")
        print(f"Hyperparameters: {self.cfg.dict()}")
        print("================================================================")
        logger.info("training starts")
        self.print_estimate_loss(0)  # Print Initial Losses!

        # optional: track gradients
        # wandb.watch(self.model)

        start_time = time.time()
        eval_start_time = time.time()
        for step in range(self.cfg.max_iters):  ## `n` steps
            if step and step % self.cfg.eval_interval == 0:
                eval_elapsed_time = time.time() - eval_start_time
                eval_start_time = time.time()
                self.print_estimate_loss(step, eval_elapsed_time)

            xb, yb = self.get_batch("train")  ## xb = B x T
            # print(f"Shapes: {xb.shape} / {yb.shape}")
            logits, loss = self.model(self.cfg.device, xb, yb)
            ## Zero out existing gradients computed for previous step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()  ## change the weights based on the gradients
            # logger.info(f"Step {step} => Loss = {loss.item()}")

        losses = self.print_estimate_loss(step + 1)
        elapsed_time = time.time() - start_time
        print("================================================================")
        if elapsed_time < 60:
            print(f"Time taken = {elapsed_time:.3f} secs")
        else:
            mins = int(elapsed_time // 60)
            secs = elapsed_time % 60
            print(f"Time taken = {elapsed_time:.3f} secs - {mins} mins, {secs:.3f} secs")
        print(f"Hyperparameters: {self.cfg.dict()}")
        print(f"Losses: {losses}")

        wandb.finish()

        # Generate Text
        self.tdata.generate_text(self.model, self.cfg)
