"""MiniGPT Trainer"""

import gc
import json
import logging
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from minigpt.config import ModelConfig
from minigpt.loaders.loader_base import BaseDataset
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)

TORCH_MANUAL_SEED = 1337

# Reference for Pytorch examples including DDP...
# https://github.com/pytorch/examples


def train_fn(rank, args, wsize):
    """
    Function called by mp.spawn. Needs to be a top-level function under __main__
    """
    if args.verbose:
        print(f"Train function called for rank {rank} / {wsize}")
    trainer = GPTTrainer(args, local_rank=rank, world_size=wsize)
    trainer.train_ddp()


class GPTTrainer:
    def __init__(self, args, **kwargs):
        """Data Loading and Hyperparameters"""
        self.local_rank = kwargs.get("local_rank", 0)
        self.world_size = kwargs.get("world_size", 1)
        # the master process will do logging, checkpointing etc.
        self.master_process = self.local_rank == 0

        self.verbose = args.verbose
        self.tdata = BaseDataset.get_loader(
            args.source, args.data_dir, verbose=self.verbose, load=True
        )
        self.cfg = ModelConfig(
            **vars(args), vocab_size=self.tdata.vocab_size, local_rank=self.local_rank
        )
        self.model = self.cfg.get_model()
        self.ctx, self.scaler = self.setup_ctx_and_scaler()

        if self.master_process:
            self.print_device_info()

    @staticmethod
    def train(args):
        """Run Loop"""
        world_size = ModelConfig.num_devices()

        if args.use_ddp:
            logger.info(f"Training using DDP: World Size = {world_size}")
            # mp.set_start_method("spawn")  # , force=True) # Not needed as we are only 'spawn'ing
            mp.spawn(
                train_fn,
                args=(
                    args,
                    world_size,
                ),
                nprocs=world_size,
                join=True,
            )
        else:
            logger.info("Training without DDP")
            trainer = GPTTrainer(args)
            trainer.train_single()

    def setup_ctx_and_scaler(self):
        # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
        # note: float16 data type will automatically use a GradScaler
        dtype = (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )
        ptdtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
        ptdtype = ptdtypes[dtype]
        ctx = (
            nullcontext()
            if self.cfg.device_type == "cpu"
            else torch.amp.autocast(device_type=self.cfg.device_type, dtype=ptdtype)
        )
        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler_enabled = torch.cuda.is_available() and dtype == "float16"
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        return ctx, scaler

    def print_device_info(self):
        print("=" * 100)
        if self.cfg.device_type == "cpu":
            print(torch.__config__.parallel_info())
        elif self.cfg.device_type == "cuda":
            # torch.cuda.device(0) # -> <torch.cuda.device at 0x7efce0b03be0>
            print("CUDA Devices Info:")
            num_devices = torch.cuda.device_count()
            print(f"  Count: {num_devices}")
            print(f"  Current Device = {torch.cuda.current_device()}")
            print(f"  bfloat16 supported: {torch.cuda.is_bf16_supported()}")
            for device_id in range(num_devices):
                print(f"  Device Name[{device_id}]: {torch.cuda.get_device_name(device_id)}")
                print(f"Memory Usage [{device_id}]")
                print(
                    f"  Allocated:",
                    round(torch.cuda.memory_allocated(device_id) / 1024**3, 1),
                    "GB",
                )
                print(
                    f"  Cached:",
                    round(torch.cuda.memory_reserved(device_id) / 1024**3, 1),
                    "GB",
                )
        print("=" * 100)

    @torch.no_grad()
    def print_estimate_loss(self, iter, eval_start_time=None, lr=None, flops=0):
        xlosses = {}
        if not self.master_process:  # Nothing to do if not master process
            return xlosses

        ### Put in `inference` mode [Not needed in our case, no different behaviour for our model!]
        eval_time = time.time() - eval_start_time if eval_start_time else None
        self.model.eval()
        est_start_time = time.time()
        for split in ["train", "val"]:
            losses = torch.zeros(self.cfg.eval_iters)
            for k in range(self.cfg.eval_iters):
                X, Y = self.get_batch(split)
                with self.ctx:
                    _logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            xlosses[split] = losses.mean().item()
            self.model.train()

        estimation_time = time.time() - est_start_time
        if not eval_start_time:
            elapsed_str = f", est time = {estimation_time:.2f} secs"
        else:
            total_secs = eval_time + estimation_time
            elapsed_str = f", eval time = {eval_time:.2f} secs, est time = {estimation_time:.2f} secs [{total_secs:.2f} secs]"
        logger.info(
            f"step {iter:4d}: train loss = {xlosses['train']:.4f}, val loss = {xlosses['val']:.4f}{elapsed_str}"
        )
        if eval_start_time:
            xlosses["eval_time"] = eval_time
            xlosses["est_time"] = estimation_time
        xlosses["lr"] = lr or self.cfg.learning_rate
        xlosses["flops"] = flops
        if self.cfg.wandb_log and self.master_process:
            wandb.log(xlosses, step=iter)
        return xlosses

    def get_batch(self, split) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch Size = Number of sequences being processed in parallel!"""
        return self.tdata.get_batch(self.cfg, split)

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(self, it):
        # 1) linear warmup for warmup_iters steps
        if it < self.cfg.warmup_iters:
            return self.cfg.learning_rate * it / self.cfg.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.cfg.lr_decay_iters:
            return self.cfg.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - self.cfg.warmup_iters) / (
            self.cfg.lr_decay_iters - self.cfg.warmup_iters
        )
        assert 0 <= decay_ratio <= 1  # nosec
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.cfg.min_lr + coeff * (self.cfg.learning_rate - self.cfg.min_lr)

    def save_model(self, iter, model, optimizer, val_loss):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter_num": iter,
            "best_val_loss": val_loss,
            "config": self.cfg,
        }
        out_dir = self.cfg.out_dir / self.cfg.source
        logger.info(f"Saving model checkpoint to {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)  # Create, if not exists.
        torch.save(checkpoint, os.path.join(out_dir, f"{self.model.name}.ckpt.pt"))

    def wandb_init(self):
        if self.cfg.wandb_log and self.master_process:
            wandb_api_key = os.environ["WANDB_API_KEY"]
            wandb.login(key=wandb_api_key)
            wandb.init(
                # set the wandb project where this run will be logged
                project="minigpt",
                id=self.cfg.run_id,
                notes=json.dumps(self.cfg.dict()),
                # starting_step=0,
                # resumed=False,
                config=self.cfg.dict(),  # track hyperparameters and run metadata
                tags=[self.cfg.model_name],
            )

    def wandb_finish(self):
        if self.cfg.wandb_log and self.master_process:
            wandb.finish()

    def print_training_info(self):
        print(
            f"Training: Model = {self.model.__class__.__name__}, "
            f"Parameter Count: {self.model.get_num_params():,}, "
            f"Device = {self.model.device}, Data Prepared: {self.tdata.prepared}"
        )
        print(f"Model Config: {self.cfg.dict()}")
        print("=" * 100)
        if self.cfg.use_ddp:
            logger.info(
                f"training starts [DDP: {self.world_size} devices,  Scaler Enabled = {self.scaler.is_enabled()}]"
            )
        else:
            logger.info(
                f"training starts [DDP: False, Scaler Enabled = {self.scaler.is_enabled()}]"
            )

    def train_single(self):
        # setup manual seed
        self.wandb_init()
        torch.manual_seed(TORCH_MANUAL_SEED)

        # use AdamW instead of torch.optim.SGD
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)

        self.print_training_info()
        self.print_estimate_loss(0)  # Print Initial Losses!

        # optional: track gradients
        # wandb.watch(self.model)

        start_time = time.time()
        eval_start_time = time.time()
        for step in range(self.cfg.max_iters):  ## `n` steps
            if step and step % self.cfg.eval_interval == 0:
                self.print_estimate_loss(step, eval_start_time=eval_start_time)
                eval_start_time = time.time()
            self.train_epoch(self.model, optimizer)

        losses = self.print_estimate_loss(step + 1)
        elapsed_time = time.time() - start_time
        print("=" * 100)
        if elapsed_time < 60:
            logger.info(f"Time taken = {elapsed_time:.3f} secs")
        else:
            mins = int(elapsed_time // 60)
            secs = elapsed_time % 60
            print(f"Time taken = {elapsed_time:.3f} secs - {mins} mins, {secs:.3f} secs")
        print(f"Losses: {losses}")

        self.save_model(step + 1, self.model, optimizer, losses["val"])
        self.wandb_finish()
        logger.info("training ends")

    def train_ddp(self):
        """Run Loop with DDP - Single node"""
        self.wandb_init()
        self.setup_ddp()
        if self.cfg.device_type == "cuda":
            device = f"cuda:{self.local_rank}"
            torch.cuda.set_device(device)
            logger.info(f"[{self.local_rank}] Setting CUDA device to {device}")
        seed_offset = self.local_rank  # each process gets a different seed

        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        gradient_accumulation_steps = self.cfg.gradient_accumulation_steps
        # assert gradient_accumulation_steps % ddp_world_size == 0  # nosec
        gradient_accumulation_steps //= self.world_size
        tokens_per_iter = (
            gradient_accumulation_steps
            * self.world_size
            * self.cfg.batch_size
            * self.cfg.block_size
        )
        if self.master_process:
            logger.info(
                f"tokens per iteration will be: {tokens_per_iter:,}, gradient_accumulation_steps = {gradient_accumulation_steps}"
            )
        # setup manual seed
        torch.manual_seed(TORCH_MANUAL_SEED + seed_offset)

        if self.cfg.device_type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
            torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

        # crop down the model block size if desired, using model surgery
        # if block_size < model.config.block_size:
        #     model.crop_block_size(block_size)
        #     model_args['block_size'] = block_size # so that the checkpoint will have the right value
        #     model.to(device)

        optimizer = self.model.configure_optimizers(self.cfg, master_process=self.master_process)
        # if init_from == "resume":
        #     optimizer.load_state_dict(checkpoint["optimizer"])
        # checkpoint = None  # free up memory

        # compile the model
        if self.cfg.compile:
            print("compiling the model... (takes a ~minute)")
            model = torch.compile(self.model)  # requires PyTorch 2.0
        else:
            model = self.model

        # wrap model into DDP container
        model = (
            DDP(model, device_ids=[self.local_rank])
            if self.cfg.device_type == "cuda"
            else DDP(model)
        )
        raw_model = model.module if ddp else model  # unwrap DDP container if needed

        if self.master_process:
            self.print_training_info()
            self.print_estimate_loss(0)  # Print Initial Losses!

        # optional: track gradients
        # wandb.watch(self.model)

        # Training Loop
        train_start_time = time.time()
        eval_start_time = time.time()
        for step in range(self.cfg.max_iters):  ## `n` steps
            # determine and set the learning rate for this iteration
            lr = get_lr(step) if decay_lr else self.cfg.learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            if step and step % self.cfg.eval_interval == 0:
                self.print_estimate_loss(step, eval_start_time=eval_start_time, lr=lr, flops=flops)
                eval_start_time = time.time()

            t0 = time.time()
            self.train_gradient_accum(model, optimizer, self.cfg.gradient_accumulation_steps)
            dt = time.time() - t0

            if step and step % self.cfg.eval_interval == 0 and master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                # let the training loop settle a bit [step > 0]
                lossf = loss.item() * gradient_accumulation_steps
                flops_achieved = raw_model.flops_achieved(
                    batch_size * gradient_accumulation_steps, dt
                )
            print(
                f"iter {step}: loss {lossf:.4f}, time {dt*1000:.2f}ms, flops {flops_achieved:.2f}%"
            )
            if self.cfg.eval_only:
                break

        train_time = time.time() - train_start_time

        # Logging Summary etc
        if self.master_process:
            losses = self.print_estimate_loss(step + 1)
            print("=" * 100)
            if train_time < 60:
                logger.info(f"Time taken = {train_time:.3f} secs")
            else:
                mins = int(train_time // 60)
                secs = train_time % 60
                logger.info(f"Time taken = {train_time:.3f} secs - {mins} mins, {secs:.3f} secs")
            print(f"Losses: {losses}")

            self.save_model(step + 1, raw_model, optimizer, losses["val"])

        self.wandb_finish()
        self.cleanup_ddp()
        if self.master_process:
            logger.info("training ends")

    # def train_single_epoch(self, optimizer):
    #     xb, yb = self.get_batch("train")  ## xb = B x T

    #     with self.ctx:
    #         _logits, loss = self.model(xb, yb)
    #     loss.backward()
    #     optimizer.step()  ## change the weights based on the gradients

    #     # flush the gradients as soon as we can, no need for this memory anymore
    #     optimizer.zero_grad(set_to_none=True)

    def train_epoch(self, model, optimizer) -> None:
        xb, yb = self.get_batch("train")  ## xb = B x T

        with self.ctx:
            _logits, loss = self.model(xb, yb)

        # Delete the input batch tensors immediately
        del xb, yb

        if self.scaler.is_enabled():
            ## Scale Gradients
            self.scaler.scale(loss).backward()

            # clip the gradient
            if self.cfg.grad_clip != 0.0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)

            ## Update Optimizer, Scaler
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            optimizer.step()  ## change the weights based on the gradients

        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # Clear memory immediately
        gc.collect()
        torch.cuda.empty_cache()

    def train_gradient_accum(self, model, optimizer, accum_steps) -> None:
        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(accum_steps):
            xb, yb = self.get_batch("train")  ## xb = B x T
            if self.cfg.use_ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = micro_step == accum_steps - 1
            with self.ctx:
                logits, loss = model(xb, yb)
                # scale the loss to account for gradient accumulation
                loss = loss / accum_steps

            # Delete the input batch tensors immediately
            del xb, yb

            # backward pass, with gradient scaling if training in fp16
            self.scaler.scale(loss).backward()
        # clip the gradient
        if self.cfg.grad_clip != 0.0:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
        # step the optimizer and scaler if training in fp16
        self.scaler.step(optimizer)
        self.scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # Clear memory immediately
        gc.collect()
        torch.cuda.empty_cache()

    def setup_ddp(self):
        # Setup MASTER_ADDR/MASTER_PORT for default init_method env://
        init_method = "env://"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"

        # init_method = "tcp://localhost:12355"

        # initialize the process group
        backend = self.cfg.ddp_device
        dist.init_process_group(
            backend=backend,
            rank=self.local_rank,
            world_size=self.world_size,
            init_method=init_method,
        )

    def cleanup_ddp(self):
        dist.destroy_process_group()
