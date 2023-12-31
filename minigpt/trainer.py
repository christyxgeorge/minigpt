"""MiniGPT Trainer"""

import contextlib
import gc
import json
import logging
import math
import os
import time
from contextlib import nullcontext
from typing import Literal

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from minigpt.config import TrainerConfig
from minigpt.loaders.base_dataset import BaseDataset
from minigpt.models.base_model import BaseLanguageModel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity, profile, record_function

logger = logging.getLogger(__name__)
xlogger = logging.getLogger("trainlog")  # TODO: To merge logger and xlogger!

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
    trainer.train_model(use_ddp=True)


class GPTTrainer:
    def __init__(self, args, **kwargs):
        """Data Loading and Hyperparameters"""
        # the master process will do logging, checkpointing etc.
        local_rank = kwargs.get("local_rank", 0)
        world_size = kwargs.get("world_size", 1)
        self.master_process = local_rank == 0
        self.resume = args.resume

        # setup manual seed
        torch.manual_seed(TORCH_MANUAL_SEED + local_rank)
        torch.cuda.manual_seed(TORCH_MANUAL_SEED + local_rank)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

        self.tdata = BaseDataset.get_loader(args, load=True)
        self.cfg = TrainerConfig(
            **vars(args),
            vocab_size=self.tdata.vocab_size,
            local_rank=local_rank,
            world_size=world_size,
        )
        self.model = BaseLanguageModel.get_model(self.cfg)

        self.ctx, self.scaler = self.setup_ctx_and_scaler()
        self.profiler_ctx = self.setup_profiler_context()
        if self.master_process and self.cfg.verbose:
            print(self.model)
            self.print_device_info()

    @staticmethod
    def train(args):
        """Run Loop"""
        world_size = TrainerConfig.num_devices()

        if args.use_ddp:
            logger.info(f"Training using DDP: World Size = {world_size}")
            # mp.set_start_method("spawn")  # , force=True) # Not needed as we are only 'spawn'ing
            mp.spawn(
                train_fn,
                args=(args, world_size),
                nprocs=world_size,
                join=True,
            )
        else:
            logger.info("Training without DDP")
            trainer = GPTTrainer(args)
            trainer.train_model(use_ddp=False)

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

    def setup_profiler_context(self):
        """Enable Profiling only on the master process"""
        if self.cfg.profile and self.master_process:
            activites = (
                [ProfilerActivity.CPU, ProfilerActivity.CUDA]
                if self.cfg.device_type == "cuda"
                else [ProfilerActivity.CPU]
            )
            return profile(
                activities=activites,
                profile_memory=True,
                record_shapes=True,
            )
        return nullcontext()

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
    def print_estimate_loss(self, iter, eval_start_time=None, **kwargs):
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

        log_msg = f"step {iter:4d}: train loss = {xlosses['train']:.4f}, val loss = {xlosses['val']:.4f}{elapsed_str}"
        logger.info(log_msg)
        if self.cfg.verbose:
            xlogger.info(log_msg)

        if eval_start_time:
            xlosses["eval_time"] = eval_time
            xlosses["est_time"] = estimation_time
        kwargs["lr"] = kwargs.get("lr", self.cfg.learning_rate)
        self.wandb_log(iter, **xlosses, **kwargs)
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
        """
        Save the model. In case of DDP, we use the unwrapped model (model.module)
        Also, we dont save the model, unless at least 100 iterations have run
        """
        if iter > 100:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter_num": iter,
                "best_val_loss": val_loss,
                "hparams": self.cfg.hparams,
                "vocab_size": self.cfg.vocab_size,
            }
            checkpoint_dir = self.cfg.work_dir / "checkpoints"
            if self.cfg.verbose:
                logger.info(f"Saving model checkpoint @ step {iter} to {checkpoint_dir}")
            if self.cfg.verbose:
                xlogger.info(f"Saving model checkpoint @ step {iter} to {checkpoint_dir}")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)  # Create, if not exists.
            torch.save(checkpoint, checkpoint_dir / f"{model.name.lower()}.ckpt.pt")

    def init_log_file(self):
        """
        Write to a log file. This can be used to check the progress on Kaggle/Colab
        where the logger is not working with the spawned process
        """
        if self.cfg.world_size == 1:
            log_file = str(self.cfg.work_dir / "train.log")
        else:
            log_file = str(self.cfg.work_dir / f"train{self.cfg.local_rank}.log")
        if os.path.exists(log_file):
            os.remove(log_file)  # Clear old logs.
        fh = logging.FileHandler(log_file)
        log_format = "{asctime}.{msecs:03.0f} {levelname} [{name}]: {message}"
        log_style: Literal["%", "{", "$"] = "{"  # type: ignore
        fh.setFormatter(logging.Formatter(fmt=log_format, style=log_style))
        xlogger.addHandler(fh)
        xlogger.propagate = False

    def wandb_init(self, first_step=0):
        if self.cfg.is_wandb_enabled and self.master_process:
            wandb_api_key = os.environ["WANDB_API_KEY"]
            wandb.login(key=wandb_api_key)
            wandb.init(
                # set the wandb project where this run will be logged
                project="minigpt",
                id=self.cfg.run_id,
                notes=json.dumps(self.cfg.dict()),
                config=dict(**self.cfg.dict(), starting_step=first_step, resumed=self.resume),
                tags=[self.model.name],
            )

    def wandb_log(self, step, **kwargs):
        if self.cfg.is_wandb_enabled and self.master_process:
            wandb.log(kwargs, step=step)

    def wandb_finish(self):
        if self.cfg.is_wandb_enabled and self.master_process:
            wandb.finish()

    def print_training_info(self):
        flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        flash_avail = "available" if flash else "not available"
        log_msg = (
            f"Training: Model = {self.model.__class__.__name__}, "
            f"Parameter Count: {self.model.get_num_params():,}, "
            f"Device = {self.model.device}, Data Prepared: {self.tdata.prepared}, "
            f"Flash Attention is {flash_avail}"
        )
        print(log_msg)
        print(f"Model Config: {self.cfg.dict()}")
        print("=" * 100)
        ddp_str = f"{self.cfg.world_size} devices" if self.cfg.use_ddp else "False"
        trg_msg = f"Training starts [Device = {self.cfg.device_type}, DDP: {ddp_str}, Scaler = {self.scaler.is_enabled()}]"
        logger.info(trg_msg)
        if self.cfg.verbose:
            xlogger.info(log_msg)
            xlogger.info(f"Model Config: {self.cfg.dict()}")
            xlogger.info(trg_msg)

    def load_checkpoint(self):
        checkpoint_dir = self.cfg.work_dir / "checkpoints"
        ckpt_path = checkpoint_dir / f"{self.model.name.lower()}.ckpt.pt"
        # TODO: Check if checkpoint does not exist!
        checkpoint = torch.load(ckpt_path, map_location=self.cfg.device)
        return checkpoint

    def restore_model_state(self, checkpoint, optimizer):
        self.cfg.update_hparams(**checkpoint["hparams"])
        state_dict = checkpoint["model"]
        iterations = checkpoint["iter_num"]
        if iterations >= self.cfg.max_iters:
            error_msg = f"Number of iterations completed {iterations} >= Maximum iterations {self.cfg.max_iters}"
            logger.warn(error_msg)
            raise ValueError(error_msg)
        best_val_loss = checkpoint["best_val_loss"]
        logger.info(
            f"Checkpoint restored. Trained for {iterations} iterations, Loss = {best_val_loss:.4f}"
        )

        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, _v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        # Restore model and optimizer state
        self.model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint["optimizer"])
        return iterations

    def train_model(self, use_ddp=False):
        """Train the model"""
        self.wandb_init()

        # use AdamW instead of torch.optim.SGD
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)
        optimizer = self.model.configure_optimizers(master_process=self.master_process)

        self.init_log_file()

        # optional: track gradients
        # wandb.watch(self.model)
        first_step = 0
        if self.resume:
            checkpoint = self.load_checkpoint()
            first_step = self.restore_model_state(checkpoint, optimizer)
            checkpoint = None  # free memory as soon as we can

        if self.master_process:
            self.print_training_info()

        train_start_time = time.time()
        if use_ddp:
            self.setup_ddp()
            self.pre_training_ddp()
            # In case of DDP, use the unwrapped DDP model to checkpoint
            raw_model = self.model.module
            self.print_estimate_loss(first_step)  # Print Initial Losses!
            with self.profiler_ctx:
                step, last_eval_stime = self.training_loop_ddp(
                    optimizer, raw_model, first_step=first_step
                )
            self.cleanup_ddp()
        else:
            raw_model = self.model
            self.print_estimate_loss(first_step)  # Print Initial Losses!
            with self.profiler_ctx:
                step, last_eval_stime = self.training_loop_single(
                    optimizer, raw_model, first_step=first_step
                )

        if self.master_process:
            losses = self.print_estimate_loss(step + 1, eval_start_time=last_eval_stime)
            self.save_model(step + 1, raw_model, optimizer, losses["val"])
            self.log_training(train_start_time)
            print(
                f"{':'*20} Training Ended: Losses = Val: {losses['val']:.4f}, Train: {losses['train']:.4f} {':'*20}"
            )
            # Now that the profile has finished, lets print the stats
            if not isinstance(self.profiler_ctx, contextlib.nullcontext):
                topn = 10
                print(
                    self.profiler_ctx.key_averages(group_by_stack_n=topn).table(
                        sort_by="self_cpu_time_total", row_limit=topn
                    )
                )

        self.wandb_finish()

    def clear_memory(self):
        """clear the memory"""
        gc.collect()  # This takes a lot of time.
        torch.cuda.empty_cache()

    def pre_training_ddp(self):
        # compile the model
        if self.cfg.compile:
            compile_start = time.time()
            logger.info("compiling the model... (takes a ~minute)")
            # import torch._dynamo
            # torch._dynamo.config.verbose = self.cfg.verbose
            self.model = torch.compile(self.model)  # requires PyTorch 2.0
            logger.info(f"compiled the model... (took {time.time() - compile_start} secs)")

        # wrap model into DDP container
        self.model = (
            DDP(self.model, device_ids=[self.cfg.local_rank])
            if self.cfg.device_type == "cuda"
            else DDP(self.model)
        )

    def training_loop_single(self, optimizer, raw_model, first_step=0):
        eval_start_time = time.time()
        for step in range(first_step, self.cfg.max_iters):  ## `n` steps
            if step > first_step and step % self.cfg.eval_interval == 0:
                losses = self.print_estimate_loss(step, eval_start_time=eval_start_time)
                with record_function("save_model"):
                    self.save_model(step, raw_model, optimizer, losses["val"])
                # self.clear_memory() # Not needed
                eval_start_time = time.time()
            self.train_single_step(self.model, optimizer)
            if self.cfg.eval_only:
                break

        return step, eval_start_time

    def training_loop_ddp(self, optimizer, raw_model, first_step=0):
        """Run Loop with DDP - Single node"""
        running_mflops = -1

        eval_start_time = time.time()
        for step in range(first_step, self.cfg.max_iters):  ## `n` steps
            # determine and set the learning rate for this iteration
            lr = self.get_lr(step) if self.cfg.decay_lr else self.cfg.learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            if step > first_step and step % self.cfg.eval_interval == 0:
                losses = self.print_estimate_loss(step, eval_start_time=eval_start_time, lr=lr)
                if self.master_process:
                    self.save_model(step, raw_model, optimizer, losses["val"])
                # self.clear_memory() # Not needed
                eval_start_time = time.time()
            elif step > first_step and self.cfg.log_interval and step % self.cfg.log_interval == 0:
                if self.master_process and self.cfg.verbose:
                    xlogger.info(
                        f"Process [{self.cfg.local_rank}] Step {step}: Learning rate = {lr}"
                    )

            t0 = time.time()
            loss = self.train_gradient_accum(optimizer)
            dt = time.time() - t0

            if step > first_step and step % self.cfg.eval_interval == 0 and self.master_process:
                # get loss as float. note: this is a CPU-GPU sync point
                # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
                # let the training loop settle a bit [step > 0]
                lossf = loss.item() * self.cfg.gradient_accumulation_steps
                mflops_achieved = self.model.module.mflops_achieved(
                    self.cfg.batch_size * self.cfg.gradient_accumulation_steps, dt
                )
                running_mflops = (
                    mflops_achieved
                    if running_mflops == -1
                    else 0.9 * running_mflops + 0.1 * mflops_achieved
                )
                if self.cfg.is_wandb_enabled and self.master_process:
                    self.wandb_log(
                        step, mflops_achieved=mflops_achieved, running_mflops=running_mflops
                    )
                log_msg = (
                    f"iter {step}: loss {lossf:.4f}, time {dt*1000:.2f}ms, "
                    f"flops {mflops_achieved:.4f} MFlops, running mflops = {running_mflops:.4f} MFlops"
                )
                logger.info(log_msg)
                if self.cfg.verbose:
                    xlogger.info(log_msg)
            if self.cfg.eval_only:
                break
        return step, eval_start_time

    def log_training(self, start_time):  # TODO: Make it a decorator!
        """Logging Summary etc!"""
        train_time = time.time() - start_time
        print("=" * 100)
        if train_time < 60:
            logger.info(f"Time taken = {train_time:.3f} secs")
            if self.cfg.verbose:
                xlogger.info(f"Time taken = {train_time:.3f} secs")
        else:
            mins = int(train_time // 60)
            secs = train_time % 60
            logger.info(f"Time taken = {train_time:.3f} secs - {mins} mins, {secs:.3f} secs")
            if self.cfg.verbose:
                xlogger.info(f"Time taken = {train_time:.3f} secs - {mins} mins, {secs:.3f} secs")

    def train_single_step(self, model, optimizer) -> None:
        with record_function("get_batch"):
            xb, yb = self.get_batch("train")  ## xb = B x T

        with self.ctx:
            _logits, loss = self.model(xb, yb)

        # Delete the input batch tensors immediately
        del xb, yb

        ## Scale Gradients
        self.scaler.scale(loss).backward()

        # clip the gradient
        if self.cfg.grad_clip != 0.0:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)

        ## Update Optimizer, Scaler
        self.scaler.step(optimizer)
        self.scaler.update()

        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

    def train_gradient_accum(self, optimizer) -> None:
        accum_steps = self.cfg.gradient_accumulation_steps
        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(accum_steps):
            with record_function("get_batch"):
                xb, yb = self.get_batch("train")  ## xb = B x T
            if self.cfg.use_ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                self.model.require_backward_grad_sync = micro_step == accum_steps - 1
            with self.ctx:
                _logits, loss = self.model(xb, yb)
                # scale the loss to account for gradient accumulation
                loss = loss / accum_steps

            # Delete the input batch tensors immediately
            del xb, yb

            # backward pass, with gradient scaling if training in fp16
            self.scaler.scale(loss).backward()
        # clip the gradient
        if self.cfg.grad_clip != 0.0:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        # step the optimizer and scaler if training in fp16
        self.scaler.step(optimizer)
        self.scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        return loss

    def setup_ddp(self):
        # Setup MASTER_ADDR/MASTER_PORT for default init_method env://
        init_method = "env://"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(self.cfg.ddp_port)

        # init_method = f"tcp://localhost:{self.cfg.ddp_port}"

        # initialize the process group
        backend = self.cfg.ddp_device
        dist.init_process_group(
            backend=backend,
            rank=self.cfg.local_rank,
            world_size=self.cfg.world_size,
            init_method=init_method,
        )

    def cleanup_ddp(self):
        dist.destroy_process_group()
