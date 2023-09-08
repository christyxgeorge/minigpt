"""MiniGPT Trainer"""

import json
import logging
import math
import os
import time
from contextlib import nullcontext

import torch
import torch.multiprocessing as mp
import wandb
from minigpt.config import ModelConfig
from minigpt.loaders.base import TextDataBase
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)

TORCH_MANUAL_SEED = 1337


class GPTTrainer:
    def __init__(self, data_root, output_root, args):
        """Data Loading and Hyperparameters"""
        # Setup the manual Seed to ensure reproducability
        self.data_root = data_root
        self.output_root = output_root
        self.verbose = args.verbose
        self.tdata = TextDataBase.get_loader(args.source, data_root, verbose=self.verbose)
        self.cfg = ModelConfig(**vars(args), vocab_size=self.tdata.vocab_size)
        self.model = self.cfg.get_model()

        # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
        # note: float16 data type will automatically use a GradScaler
        dtype = (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )
        ptdtypes = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
        ptdtype = ptdtypes[dtype]
        self.ctx = (
            nullcontext()
            if self.cfg.device_type == "cpu"
            else torch.amp.autocast(device_type=self.cfg.device_type, dtype=ptdtype)
        )
        # initialize a GradScaler. If enabled=False scaler is a no-op
        scaler_enabled = torch.cuda.is_available() and dtype == "float16"
        self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    @torch.no_grad()
    def print_estimate_loss(self, iter, eval_start_time=None, master_process=True):
        xlosses = {}
        ### Put in `inference` mode [Not needed in our case, no different behaviour for our model!]
        self.model.eval()
        eval_elapsed_time = time.time() - eval_start_time if eval_start_time else None
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
            total_secs = eval_elapsed_time + estimation_time
            elapsed_str = f", elapsed time = {eval_elapsed_time:.2f} secs, est time = {estimation_time:.2f} secs [{total_secs:2f}]"
        logger.info(
            f"step {iter:4d}: train loss = {xlosses['train']}, val loss = {xlosses['val']}{elapsed_str}"
        )
        if eval_start_time:
            xlosses["eval_time"] = eval_elapsed_time
            xlosses["est_time"] = estimation_time
        xlosses[iter] = iter
        if self.cfg.wandb_log and master_process:
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

    def get_batch_bin(self, split):
        """Batch Size = Number of sequences being processed in parallel!"""

        # poor man's data loader [Need to enable this when we use train.bin, val.bin]
        # data_dir = os.path.join('data', dataset)
        # train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        # val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

        data = self.tdata.train_data if split == "train" else self.tdata.val_data
        ix = torch.randint(
            len(data) - self.cfg.block_size, (self.cfg.batch_size,)
        )  # Generate `batch_size` random offsets
        x = torch.stack(
            [data[i : i + self.cfg.block_size] for i in ix]
        )  # Each sample is stacked as a row!
        y = torch.stack([data[i + 1 : i + self.cfg.block_size + 1] for i in ix])
        if self.cfg.device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.cfg.device, non_blocking=True), y.pin_memory().to(
                self.cfg.device, non_blocking=True
            )
        else:
            x, y = x.to(self.cfg.device), y.to(self.cfg.device)
        return x, y

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

    def save_model(self, iter, optimizer, val_loss):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter_num": iter,
            "best_val_loss": val_loss,
            "config": self.cfg,
        }
        out_dir = self.output_root / "checkpoints" / self.cfg.source
        logger.info(f"Saving model checkpoint to {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)  # Create, if not exists.
        torch.save(checkpoint, os.path.join(out_dir, f"{self.model.name}.ckpt.pt"))

    def wandb_init(self, master_process):
        if self.cfg.wandb_log and master_process:
            wandb.init(
                # set the wandb project where this run will be logged
                project="minigpt",
                id=str(self.cfg),
                notes=json.dumps(self.cfg.dict()),
                # starting_step=0,
                # resumed=False,
                config=self.cfg.dict(),  # track hyperparameters and run metadata
                tags=[self.cfg.model_name],
            )

    def wandb_finish(self, master_process):
        if self.cfg.wandb_log and master_process:
            wandb.finish()

    def train(self):
        """Run Loop"""
        # is this a ddp run?
        world_size = self.cfg.device_count
        ddp = world_size > 1
        if ddp:
            train_fn = self.train_ddp
            mp.spawn(train_fn, args=(world_size,), nprocs=world_size, join=True)
        else:
            self.train_single()

    def train_single(self):
        # setup manual seed
        self.wandb_init(master_process=True)
        torch.manual_seed(TORCH_MANUAL_SEED)

        # use AdamW instead of torch.optim.SGD
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.learning_rate)

        print(
            f"Training: Model = {self.model.__class__.__name__}, Parameter Count: {self.model.get_num_params()}, Device = {self.model.device}"
        )
        print(f"Model Config: {self.cfg.dict()}")
        print("================================================================")
        logger.info("training starts [DDP: False]")
        self.print_estimate_loss(0)  # Print Initial Losses!

        # optional: track gradients
        # wandb.watch(self.model)

        start_time = time.time()
        eval_start_time = time.time()
        for step in range(self.cfg.max_iters):  ## `n` steps
            if step and step % self.cfg.eval_interval == 0:
                self.print_estimate_loss(step, eval_start_time=eval_start_time)
                eval_start_time = time.time()

            xb, yb = self.get_batch("train")  ## xb = B x T
            # print(f"Shapes: {xb.shape} / {yb.shape}")
            with self.ctx:
                _logits, loss = self.model(xb, yb)
            ## Zero out existing gradients computed for previous step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()  ## change the weights based on the gradients
            # logger.info(f"Step {step} => Loss = {loss.item()}")

        losses = self.print_estimate_loss(step + 1)
        elapsed_time = time.time() - start_time
        print("================================================================")
        if elapsed_time < 60:
            logger.info(f"Time taken = {elapsed_time:.3f} secs")
        else:
            mins = int(elapsed_time // 60)
            secs = elapsed_time % 60
            print(f"Time taken = {elapsed_time:.3f} secs - {mins} mins, {secs:.3f} secs")
        print(f"Losses: {losses}")

        self.save_model(step + 1, optimizer, losses["val"])
        self.wandb_finish(master_process=True)
        logger.info("training ends")

    def train_ddp(self, ddp_local_rank, ddp_world_size):
        """Run Loop with DDP - Single node"""
        master_process = ddp_local_rank == 0  # this process will do logging, checkpointing etc.
        self.wandb_init(master_process=master_process)
        backend = self.cfg.ddp_device
        init_process_group(backend=backend)
        device = f"cuda:{ddp_local_rank}"

        torch.cuda.set_device(device)
        seed_offset = ddp_local_rank  # each process gets a different seed

        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0  # nosec
        gradient_accumulation_steps //= ddp_world_size
        tokens_per_iter = (
            gradient_accumulation_steps * ddp_world_size * self.cfgbatch_size * self.cfg.block_size
        )
        print(f"tokens per iteration will be: {tokens_per_iter:,}")

        if master_process:
            out_dir = self.output_root / "ddp"
            os.makedirs(out_dir, exist_ok=True)
        # setup manual seed
        torch.manual_seed(TORCH_MANUAL_SEED + seed_offset)
        if self.cfg.device_type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
            torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

        # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
        # iter_num = 0
        # best_val_loss = 1e9

        # crop down the model block size if desired, using model surgery
        # if block_size < model.config.block_size:
        #     model.crop_block_size(block_size)
        #     model_args['block_size'] = block_size # so that the checkpoint will have the right value
        #     model.to(device)
        # optimizer

        optimizer = self.model.configure_optimizers(
            self.cfg.weight_decay,
            self.cfg.learning_rate,
            (self.cfg.beta1, self.cfg.beta2),
            self.cfg.device_type,
        )
        # if init_from == "resume":
        #     optimizer.load_state_dict(checkpoint["optimizer"])
        # checkpoint = None  # free up memory

        # compile the model
        if self.cfg.compile:
            print("compiling the model... (takes a ~minute)")
            model = torch.compile(model)  # requires PyTorch 2.0

        # wrap model into DDP container
        model = DDP(model, device_ids=[ddp_local_rank])

        print(
            f"Training: Model = {self.model.__class__.__name__}, Parameter Count: {self.model.get_num_params()}, Device = {self.model.device}"
        )
        print(f"Model Config: {self.cfg.dict()}")
        print("================================================================")
        logger.info("training starts [DDP: True]")
        self.print_estimate_loss(0, master_process=master_process)  # Print Initial Losses!

        # optional: track gradients
        # wandb.watch(self.model)

        start_time = time.time()
        eval_start_time = time.time()
        for step in range(self.cfg.max_iters):  ## `n` steps
            if step and step % self.cfg.eval_interval == 0:
                self.print_estimate_loss(
                    step, eval_start_time=eval_start_time, master_process=master_process
                )
                eval_start_time = time.time()

            xb, yb = self.get_batch("train")  ## xb = B x T
            # print(f"Shapes: {xb.shape} / {yb.shape}")

            with self.ctx:
                _logits, loss = self.model(xb, yb)
            # logger.info(f"Step {step} => Loss = {loss.item()}")

            ## Scale Gradients
            self.scaler.scale(loss).backward()
            ## Update Optimizer
            self.scaler.step(optimizer)
            self.scaler.update()

            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

        losses = self.print_estimate_loss(step + 1, master_process=master_process)
        elapsed_time = time.time() - start_time
        print("================================================================")
        if elapsed_time < 60:
            logger.info(f"Time taken = {elapsed_time:.3f} secs")
        else:
            mins = int(elapsed_time // 60)
            secs = elapsed_time % 60
            print(f"Time taken = {elapsed_time:.3f} secs - {mins} mins, {secs:.3f} secs")
        print(f"Losses: {losses}")

        self.save_model(step + 1, optimizer, losses["val"])
        self.wandb_finish(master_process=master_process)
        logger.info("training ends")

    def generate(self):
        # Generate Text
        self.model.generate_text(self.tdata, self.cfg)


class GPTGenerator:
    def __init__(self, data_root, output_root, args):
        torch.manual_seed(1337)
        self.data_root = data_root
        self.output_root = output_root
        self.num_tokens = args.tokens
        self.verbose = args.verbose
        self.tdata = TextDataBase.get_loader(args.source, data_root, verbose=False)
        checkpoint = self.load_checkpoint(args.model_id, args.source)
        self.cfg = checkpoint["config"]
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, _v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        self.model = self.cfg.get_model()
        self.model.load_state_dict(state_dict)

    def load_checkpoint(self, model_id, source):
        out_dir = self.output_root / "checkpoints" / source
        model_name = ModelConfig.modelname_fromid(model_id).lower()
        ckpt_path = os.path.join(out_dir, f"{model_name}.ckpt.pt")
        device = ModelConfig.default_device()
        checkpoint = torch.load(ckpt_path, map_location=device)
        return checkpoint

    def generate(self):
        # Generate Text
        self.model.generate_text(self.tdata, self.cfg, num_tokens=self.num_tokens)
