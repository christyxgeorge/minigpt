"""Generate samples from the trained models"""
import logging
import time

import torch
from minigpt.config import GeneratorConfig, TrainerConfig
from minigpt.loaders.base_dataset import BaseDataset
from minigpt.models.base_model import BaseLanguageModel

logger = logging.getLogger(__name__)

TORCH_MANUAL_SEED = 1337


class GPTGenerator:
    def __init__(self, args):
        """Initialize GPT Generator"""

        # setup manual seed
        torch.manual_seed(TORCH_MANUAL_SEED)
        torch.cuda.manual_seed(TORCH_MANUAL_SEED)
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

        checkpoint = self.load_checkpoint(args.model_id, args.work_dir)
        state_dict = checkpoint["model"]
        iterations = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
        print(
            f"Checkpoint restored. Trained for {iterations} iterations, Loss = {best_val_loss:.4f}"
        )
        unwanted_prefix = "_orig_mod."
        for k, _v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        args.vocab_source = checkpoint.get("vocab_source", args.vocab_source)
        self.gen_cfg = GeneratorConfig(**vars(args))
        self.tdata = BaseDataset.get_loader(args, load=True)
        self.train_cfg = TrainerConfig(**self.gen_cfg.common_params)
        self.train_cfg.update_hparams(**checkpoint["hparams"])  ## Vocab source is duplicate!
        self.model = BaseLanguageModel.get_model(self.train_cfg)
        self.model.load_state_dict(state_dict)
        checkpoint = None  # free memory as soon as we can

    @staticmethod
    def generate(args):
        generator = GPTGenerator(args)
        generator.generate_text(
            generator.gen_cfg.start_with, generator.gen_cfg.tokens, generator.gen_cfg.temperature
        )

    def load_checkpoint(self, model_id, work_dir):
        checkpoint_dir = work_dir / "checkpoints"
        model_name = BaseLanguageModel.model_name(model_id).lower()
        ckpt_path = checkpoint_dir / f"{model_name}.ckpt.pt"
        device = TrainerConfig.default_device()
        checkpoint = torch.load(ckpt_path, map_location=device)
        return checkpoint

    def generate_text(self, start_with, num_tokens, _temperature):
        # Generate Text
        start_time = time.time()
        self.model.generate_text(self.tdata, num_tokens=num_tokens, start_with=start_with)
        self.log_generation(start_time, num_tokens)

    def log_generation(self, start_time, num_tokens):
        """Logging Summary etc!"""
        elapsed_time = time.time() - start_time
        tokens_per_sec = num_tokens / elapsed_time
        print("=" * 100)
        if elapsed_time < 60:
            elapsed_str = f"{elapsed_time:.3f} secs"
        else:
            mins = int(elapsed_time // 60)
            secs = elapsed_time % 60
            elapsed_str = f"{mins} mins, {secs:.3f} secs"
        logger.info(f"Time taken = {elapsed_str}, Tokens/Sec = {tokens_per_sec:2f}")
