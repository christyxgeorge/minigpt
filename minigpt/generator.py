"""Generate samples from the trained models"""
import logging
import os

import torch
from minigpt.config import ModelConfig
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

        self.start_with = args.start_with
        self.num_tokens = args.tokens
        self.temperature = args.temperature
        del args.start_with, args.tokens, args.temperature

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

        args.vocab_source = checkpoint.get("vocab_source", "llama2")
        self.tdata = BaseDataset.get_loader(args, load=True)
        self.cfg = ModelConfig(
            **vars(args),
            vocab_size=self.tdata.vocab_size,
        )
        self.cfg.update_hparams(**checkpoint["hparams"])
        self.model = BaseLanguageModel.get_model(self.cfg)
        self.model.load_state_dict(state_dict)
        checkpoint = None  # free memory as soon as we can

    @staticmethod
    def generate(args):
        generator = GPTGenerator(args)
        generator.generate_text(generator.start_with, generator.num_tokens)

    def load_checkpoint(self, model_id, work_dir):
        checkpoint_dir = work_dir / "checkpoints"
        model_name = BaseLanguageModel.model_name(model_id).lower()
        ckpt_path = checkpoint_dir / f"{model_name}.ckpt.pt"
        device = ModelConfig.default_device()
        checkpoint = torch.load(ckpt_path, map_location=device)
        return checkpoint

    def generate_text(self, start_with, num_tokens, _temperature):
        # Generate Text
        self.model.generate_text(
            self.tdata, num_tokens=self.num_tokens, start_with=self.start_with
        )
