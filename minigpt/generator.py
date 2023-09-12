"""Generate samples from the trained models"""
import logging
import os

import torch
from minigpt.config import ModelConfig
from minigpt.loaders.loader_base import BaseDataset

logger = logging.getLogger(__name__)

TORCH_MANUAL_SEED = 1337


class GPTGenerator:
    def __init__(self, args):
        torch.manual_seed(1337)
        self.num_tokens = args.tokens
        self.verbose = args.verbose
        self.tdata = BaseDataset.get_loader(args.source, args.work_dir, verbose=False, load=True)
        checkpoint = self.load_checkpoint(args.model_id, args.work_dir)
        self.cfg = checkpoint["config"]
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, _v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

        self.model = self.cfg.get_model()
        self.model.load_state_dict(state_dict)

    @staticmethod
    def generate(args):
        generator = GPTGenerator(args)
        generator.generate_text(args.start_with)

    def load_checkpoint(self, model_id, work_dir):
        checkpoint_dir = work_dir / "checkpoints"
        model_name = ModelConfig.modelname_fromid(model_id).lower()
        ckpt_path = work_dir / f"{model_name}.ckpt.pt"
        device = ModelConfig.default_device()
        checkpoint = torch.load(ckpt_path, map_location=device)
        return checkpoint

    def generate_text(self, start_with):
        # Generate Text
        self.model.generate_text(self.tdata, num_tokens=self.num_tokens, start_with=start_with)
