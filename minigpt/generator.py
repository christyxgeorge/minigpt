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
        self.out_dir = args.out_dir
        self.num_tokens = args.tokens
        self.verbose = args.verbose
        self.tdata = BaseDataset.get_loader(args.source, args.data_dir, verbose=False)
        checkpoint = self.load_checkpoint(args.model_id, args.source)
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
        generator.generate_text()

    def load_checkpoint(self, model_id, source):
        out_dir = self.out_dir / source
        model_name = ModelConfig.modelname_fromid(model_id).lower()
        ckpt_path = os.path.join(out_dir, f"{model_name}.ckpt.pt")
        device = ModelConfig.default_device()
        checkpoint = torch.load(ckpt_path, map_location=device)
        return checkpoint

    def generate_text(self):
        # Generate Text
        self.model.generate_text(self.tdata, self.cfg, num_tokens=self.num_tokens)
