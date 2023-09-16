"""The Language Model Classes"""
import importlib
import inspect
import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from minigpt.config import ModelConfig

from torch.nn import functional as F

logger = logging.getLogger(__name__)

MODELS = {
    "bg": "BigramLanguageModel",
    "m1": "GPTLanguageModelv1",
    "m2": "GPTLanguageModelv2",
    "m3": "GPTLanguageModelv3",
    "m4": "GPTLanguageModelv4",
    "m5": "GPTLanguageModelv5",
    "m6": "GPTLanguageModelv6",
    "m7": "GPTLanguageModelv7",
    "l2": "GPTLanguageModelLlama2",
    "g2": "GPT2PretainedModel",  ## For loading pre-trained GPT2 etc.
}
DEFAULT_MODEL = "bg"


class BaseLanguageModel(nn.Module):
    """Base Model"""

    def __init__(self, cfg: "ModelConfig"):
        super().__init__()
        self.cfg = cfg

    @staticmethod
    def default_model():
        return DEFAULT_MODEL

    @staticmethod
    def models():
        return list(MODELS.keys())

    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def model_name(model_id) -> str:
        return MODELS.get(model_id, "BigramLanguageModel")

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self, "position_embedding_table"):
            # print(f"Position Embedding Count: {self.position_embedding_table.weight.numel()}")
            n_params -= self.position_embedding_table.weight.numel()
        return n_params

    def configure_optimizers(self, master_process=True):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and self.cfg.device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        betas = (self.cfg.beta1, self.cfg.beta2)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.cfg.learning_rate, betas=betas, **extra_args
        )
        # Log pertinent information
        if master_process:
            logger.info(
                f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
            )
            logger.info(
                f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
            )
            logger.info(f"using fused AdamW: {use_fused}")

        return optimizer

    def mflops_achieved(self, fwdbwd_per_iter, dt):
        """Compute the achieved mflops. Based on the GPU, we can figure out the MFU"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.cfg
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.n_embed // cfg.n_heads, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        # estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS
        # flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        # mfu = flops_achieved / flops_promised
        return flops_achieved / 1e6

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.cfg.block_size  # nosec
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """Compute loss for the predicted logits v/s the targets"""
        if targets is None:
            loss = None
        else:
            # loss = F.cross_entropy(logits, targets) # Does not work because pytorch needs B * C * T for multi-dimensional array
            # So, Reshaping so that cross_entropy works
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return loss

    def generate_text(self, tdata, num_tokens=200, start_with=None):
        print("=" * 100)
        print(f"  Generating Text [{num_tokens} tokens]")
        print("=" * 100)
        ## Create the initial 'text' to generate the continuation --> Using 0 = \n or ` start_with`
        if start_with:
            tokens = self.tdata.encode(idx)
            idx = torch.tensor(tokens, dtype=torch.long, device=self.cfg.device)
        else:
            idx = torch.zeros((1, 1), dtype=torch.long, device=self.cfg.device)
        tokens = self.generate(idx, num_tokens=num_tokens)
        print(tdata.decode(tokens[0].tolist()))
        print("=" * 100)

    @torch.no_grad()
    def generate(self, idx, num_tokens, temperature=1.0, top_k=None):
        """Generate next `num_tokens` tokens, idx --> B x T"""
        for _i in range(num_tokens):
            # print(f"Generating {i} token...")
            # crop idx to the last block size tokens [Remove extra tokens from the beginning!]
            # because positional encodings are defined only upto block_size
            idx_cond = idx[:, -self.cfg.block_size :]

            # get the predictions/losses
            logits, _loss = self(idx_cond)  ## logits --> B x T x C

            # focus on last time step (only the last character) and scale by desired temperature
            logits = logits[:, -1, :] / temperature  ## --> B x C
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Apply Softmax
            # counts = logits.exp() # counts, equivalent to N
            # probs = counts / counts.sum(1, keepdims=True) # probabilities for next character
            probs = F.softmax(logits, dim=-1)  ## --> B x C

            # Sample from the probability distribution to get the next idx.
            idx_next = torch.multinomial(probs, num_samples=1)  ## --> B x 1

            # Append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)  ## --> B x T+1
        return idx

    @staticmethod
    def get_model(cfg) -> nn.Module:
        current_package = importlib.import_module(__package__)
        cls_name = MODELS.get(cfg.model_id)
        if cls_name:
            model_cls = getattr(current_package, cls_name)
            if cfg.model_id == "g2":
                m = model_cls.from_pretrained(cfg)
            else:
                model_params = {"cfg": cfg}
                m = model_cls(**model_params)
        else:
            error_msg = f"Unknown Model ID: {cfg.model_id} - Use one of {MODELS.keys()}"
            logger.warn(error_msg)
            raise ValueError(error_msg)
        return m.to(cfg.device)
