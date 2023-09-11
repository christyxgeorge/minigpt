"""The Language Model Classes"""
import inspect
import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from minigpt.config import ModelConfig

from torch.nn import functional as F

logger = logging.getLogger(__name__)


class LanguageModelBase(nn.Module):
    """Base Model"""

    def __init__(self, cfg: "ModelConfig"):
        super().__init__()
        self.cfg = cfg

    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()

    @property
    def device(self):
        return next(self.parameters()).device

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

    def configure_optimizers(self, cfg, master_process=True):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and cfg.device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        betas = (cfg.beta1, cfg.beta2)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=cfg.learning_rate, betas=betas, **extra_args
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

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.cfg
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

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

    def generate_text(self, tdata, cfg, num_tokens=200, start_with=None):
        print("=" * 100)
        print(f"  Generating Text [{num_tokens} tokens]")
        print("=" * 100)
        ## Create the initial 'text' to generate the continuation --> Using 0 = \n
        idx = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
        # if start_with:
        #     num_tokens = self.tdata.encode(idx)
        #     idx = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
        # else:
        #     idx = torch.zeros((1, 1), dtype=torch.long, device=cfg.device)
        tokens = self.generate(cfg, idx, num_tokens=num_tokens)
        print(tdata.decode(tokens[0].tolist()))
        print("=" * 100)

    @torch.no_grad()
    def generate(self, cfg, idx, num_tokens, temperature=1.0, top_k=None):
        """Generate next `num_tokens` tokens, idx --> B x T"""
        for _i in range(num_tokens):
            # print(f"Generating {i} token...")
            # crop idx to the last block size tokens [Remove extra tokens from the beginning!]
            # because positional encodings are defined only upto block_size
            idx_cond = idx[:, -cfg.block_size :]

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
