# minigpt

Adapted from NanoGPT [https://github.com/karpathy/nanoGPT]

1. Restructured the code according to my learning
2. Retained multiple intermediate models as gpt1, gpt2, ..., gpt7
3. Several things remain undone. List is below
4. Added support for Spotify million songs to auto-generate lyrics!

#========================================================================================
Pending things and issues encountered

- Learning Decay
- Resume learning (with local_iter_num), checkpointing (best_loss_val)
- Load from GPT2 weights
- crop down the model block size (why?)
- Support for TPU/XLA --> torch_xla.multiprocessing as xmp, bfloat16 supported.
  - Number of cores, memory??
- Track gradients on WANDB (wandb.watch)
- wandb.watch to see if model parameters and model architecture can be seen
- Check Pytorch 2.0 compile! - Fails on mac with omp.h issue!
- Need to check this. using fused AdamW (model_base.py / configure_optimizers)
- Verbose is not used properly all across
- DDP logs after mp.spawn is missing on kaggle!
- Sample Generation can also be moved to DDP, if possible? (llama2.py)
- Tinystories processing.
- Async prefetch of get_batch, local_iter_num??
- profile?
- where was it run???
- gpu_model to compute mfu properly (t4, p100, v100, a100, h100, tpu v2, v3...)

Need to check/understand the following

- Check torchrun. We will need torchrun if we are using across multiple nodes.
- Scaling laws Notebook
- bench.py

#========================================================================================
