# minigpt

Adapted from NanoGPT [https://github.com/karpathy/nanoGPT]

1. Restructured the code according to my learning
2. Retained multiple intermediate models as gpt1, gpt2, ..., gpt7
3. Several things remain undone. List is below
4. Added support for Spotify million songs to auto-generate lyrics!

#========================================================================================
Pending things and issues encountered

- Learning Decay
- Resume learning, checkpointing
- Load from GPT2 weights
- crop down the model block size (why?)
- Support for TPU/XLA
- Track gradients on WANDB (wandb.watch)
- wandb.watch to see if model parameters and model architecture can be seen
- Check Pytorch 2.0 compile!
- Need to check this. using fused AdamW (model_base.py / configure_optimizers)
- Verbose is not used properly all across
- DDP logs after mp.spawn is missing on kaggle!
- Sample Generation can also be moved to DDP, if possible?
- Tinystories processing.
- Async prefetch of get_batch, local_iter_num??

Need to check/understand the following

- Check torchrun. We will need torchrun if we are using across multiple nodes.
- Scaling laws Notebook
- bench.py

#========================================================================================
