# minigpt

Adapted from NanoGPT [https://github.com/karpathy/nanoGPT]

1. Restructured the code according to my learning
2. Retained multiple intermediate models as gpt1, gpt2, ..., gpt7
3. Several things remain undone. List is below
4. Added support for Spotify million songs to auto-generate lyrics!

#========================================================================================
Pending things and issues encountered

- crop down the model block size (why?)
- Support for TPU/XLA --> torch_xla.multiprocessing as xmp, bfloat16 supported.
  - Number of cores, memory??
- Track gradients on WANDB (wandb.watch)
- wandb.watch to see if model parameters and model architecture can be seen
- Check Pytorch 2.0 compile! - Fails on mac with omp.h issue!
- Verbose is not used properly all across
- DDP logs after mp.spawn is missing on kaggle!
- Sample Generation/Inference to be made faster, if possible? (llama2.py)
- Async prefetch of get_batch - Is it needed?
- cfg.profile?
- TinyStories => HF Dataset has separate train/val files in .txt format. and only GPT4 data. Can we use it?
- where was it run???
- gpu_model to compute mfu properly (t4, p100, v100, a100, h100, tpu v2, v3...)
- model cropping (why do we need this, crop_block_size function in nanogpt/trainer.py)
- tdata and load_metadata irrelevant for generate/resume as params are in checkpoint metadata as well!

Need to check/understand the following

- Check torchrun. We will need torchrun if we are using across multiple nodes.
- Scaling laws Notebook
- bench.py

#========================================================================================
