# minigpt

Adapted from NanoGPT [https://github.com/karpathy/nanoGPT]

1. Restructured the code according to my learning
2. Retained multiple intermediate models as gpt1, gpt2, ..., gpt7
3. Several things remain undone. List is below

#========================================================================================
Pending things and issues encountered

1. Learning Decay
2. Resume learning, checkpointing
3. GPT2 weights
4. crop down the model block size (why?)
5. micro_step in range(gradient_accumulation_steps)
6. grad_clip
7. running_mfu
8. creating train.bin, val.bin
9. Support for TPU/XLA
10. Track gradients on WANDB
11. Check torchrun
12. wandb.watch to see if model parameters and model architecture can be seen
13. Check Pytorch 2.0 compile!
14. using fused AdamW (model_base.py / configure_optimizers)
15. Diff in val-loss => between using DDP and no DDP [Seems to be OK now]
16. Verbose is not used properly all across
17. DDP logs from spawned processes are missing on kaggle!
18. Sample Generation can also be moved to DDP.
19. Scaling laws Notebook
20. bench.py -- to be checked

#========================================================================================
