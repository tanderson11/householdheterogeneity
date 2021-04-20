# A file that contains settings for using the module

#GPU = False

import torch
GPU = "cuda:0"
torch.cuda.set_device(GPU)
