# A file that contains settings for using the module
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#GPU = "cuda:0"
#torch.cuda.set_device(GPU)
