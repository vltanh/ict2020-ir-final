import torch
import random
import numpy as np

def set_deterministic():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_seed(seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)