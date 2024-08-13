import torch
import torch.nn as nn
import os

class Segmentation(nn.Module):
    def __init__(self, 
                 sam_extend,
                 unetpp):
        self.sam_extend = sam_extend
        self.unetpp = unetpp
