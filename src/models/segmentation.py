import torch
import torch.nn as nn
import os

class Segmentation_Model(nn.Module):
    def __init__(self, 
                 sam_extend,
                 unetpp):
        self.sam_extend = sam_extend
        self.unetpp = unetpp

    def forward(self,x):
        x = self.sam_extend(x)  # image >> RGB mask
        x = self.unetpp(x)  # RGB mask >> Binary Mask

        return x
    