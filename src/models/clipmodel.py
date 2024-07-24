import torch 
import torch.nn as nn
from PIL import Image
import yaml
from box import Box

from transformers import (
    CLIPImageProcessor, 
    CLIPVisionModel,
)

with open("config.yaml") as f:
    cfg = Box(yaml.safe_load(f))

class ClipVisionEncoder(nn.Module):
    
    def __init__(self,):
        super(ClipVisionEncoder, self).__init__()
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

    def forward(self, image_paths, device):

        _hid = list()
        for image_path in image_paths:
            image = Image.open(image_path)
            inputs = self.processor(images=image, return_tensors="pt").to(device)
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state
        
            _hid.append(last_hidden_state.squeeze(0))
        
        # hidden: (B, L, 768)
        return torch.stack(_hid).to(device)#, torch.stack(_pool).to(device)
    
class Projector(nn.Module):

    def __init__(self, num_classes):
        super(Projector, self).__init__()
        self.final_lin1 = nn.Linear(768, 512)
        self.final_lin2 = nn.Linear(512, 64)
        self.final_lin3 = nn.Linear(64, num_classes)
        self.norm1 = nn.BatchNorm1d(512)
        self.norm2 = nn.BatchNorm1d(64)
        self.nor3 = nn.BatchNorm1d(num_classes)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.norm(self.final_lin1(x)))
        x = self.gelu(self.norm(self.final_lin2(x)))
        x = self.pool(x)
        x = torch.flatten(x, -2,-1)
        x = self.gelu(self.norm(self.final_lin3(x)))

        return x   # (B,num_classes)

class Cataract_Model(nn.Module):

    def __init__(self, num_classes):

        super(Cataract_Model, self).__init__()
        self.clipenc = ClipVisionEncoder()
        
        self.projector = Projector(num_classes)
        
        for param in self.clipenc.parameters():
            param.requires_grad = False

    def forward(
            self, 
            imgs,
            device,
        ):

        encoded_imgs = self.clipenc(imgs, device)  # (B, L=w*h, dim)  
        print("encoded_imagges shape: ", encoded_imgs.shape)
        exit()

        projoutput = self.projector(encoded_imgs) # (B,num_classes)
        
        return projoutput