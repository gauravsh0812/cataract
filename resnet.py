import os
from PIL import Image
import torch
from torchvision import models, transforms
import yaml
from box import Box
import tqdm

# reading config file
with open("config.yaml") as f:
    cfg = Box(yaml.safe_load(f))

# Accessing features from an intermediate layer for demonstration
class ResNetFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-2])  # Remove the avgpool and fc layers

    def forward(self, x):
        x = self.resnet(x)
        return x

# Define a transformation to preprocess the images
preprocess = transforms.Compose([
    transforms.Resize(256),  # Resize to 224x224
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

def load_and_preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Open and convert to RGB
    image_tensor = preprocess(image)  # Apply preprocessing
    return image_tensor

# Define the directory containing images and the directory to save tensors
image_dir = f"{cfg.dataset.path_to_data}/frames"
tensor_dir = f"{cfg.dataset.path_to_data}/resnet_frame_tensors"
os.makedirs(tensor_dir, exist_ok=True)

model = ResNetFeatureExtractor()

# Process each image in the directory
for case in tqdm.tqdm(os.listdir(image_dir)):
    os.makedirs(f"{tensor_dir}/{case}", exist_ok=True)
    for image_name in os.listdir(f"{image_dir}/{case}"):
        image_path = os.path.join(image_dir, f"{case}/{image_name}")
        tensor = load_and_preprocess_image(image_path)
        
        # Save tensor with the same name as the image but with .pt extension
        tensor_path = os.path.join(f"{tensor_dir}/{case}", f'{os.path.splitext(image_name)[0]}.pt')
        
        features = model(tensor)
        torch.save(features, tensor_path)

print("Preprocessing and saving tensors completed.")
