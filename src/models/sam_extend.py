import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import random
import torchvision.transforms.functional as TF

class CataractDataset(Dataset):
    def __init__(self, csv_file, transform=None, augment=False):
        self.data = []
        self.questions = set()
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row)
                self.questions.add(row['Questions'])
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['Image_Paths']).convert('RGB')
        mask = Image.open(item['Mask_Paths']).convert('L')
        question = item['Questions']
        label = item['Labels']

        if self.augment:
            image, mask = self.apply_augmentation(image, mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask, question, label

    def apply_augmentation(self, image, mask):
        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Random rotation
        angle = random.uniform(-10, 10)
        image = TF.rotate(image, angle)
        mask = TF.rotate(mask, angle)
        
        # Random brightness adjustment
        brightness_factor = random.uniform(0.8, 1.2)
        image = TF.adjust_brightness(image, brightness_factor)
        
        return image, mask

class ImageEncoder(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            self.conv_block(3, 64, dropout_rate),
            self.conv_block(64, 128, dropout_rate),
            self.conv_block(128, 256, dropout_rate),
            self.conv_block(256, 512, dropout_rate),
            self.conv_block(512, 1024, dropout_rate),
        )
        self.fc = nn.Linear(1024 * 8 * 8, 1024)
        self.dropout = nn.Dropout(dropout_rate)

    def conv_block(self, in_channels, out_channels, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.encoder(x)
        features = x
        x = x.view(x.size(0), -1)
        x = self.dropout(self.fc(x))
        return x, features

class LLMPromptEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.fc = nn.Linear(768, 1024)

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.fc(pooled_output)

class MaskDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            self.upconv_block(1024, 512),
            self.upconv_block(512, 256),
            self.upconv_block(256, 128),
            self.upconv_block(128, 64),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.decoder(x)

class LLMSupervisedSAM(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super().__init__()
        self.image_encoder = ImageEncoder(dropout_rate)
        self.prompt_encoder = LLMPromptEncoder()
        self.mask_decoder = MaskDecoder()

    def forward(self, image, questions):
        image_features, feature_map = self.image_encoder(image)
        
        batch_size = image_features.size(0)
        prompt_features_list = []
        
        for i in range(0, len(questions), batch_size):
            batch_questions = questions[i:i+batch_size]
            batch_prompt_features = self.prompt_encoder(batch_questions)
            prompt_features_list.append(batch_prompt_features)
        
        prompt_features = torch.cat(prompt_features_list, dim=0)[:batch_size]
        
        combined_features = image_features + prompt_features
        combined_features = combined_features.view(-1, 1024, 1, 1).expand(-1, -1, 8, 8)
        mask = self.mask_decoder(combined_features + feature_map)
        return mask

def dice_coefficient(pred, target):
    smooth = 1.0
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def visualize_results(image, mask, prediction, question, output_path):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image.permute(1, 2, 0).detach().cpu())
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(image.permute(1, 2, 0).detach().cpu())
    plt.imshow(mask.squeeze().detach().cpu(), alpha=0.5, cmap='jet')
    plt.title("Segmented image")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(image.permute(1, 2, 0).detach().cpu())
    plt.imshow(prediction.squeeze().detach().cpu(), alpha=0.5, cmap='jet')
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.suptitle(f"Question: {question}")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, device, output_folder):
    train_folder = os.path.join(output_folder, "train")
    val_folder = os.path.join(output_folder, "validation")
    test_folder = os.path.join(output_folder, "test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_dice = 0
        for i, (images, masks, questions, _) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images, questions)
            loss = criterion(outputs, masks)
            dice = dice_coefficient(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_dice += dice.item()

            if i % 10 == 0:
                visualize_results(images[0], masks[0], outputs[0], questions[0], 
                                  os.path.join(train_folder, f"epoch_{epoch+1}_batch_{i}.png"))

        avg_loss = total_loss / len(train_loader)
        avg_dice = total_dice / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}")

        model.eval()
        val_loss = 0
        val_dice = 0
        with torch.no_grad():
            for i, (images, masks, questions, _) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images, questions)
                val_loss += criterion(outputs, masks).item()
                val_dice += dice_coefficient(outputs, masks).item()

                if i % 5 == 0:
                    visualize_results(images[0], masks[0], outputs[0], questions[0], 
                                      os.path.join(val_folder, f"epoch_{epoch+1}_batch_{i}.png"))

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Dice: {avg_val_dice:.4f}")

    # Test set evaluation
    model.eval()
    test_loss = 0
    test_dice = 0
    with torch.no_grad():
        for i, (images, masks, questions, _) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images, questions)
            test_loss += criterion(outputs, masks).item()
            test_dice += dice_coefficient(outputs, masks).item()
            visualize_results(images[0], masks[0], outputs[0], questions[0], 
                              os.path.join(test_folder, f"test_sample_{i}.png"))

    avg_test_loss = test_loss / len(test_loader)
    avg_test_dice = test_dice / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}, Test Dice: {avg_test_dice:.4f}")

def main(csv_file):
    output_folder = "segmentation_results"
    os.makedirs(output_folder, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Create separate datasets for train, val, and test
    train_dataset = CataractDataset(csv_file, transform=transform, augment=True)
    val_dataset = CataractDataset(csv_file, transform=transform, augment=False)
    test_dataset = CataractDataset(csv_file, transform=transform, augment=False)

    # Split the data
    train_indices, test_indices = train_test_split(range(len(train_dataset)), test_size=0.3, random_state=42)
    test_indices, val_indices = train_test_split(test_indices, test_size=0.5, random_state=42)

    # Create subset datasets
    train_data = Subset(train_dataset, train_indices)
    val_data = Subset(val_dataset, val_indices)
    test_data = Subset(test_dataset, test_indices)

    batch_size = 32 * 6  # Increase batch size to utilize all GPUs
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=12)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=12)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LLMSupervisedSAM(dropout_rate=0.2)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added L2 regularization

    num_epochs = 100
    train_and_evaluate(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, device, output_folder)

    print("Training and evaluation completed. Results saved in the 'segmentation_results' folder.")

if __name__ == "__main__":
    csv_file = "/home/chethanat/retinal_segmentation/segmentation/final_data_for_segmentation/final_dataset.csv"
    main(csv_file)