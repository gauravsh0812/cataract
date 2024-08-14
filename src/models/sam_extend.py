import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

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
            
            # getting RGB mask
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
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

class LLMSupervisedSAM_Extend(nn.Module):
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
