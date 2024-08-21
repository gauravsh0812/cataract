import torch, math
import torch.nn as nn
import torch.nn.functional as F
import random

# random.seed(20)
# torch.manual_seed(20)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(20)

# device = "cpu"

class CNN_Encoder(nn.Module):
    def __init__(self):
        super(CNN_Encoder, self).__init__()
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.linear = nn.Linear(512, 512)

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),   
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )


    def get_range_vector(self, size: int, device) -> torch.Tensor:
        return torch.arange(0, size, dtype=torch.long, device=device)

    def add_positional_features(
        self,
        tensor: torch.Tensor,
        min_timescale: float = 1.0,
        max_timescale: float = 1.0e4,
    ):
        _, timesteps, hidden_dim = tensor.size()

        timestep_range = self.get_range_vector(timesteps, tensor.device).data.float()
        # We're generating both cos and sin frequencies,
        # so half for each.
        num_timescales = hidden_dim // 2
        timescale_range = self.get_range_vector(
            num_timescales, tensor.device
        ).data.float()

        log_timescale_increments = math.log(
            float(max_timescale) / float(min_timescale)
        ) / float(num_timescales - 1)
        inverse_timescales = min_timescale * torch.exp(
            timescale_range * -log_timescale_increments
        )

        # Broadcasted multiplication - shape (timesteps, num_timescales)
        scaled_time = timestep_range.unsqueeze(1) * inverse_timescales.unsqueeze(0)
        # shape (timesteps, 2 * num_timescales)
        sinusoids = torch.randn(
            scaled_time.size(0), 2 * scaled_time.size(1), device=tensor.device
        )
        sinusoids[:, ::2] = torch.sin(scaled_time)
        sinusoids[:, 1::2] = torch.cos(scaled_time)
        if hidden_dim % 2 != 0:
            # if the number of dimensions is odd, the cos and sin
            # timescales had size (hidden_dim - 1) / 2, so we need
            # to add a row of zeros to make up the difference.
            sinusoids = torch.cat(
                [sinusoids, sinusoids.new_zeros(timesteps, 1)], 1
            )
        return tensor + sinusoids.unsqueeze(0)

    def forward(self, x):

        x = self.layer1(x)
        enc1, ind1 = self.pool(x)  # 64 channels

        x = self.layer2(enc1)
        enc2, ind2 = self.pool(x)  # 128 channels

        x = self.layer3(enc2)
        enc3, ind3 = self.pool(x)  # 256 channels

        x = self.layer4(enc3)
        enc4, ind4 = self.pool(x)  # 512 channels

        _shape = x.shape

        x = torch.flatten(x, 2, -1)  # (B, 512, L=H*W)
        x = x.permute(0, 2, 1)  # (B, L, 512)
        x += self.add_positional_features(x)  # (B, L, 512)

        return x, enc1, enc2, enc3, _shape

class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, model_dimension)  # (max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, model_dimension, 2).float()
            * (-math.log(10000.0) / model_dimension)
        )  # ([model_dim//2])
        pe[:, 0::2] = torch.sin(position * div_term)  # (max_len, model_dim//2)
        pe[:, 1::2] = torch.cos(position * div_term)  # (max_len, model_dim//2)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, model_dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # x: (max_len, B, embed_dim)
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)

class Transformer_Encoder(nn.Module):
    def __init__(
        self,
        dec_hid_dim,
        nheads,
        dropout,
        device,
        n_xfmer_encoder_layers,
        dim_feedfwd,
    ):
        super(Transformer_Encoder, self).__init__()
        self.dec_hid_dim = dec_hid_dim
        self.device = device
        self.pos = PositionalEncoding(dec_hid_dim, dropout, max_len=1830)

        """
        NOTE:
        nn.TransformerDecoderLayer doesn't have 'batch_first' argument anymore.
        Therefore, the sequences will be in the shape of (max_len, B)
        """
        xfmer_enc_layer = nn.TransformerEncoderLayer(
            d_model=dec_hid_dim,
            nhead=nheads,
            dim_feedforward=dim_feedfwd,
            dropout=dropout,
        )

        self.xfmer_encoder = nn.TransformerEncoder(
            xfmer_enc_layer, num_layers=n_xfmer_encoder_layers
        )

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, src_from_cnn):
        # src_from_cnn: (B, L, dec_hid_dim)
        # change the L=H*W to max_len
        src_from_cnn = src_from_cnn.permute(0, 2, 1)  # (B, dec_hid_dim, L)
        src_from_cnn = src_from_cnn.permute(
            2, 0, 1
        )  # (L, B, dec_hid_dim)

        # embedding + normalization
        """
        no need to embed as src from cnn already has dec_hid_dim as the 3rd dim
        """
        src_from_cnn *= math.sqrt(
            self.dec_hid_dim
        )  # (L, B, dec_hid_dim)

        # adding positoinal encoding
        pos_src = self.pos(src_from_cnn)  # (L, B, dec_hid_dim)

        # xfmer encoder
        self.generate_square_subsequent_mask(pos_src.shape[0]).to(
            self.device
        )
        xfmer_enc_output = self.xfmer_encoder(
            src=pos_src, mask=None
        )  # (L, B, dec_hid_dim)

        return xfmer_enc_output

class UNetDecoder(nn.Module):
    def __init__(self):
        super(UNetDecoder, self).__init__()
        
        # Transposed Convolutional layers for upsampling
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # Upsample to [5, 256, 60, 60]
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # Upsample to [5, 128, 120, 120]
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)   # Upsample to [5, 64, 240, 240]
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)    # Upsample to [5, 32, 480, 480]

        # Convolutional layers with BatchNorm after upsampling
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)  # Final layer with 1 output channel

    def forward(self, x, enc1, enc2, enc3):
        # Layer 1: Upsample + Skip Connection from enc3
        x = self.upconv1(x)  # Upsample to [5, 256, 60, 60]
        x = torch.nn.functional.interpolate(x, size=enc3.shape[2:])  # Align spatial dimensions to [5, 256, 30, 30]
        x = torch.cat([x, enc3], dim=1)  # Concatenate with encoder output
        x = self.conv1(x)  # Apply convolution to reduce channels

        # Layer 2: Upsample + Skip Connection from enc2
        x = self.upconv2(x)  # Upsample to [5, 128, 120, 120]
        x = torch.nn.functional.interpolate(x, size=enc2.shape[2:])  # Align spatial dimensions to [5, 128, 61, 61]
        x = torch.cat([x, enc2], dim=1)  # Concatenate with encoder output
        x = self.conv2(x)  # Apply convolution to reduce channels

        # Layer 3: Upsample + Skip Connection from enc1
        x = self.upconv3(x)  # Upsample to [5, 64, 240, 240]
        x = torch.nn.functional.interpolate(x, size=enc1.shape[2:])  # Align spatial dimensions to [5, 64, 122, 122]
        x = torch.cat([x, enc1], dim=1)  # Concatenate with encoder output
        x = self.conv3(x)  # Apply convolution to reduce channels

        # Final upsampling to reach the original input size
        x = torch.nn.functional.interpolate(x, size=(244, 244))  # Align spatial dimensions to [5, 64, 244, 244]
        x = self.final_conv(x)  # Output [5, 1, 244, 244]

        return x


class ReCal(nn.Module):
    # Chethana
    pass

class Segmentation_Model(nn.Module):
    def __init__(self,device):
        super(Segmentation_Model, self).__init__()  
        self.cnn = CNN_Encoder()
        self.xfmer = Transformer_Encoder(dec_hid_dim=512, 
                                         nheads=4,
                                         dropout=0.1,
                                         device=device,
                                         n_xfmer_encoder_layers=8,
                                         dim_feedfwd=1024,)
        
        self.up = UNetDecoder()

        self.init_weights()

    def init_weights(self):
        """
        initializing the model wghts with values
        drawn from normal distribution.
        else initialize them with 0.
        """
        parameters = [self.cnn.named_parameters(),
                      self.xfmer.named_parameters(),
                      self.up.named_parameters()]
        for parameter in parameters:
            for name, param in parameter:
                if "nn.Conv2d" in name or "nn.Linear" in name:
                    if "weight" in name:
                        nn.init.normal_(param.data, mean=0, std=0.1)
                    elif "bias" in name:
                        nn.init.constant_(param.data, 0)
                elif "nn.BatchNorm2d" in name:
                    if "weight" in name:
                        nn.init.constant_(param.data, 1)
                    elif "bias" in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, x):
        x, enc1, enc2, enc3,_shape = self.cnn(x)   # (B, L, 512)   
        B,D,W,H = _shape
        x = self.xfmer(x).permute(1,0,2)  # (B, L, 512)
        x = x.permute(0,2,1)  # (B, 512, L)
        x = x.view(B,D,W,H) # (B, 512, W, H)             
        x = self.up(x, enc1, enc2, enc3)

# cm = Cataract_Model()
# x = torch.rand((5, 3, 244, 244))
# cm(x)
