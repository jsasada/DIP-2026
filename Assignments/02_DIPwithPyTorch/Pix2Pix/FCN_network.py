import torch
import torch.nn as nn

# 下采样模块
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.model(x)
    
# 上采样模块
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        y = self.model(x)
        return torch.cat((y, skip_input), dim=1)

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder (Convolutional Layers)
        self.down1 = DownBlock(3, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        self.down4 = DownBlock(256, 512)
        self.down5 = DownBlock(512, 512)
        self.down6 = DownBlock(512, 512)
        self.down7 = DownBlock(512, 512)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder (Deconvolutional Layers)
        self.up1 = UpBlock(512, 512, use_dropout=True)
        self.up2 = UpBlock(1024, 512, use_dropout=True)  
        self.up3 = UpBlock(1024, 512, use_dropout=True)  
        self.up4 = UpBlock(1024, 512)           
        self.up5 = UpBlock(1024, 256)              
        self.up6 = UpBlock(512, 128)                    
        self.up7 = UpBlock(256, 64)

        # Final Layer
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder forward pass
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        bottleneck = self.bottleneck(d7)

        # Decoder forward pass
        u1 = self.up1(bottleneck, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        
        return self.final_up(u7)
    