import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.transforms as transforms

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.down_conv_1 = DoubleConv(3, 64)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_conv_2 = DoubleConv(64, 128)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_conv_3 = DoubleConv(128, 256)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_conv_4 = DoubleConv(256, 512)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv_1 = DoubleConv(1024, 512)

        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv_2 = DoubleConv(512, 256)

        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv_3 = DoubleConv(256, 128)

        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv_4 = DoubleConv(128, 64)

        self.output_conv = nn.Conv2d(
            in_channels=64, out_channels=3, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.down_conv_1(x)
        x2 = self.pool_1(x1)
        
        x3 = self.down_conv_2(x2)
        x4 = self.pool_2(x3)
        
        x5 = self.down_conv_3(x4)
        x6 = self.pool_3(x5)
        
        x7 = self.down_conv_4(x6)
        x8 = self.pool_4(x7)

        # Bottleneck
        x9 = self.bottleneck(x8)

        # Decoder
        x = self.up_transpose_1(x9)
        x = torch.cat([x, x7], dim=1)
        x = self.up_conv_1(x)
        
        x = self.up_transpose_2(x)
        x = torch.cat([x, x5], dim=1)
        x = self.up_conv_2(x)
        
        x = self.up_transpose_3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.up_conv_3(x)
        
        x = self.up_transpose_4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv_4(x)
        
        x = self.output_conv(x)
        
        # Apply softmax activation function along the channel dimension
        x = F.softmax(x, dim=1)
        return x
